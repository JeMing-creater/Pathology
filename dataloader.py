import os
import cv2
import time
import math
import yaml
import monai
import torch
import numpy as np
import pandas as pd
from PIL import Image
from monai.data import Dataset
from easydict import EasyDict
from openslide import open_slide
from torchvision import transforms
from histolab.slide import Slide
from histolab.tiler import GridTiler
from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImage, Resize, Compose, ToTensor



def load_tcga_clinical_data(tsv_path):
    # 读取TSV文件
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)  # 以字符串形式读取所有列，防止数据格式问题
    
    # 确保case_submitter_id列存在
    if 'case_submitter_id' not in df.columns:
        raise KeyError("Column 'case_submitter_id' not found in the TSV file.")
    
    # 构建嵌套字典
    clinical_dict = {
        row['case_submitter_id']: {col: row[col] for col in df.columns if col != 'case_submitter_id'}
        for _, row in df.iterrows()
    }
    
    return clinical_dict

def find_image_paths(root_dir, clinical_dict):
    # 遍历 root_dir 下的所有目录及文件
    for id_folder in os.listdir(root_dir):
        id_folder_path = os.path.join(root_dir, id_folder)
        if os.path.isdir(id_folder_path):  # 确保是目录
            for file in os.listdir(id_folder_path):
                if file.endswith(".svs"):
                    for case_id in clinical_dict.keys():
                        if case_id in file:
                            clinical_dict[case_id]['image_path'] = os.path.join(id_folder_path, file)
                            break
    
    # no file path for this case id.
    cases_to_remove = [case_id for case_id, data in clinical_dict.items() if 'image_path' not in data]
    for case_id in cases_to_remove:
        print(f"Warning: {case_id} has no file!")
        del clinical_dict[case_id]
    
    return clinical_dict

# def extract_patches(image_path, processed_dir, patch_size=(256, 256), magnification=20):
#     processed_path = os.path.join(os.path.dirname(processed_dir), image_path.split('/')[-1].replace('.svs', ''))
#     os.makedirs(processed_path, exist_ok=True)
#     slide = Slide(image_path, processed_path=processed_path)
#     tiler = GridTiler(tile_size=patch_size, level=0, check_tissue=True)
#     tiles = list(tiler.extract(slide))
#     patches = [np.array(tile.image).transpose(2, 0, 1) for tile in tiles]
    
#     return torch.tensor(np.stack(patches)) if patches else torch.empty(0, 3, *patch_size)  # (n, 3, 256, 256)



def calculate_target_level(slide, magnification):
    base_mpp_x = float(slide.properties.get('openslide.mpp-x', 1.0))
    desired_mpp = 10.0 / magnification  # Assuming 10 um/px at 1x magnification
    target_level = min(slide.level_count - 1, int(math.log2(base_mpp_x / desired_mpp)))
    return max(0, target_level)


def detect_non_empty_patch(patch):
    # Convert patch to grayscale
    gray = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2GRAY)
    
    # Threshold the image to binary (black and white)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels (non-empty pixels)
    non_zero_pixels = cv2.countNonZero(binary)
    
    return non_zero_pixels > 0

def extract_patches_from_image(image_path, patch_size=(256, 256), target_level=0):
    slide = open_slide(image_path)
    # level_count = slide.level_count
    
    # Calculate the target level based on the desired magnification
    target_level = target_level
    
    # Get the dimensions at the target level
    width, height = slide.level_dimensions[target_level]
    
    patches = []
    
    # Iterate over the image in steps of patch_size
    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            # Read the region from the slide
            location = (x, y)
            region = slide.read_region(location, target_level, patch_size).convert("RGB")
            
            # Check if the patch is non-empty
            if detect_non_empty_patch(region):
                # Convert to numpy array and transpose to (C, H, W)
                patch_array = np.array(region).transpose(2, 0, 1)
                patches.append(patch_array)
    
    if not patches:
        return torch.empty(0, 3, *patch_size)
    
    # Convert list of numpy arrays to a single PyTorch tensor
    patches_tensor = torch.tensor(np.stack(patches)).float() / 255.0  # Normalize pixel values
    
    return patches_tensor  # Shape: (n, 3, 256, 256)


def encode_label1(label):
    label_map = {'M0': 0, 'M1': 1}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

def encode_label2(label):
    label_map = {'N0': 0, 'N1': 1, 'NX': 2}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

def encode_label3(label):
    label_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

def encode_label4(label):
    label_map = {'T1': 0, 'T1a': 1, 'T1b': 2, 'T2': 3, 'T2a': 4, 'T2b': 5, 'T3': 6, 'T3a': 7, 'T3b': 8}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

class TCGAImageDataset(Dataset):
    def __init__(self, image_size, processed_dir, clinical_dict, target_level=0):
        self.data = []
        self.image_size = image_size
        self.processed_dir = processed_dir
        self.target_level = target_level
        for case_id, info in clinical_dict.items():
            if 'image_path' in info and 'ajcc_pathologic_stage' in info:
                self.data.append((info['image_path'], info['ajcc_pathologic_m'], info['ajcc_pathologic_n'], info['ajcc_pathologic_stage'], info['ajcc_pathologic_t']))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, label1, label2, label3, label4 = self.data[index]
        image_tensor = extract_patches_from_image(image_path, 
                                       patch_size=self.image_size,
                                       target_level=self.target_level
                                       )
        label1_tensor = encode_label1(label1)
        label2_tensor = encode_label2(label2)
        label3_tensor = encode_label3(label3)
        label4_tensor = encode_label4(label4)
        return image_tensor, label1_tensor, label2_tensor, label3_tensor, label4_tensor

def get_TCGA_dataloader(config):
    root = config.loader.root
    for project in root.keys():
        root_dir = root[project]
        image_size = config.loader.image_size[project]
        processed_dir = config.loader.processed_dir[project]
        target_level = config.loader.target_level[project]
        if project == 'KRIC':
            # load tsv
            tsv_path = root_dir + '/' + 'clinical.tsv'
            clinical_dict = load_tcga_clinical_data(tsv_path)
            # get image path
            clinical_dict = find_image_paths(root_dir, clinical_dict)
                
            dataset = TCGAImageDataset(image_size, processed_dir, clinical_dict, target_level)
    
    train_loader = monai.data.DataLoader(dataset, num_workers=config.loader.num_workers,
                                         batch_size=config.trainer.batch_size, shuffle=True)
    
    return train_loader
    
if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    # 创建数据集
    train_loader = get_TCGA_dataloader(config)  
    
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
    
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time  # Calculate the elapsed time in seconds
    elapsed_time_minutes = elapsed_time_seconds / 60  # Convert to minutes
    print(f"Execution Time: {elapsed_time_minutes:.2f} minutes")
    
    