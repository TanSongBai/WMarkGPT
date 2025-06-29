import os
import cv2
import numpy as np
import json
from tqdm import tqdm

mask_dir = r'/home/aiseon/storage1/dataset/WQ-Bench/mask'
mask_bbox_save = r'/home/aiseon/storage1/dataset/WQ-Bench/bbox'
json_save = r'/home/aiseon/storage1/dataset/WQ-Bench/secret_bbox_json.json'
os.makedirs(mask_bbox_save, exist_ok=True)
mask_l = os.listdir(mask_dir)

edge_percentage = 0.05

memory = {}
for mask_path in tqdm(mask_l):
    memory[mask_path] = {'label': mask_path.split('_')[0], 'bbox': []}
    mask = cv2.imread(os.path.join(mask_dir, mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 150).astype(np.uint8)  # 二值化掩码
    h, w = mask.shape
    coordinates = np.column_stack(np.where(mask == 0))

    # 按距离中心点排序
    dist_from_center = (coordinates[:, 0] - h / 2) ** 2 + (coordinates[:, 1] - w / 2) ** 2
    sorted_indices = np.argsort(dist_from_center)
    coordinates = coordinates[sorted_indices]

    num_points = len(coordinates)
    if num_points > 0:
        start_idx = int(num_points * edge_percentage)
        end_idx = int(num_points * (1 - edge_percentage))

        valid_coordinates = coordinates[start_idx:end_idx]
        y = valid_coordinates[:, 0]
        x = valid_coordinates[:, 1]
        # 计算矩形边界
        if valid_coordinates.size > 0:
            min_x, max_x = x.min(), x.max()
            min_y, max_y = y.min(), y.max()
            bbox = [int(min_x), int(min_y), int(max_x), int(max_y)]
            memory[mask_path]['bbox'].append(bbox)
            mask = mask * 255
            cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), (0), 2)  # 黑色的线条，2是线条宽度

    # cv2.imwrite(os.path.join(mask_bbox_save, mask_path), mask)
with open(json_save, 'w') as json_file:
    json.dump(memory, json_file, indent=4)
