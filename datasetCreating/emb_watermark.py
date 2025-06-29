import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import json
import os

def adjust_brightness(image, target_brightness=128):
    current_brightness = np.mean(image)
    adjustment = target_brightness - current_brightness
    image = np.clip(image + adjustment, 0, 255)
    return image.astype(np.uint8)

def embed_watermark(cover_image_path, secret_image_path, mask_image_path, output_image_path, size=224, bbox=None, weight=0.1):
    image1 = Image.open(cover_image_path).convert('RGB')
    image1 = image1.resize((size, size))

    image2 = Image.open(secret_image_path).convert('RGB')
    h_, w_ = image2.size

    image2 = image2.resize((int(size), int(size)))

    mask = Image.open(mask_image_path)
    mask = mask.resize((int(size), int(size)))

    scale = h_ / image2.size[0]
    carrier_width, carrier_height = image1.size
    x1, y1, x2, y2 = [int(x / scale) for x in bbox]
    watermark_width, watermark_height = x2 - x1, y2 - y1

    if watermark_width > carrier_width or watermark_height > carrier_height:
        print("水印尺寸大于载体图片，无法嵌入。")
        return None

    image1_np = np.array(image1)
    image2_np = np.array(image2)[y1:y2, x1:x2, :]
    mask_np = np.array(mask)[y1:y2, x1:x2]
    mask_np = (mask_np < 150).astype(np.uint8)

    start_x = random.randint(0, carrier_width - watermark_width)
    start_y = random.randint(0, carrier_height - watermark_height)

    target_brightness = np.mean(image1_np)

    image1_norm = image1_np / 255.0
    image2_norm = image2_np / 255.0

    fused_image = image1_norm.copy()

    mask_positions = np.where(mask_np == 1)  # 获取所有掩码为1的位置
    for i, j in zip(mask_positions[0], mask_positions[1]):

        fused_image[start_y + i, start_x + j, :] = ((1 - weight) * fused_image[start_y + i, start_x + j, :] +
                                                    weight * image2_norm[i, j, :])

    fused_image = (fused_image * 255).astype(np.uint8)

    fused_image = adjust_brightness(fused_image, target_brightness)
    fused_image_pil = Image.fromarray(fused_image)
    fused_image_pil.save(output_image_path)
    # print('成功保存图片:', output_image_path)
    return [start_x, start_y, start_x + watermark_width, start_y + watermark_height]


if __name__ == '__main__':
    cover_dir = r"/home/aiseon/storage1/dataset/WQ-Bench/train2017"
    cover_json_path = r"/home/aiseon/storage1/dataset/WQ-Bench/annotations/bbox_scale_out_224.json"

    secret_dir = '/home/aiseon/storage1/dataset/WQ-Bench/logo'
    secret_mask_dir = '/home/aiseon/storage1/dataset/WQ-Bench/mask'
    secret_json_path = '/home/aiseon/storage1/dataset/WQ-Bench/secret_bbox.json'

    secret_cover_map_path = '/home/aiseon/storage1/dataset/WQ-Bench/stego_secret_map.txt'

    save_dir = r'/home/aiseon/storage1/dataset/WQ-Bench/stego'
    save_bbox_json = r'/home/aiseon/storage1/dataset/WQ-Bench/stego_bbox.json'

    with open(cover_json_path, 'r') as f:
        cover_bbox_ann = json.load(f)
    f.close()

    with open(secret_json_path, 'r') as f:
        secret_bbox_ann = json.load(f)
    f.close()

    with open(secret_cover_map_path, 'r') as f:
        secret_cover_l = f.readlines()
        secret_cover_l = [x.strip() for x in secret_cover_l]
        secret_cover_map = {}
        for i in secret_cover_l:
            cover_, secret_ = i.split('\t')
            secret_cover_map[cover_] = secret_
    f.close()

    os.makedirs(save_dir, exist_ok=True)

    cover_l = os.listdir(cover_dir)
    memory = {}

    start = 0
    end = 0.25
    # num_points = 5
    # step = (end - start) / (num_points - 1)
    # weights = [round(start + i * step, 2) for i in range(num_points)]

    weights = np.random.uniform(start, end, len(cover_l))
    weights = np.round(weights, 2)

    for idx, name in enumerate(tqdm(cover_l)):
        cover_bbox = cover_bbox_ann[name]
        weight = float(weights[idx])
        cover_path = os.path.join(cover_dir, name)
        secret_path = os.path.join(secret_dir, secret_cover_map[name])
        cover_base = name.split('.')[0]
        save_stego_path = os.path.join(save_dir, f'{cover_base}_{weight}.png')

        secret_bbox = secret_bbox_ann[secret_cover_map[name]] # --> [x1, y1, x2, y2]
        secret_mask_path = os.path.join(secret_mask_dir, secret_cover_map[name])

        stego_bbox = embed_watermark(cover_path, secret_path, secret_mask_path,
                                      save_stego_path, size=224, bbox=secret_bbox['bbox'][0], weight=weight)

        score = round((5 - 20 * weight), 2)
        memory[f'{cover_base}_{weight}.png'] = {
            'secret_image':secret_cover_map[name],
            'score': score,
            'cover_bbox':{},
            'secret_bbox':{}
        }

        memory[f'{cover_base}_{weight}.png']['cover_bbox'] = cover_bbox
        memory[f'{cover_base}_{weight}.png']['secret_bbox'][secret_bbox['label']] = [stego_bbox]

    with open(save_bbox_json, 'w') as json_file:
        json.dump(memory, json_file, indent=4)


