import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, cv2

def read_bbox_to_json(img_path, json_path, out_json):
    with open(json_path) as annos:
        annotation_json = json.load(annos)

    print('the annotation_json num_key is:', len(annotation_json))
    print('the annotation_json key is:', annotation_json.keys())
    print('the annotation_json num_images is:', len(annotation_json['images']))

    name_map = {}
    memory_bbox = {}
    for i in annotation_json['images']:
        name_map[i['file_name']] = i['id']
    for image_name in tqdm(os.listdir(img_path)):
        id = name_map[image_name]
        memory_bbox[image_name] = {}
        num_bbox = 0

        category_id_map = {}
        category = annotation_json['categories']
        for idx, i in enumerate(category):
            category_id_map[i['id']] = idx
        for i in range(len(annotation_json['annotations'][::])):
            if annotation_json['annotations'][i]['image_id'] == id:
                num_bbox = num_bbox + 1
                category_id = annotation_json['annotations'][i - 1]['category_id']
                category_name = category[category_id_map[category_id]]['name']
                # category_l.append(category[category_id_map[category_id]]['name'])
                x, y, w, h = annotation_json['annotations'][i - 1]['bbox']
                # image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
                if category_name not in memory_bbox[image_name]:
                    memory_bbox[image_name][category_name] = []
                memory_bbox[image_name][category_name].append([x, y, x+w, y+h])

    with open(out_json, 'w') as json_file:
        json.dump(memory_bbox, json_file, indent=4)

def read_bbox_scale_to_json(train_json, bbox_json, save_json, resize=224):
    with open(train_json) as annos:
        annotation_json = json.load(annos)
    with open(bbox_json) as annos:
        bbox_json = json.load(annos)
    name_map = {}
    for idx, i in enumerate(annotation_json['images']):
        name_map[i['file_name']] = idx

    memory_bbox = bbox_json.copy()
    for i in memory_bbox:
        h, w = annotation_json['images'][name_map[i]]['height'], annotation_json['images'][name_map[i]]['width']
        scale_h, scale_w = resize/h, resize/w
        for j in memory_bbox[i]:
            bbox_l = memory_bbox[i][j]
            for k in range(len(bbox_l)):
                bbox = bbox_l[k]
                scales = [scale_w, scale_h, scale_w, scale_h]
                memory_bbox[i][j][k] = [a * b for a, b in zip(bbox, scales)]
    with open(save_json, 'w') as json_file:
        json.dump(memory_bbox, json_file, indent=4)

if __name__ == "__main__":
    img_path = r"D:\guangming\dataset\coco\train2017"
    train_json = r"D:\guangming\dataset\coco\annotations\instances_train2017.json"
    bbox_json = r"D:\guangming\dataset\coco\annotations\bbox_out.json"
    bbox_scale_json = r"D:\guangming\dataset\coco\annotations\bbox_scale_out_224.json"
    # step 1
    read_bbox_to_json(img_path, train_json, bbox_json)

    # step 2
    read_bbox_scale_to_json(train_json, bbox_json, bbox_scale_json, resize=224)