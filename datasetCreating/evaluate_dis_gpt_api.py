# 单张图像调用openai-sb输出文本

import os
import os.path as osp
import base64
import json
import textwrap
from tqdm import tqdm
import requests
import random
import time


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def map_quality_label(score):
    if 0.0 <= score < 1.0:
        return 'obvious'
    elif 1.0 <= score < 2.0:
        return 'visible'
    elif 2.0 <= score < 3.0:
        return 'moderate'
    elif 3.0 <= score < 4.0:
        return 'faint'
    elif 4.0 <= score <= 5.0:
        return 'invisible'


def gpt(url, headers, proxies, data_root, lenth):
    json_path = osp.join(data_root, 'stego_bbox.json')
    stego_root = osp.join(data_root, 'stego')
    save_root = osp.join(data_root, 'stego_gpt')
    os.makedirs(save_root, exist_ok=True)

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    paths = [x for x in json_data]
    random.shuffle(paths)

    for path in tqdm(paths[:lenth]):
        image_path = osp.join(stego_root, path)
        text_path = osp.join(save_root, path[:-3] + 'txt')
        if osp.exists(text_path):
            print('exists............', text_path)
            continue

        assert os.path.exists(image_path)
        base64_image = encode_image(image_path)

        cover_bbox = json_data[path]['cover_bbox']
        secret_bbox = json_data[path]['secret_bbox']

        quality_score = json_data[path]['score']
        quality_label = map_quality_label(quality_score)

        sys_prompts = 'You are a helpful watermark discriminator.'

        user_prompts = textwrap.dedent(f"""The image I provided has a watermark embedded in it. The image contains the following objects with their bounding box coordinates:
        {str(cover_bbox)}
        The watermark is embedded in the following areas:
        {str(secret_bbox)} (The visibility of watermark is {quality_label})
        In the input:
        - [x1, y1, x2, y2] represents the bounding box coordinates for detected objects in the image.
        - The watermark visibility is described with the following terms: invisible, faint, moderate, visible and obvious. The more obvious the watermark, the worse the watermark embedding effect.

        First, describe the specific orientation of the watermark in the image. Next, explain the relative position of the watermark to the objects in the image, indicating whether the watermark is inside, partially overlapping, or near the edge of the objects. Avoid mentioning any bounding box or coordinate information in your response. Then, briefly describe the watermark's characteristics, such as appearance or texture. Finally, assess the watermark visibility using the format: 'The watermark visibility is: [invisible/faint/moderate/visible/obvious]'.
        Present all this information in a cohesive and concise paragraph.""").replace("        ", " ")

        data = {
            "model": "gpt-4o-2024-11-20",
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompts,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{user_prompts}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(data), proxies=proxies)
        result = response.json()
        text = result['choices'][0]['message']['content']

        with open(text_path, 'w') as f:
            f.write(text)
        print(text)

url = "https://api.openai-sb.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer sb-********",
    "Content-Type": "application/json",
    'Connection': 'close'
}
# proxies = {
#     "http": "http://ip address:4780",
#     "https": "http://ip address:4780",
# }

data_root = '/home/aiseon/storage1/dataset/WQ-Bench'
gpt(url, headers, proxies, data_root, lenth=50000)

