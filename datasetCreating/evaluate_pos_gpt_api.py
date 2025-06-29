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

def gpt(url, headers, proxies, ref_dic, result_dic, save_json):
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            all_data = json.load(f)
        f.close()
    else:
        all_data = {}

    for name in tqdm(ref_dic):
        if len(all_data) == len(ref_dic):
            print('all prompts have been judged!')
            break
        if name in all_data:
            continue

        description = ref_dic[name]
        ground_truth = result_dic[name]

        sys_prompts = 'You are a helpful watermark discriminator.'

        user_prompts = textwrap.dedent(f"""Task: Determine whether the following two sentences describe approximately the same watermark position, allowing for minor variations in phrasing or slight positional bias.

        Sentence 1: {description}
        Sentence 2: {ground_truth}

        Judgment Criteria:
        1.  Do they describe the same general location, even if there are small differences?
        2.  Are the differences within an acceptable range (e.g., slight shifts in coordinates or wording variations like "top-left" vs. "upper-left")?
        3.  Is there any ambiguity that affects the interpretation?
        
        Expected Output:
        Just Final Consistency Judgment: (Yes/No/Unclear)""").replace("        ", "")

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompts,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{user_prompts}"}
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(data), proxies=proxies)
        result = response.json()
        text = result['choices'][0]['message']['content']
        print(text)
        all_data[name] = text
        with open(osp.join(save_json), 'w') as f:
            json.dump(all_data, f, indent=4)

##################################################

url = "https://api.openai-sb.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer sb-********",
    "Content-Type": "application/json",
    'Connection': 'close'
}

proxies = None

data_root = '/home/aiseon/storage1/dataset/WQ-Bench/test_wqbench.json'
with open(data_root) as f:
    all_data = json.load(f)
ref_dic = {}
for data in all_data:
    name = data['image'][:-3] + 'txt'
    ref_dic[name] = data['conversations'][1]['value']

result_root = '/home/aiseon/storage1/WMarkGPT/result/002_out'
result_dic = {}
for name in os.listdir(result_root):
    with open(osp.join(result_root, name), 'r') as f:
        data = f.readline().replace("</s>", "")
    result_dic[name] = data


save_json = '/home/aiseon/storage1/WmarkGPT/result/002_pos.json'
gpt(url, headers, proxies, ref_dic, result_dic, save_json)
