import json
import numpy as np

label_path = 'playground/WQA-Synthetic/test.json'

paths = [
    'result/WQA-Synthetic/wmarkgpt-7b-result.json',
    'result/WQA-Synthetic/qwen-vl-chat-result.json',
    'result/WQA-Synthetic/qwen2-vl-7b-instruct-result.json',
    'result/WQA-Synthetic/mplug-owl2-7b-result.json',
    'result/WQA-Synthetic/llava_1.5_13b_hf-result.json',
    'result/WQA-Synthetic/llava_1.5_7b_hf-result.json',
    'result/WQA-Synthetic/Efficient-Large-Model-NVILA-8B-result.json'
]

paths_ = [
    'result/WQA-Synthetic/wmarkgpt-7b.json',
    'result/WQA-Synthetic/qwen-vl-chat.json',
    'result/WQA-Synthetic/qwen2-vl-7b-instruct.json',
    'result/WQA-Synthetic/mplug-owl2-7b.json',
    'result/WQA-Synthetic/llava_1.5_13b_hf.json',
    'result/WQA-Synthetic/llava_1.5_7b_hf.json',
    'result/WQA-Synthetic/Efficient-Large-Model-NVILA-8B.json'
]

with open(label_path, 'r') as f:
    label_data = json.load(f)
label_visibility = {}
for x in label_data:
    label_visibility[x['image']] = x['visibility']

for i, path in enumerate(paths):
    with open(path, 'r') as f:
        datas = json.load(f)
    with open(paths_[i], 'r') as f:
        visible_datas = json.load(f)
    visibility = {}
    for x in visible_datas:
        visibility[x['image']] = x['visibility']
    llm_scores = []
    BLEU_1 = []
    ROUGE_L = []
    ACC = []
    for name in datas:
        llm_scores.append(datas[name]['LLM_SCORE'])
        BLEU_1.append(datas[name]['BLEU_1'])
        ROUGE_L.append(datas[name]['ROUGE_L'])
        name_ = name.replace('JPEG', 'JJPEG')
        ACC.append(label_visibility[name] in visibility[name_])

    print(path, '\n')
    print('BLEU_1', np.mean(BLEU_1), '\n')
    print('ROUGE_L', np.mean(ROUGE_L), '\n')
    print('LLM_SCORES', np.mean(llm_scores)*25, '\n')
    print('ACC', np.mean(ACC), '\n')




