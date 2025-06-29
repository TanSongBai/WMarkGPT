import json
import os
from llm.metrics.gpt_api import gpt
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create answer json")

    parser.add_argument("--result", default="result/WQA-Synthetic/out",
                        type=str, help="Path to result file")
    parser.add_argument("--api_key", default="**-***************",
                        type=str, help="Your api key")

    args = parser.parse_args()

    end = 'png'
    query = "Please provide a detailed description of the watermark in terms of its direction, its placement relative to objects, its appearance, and its visibility.\n First, describe the specific orientation of the watermark in the image. Next, explain the relative position of the watermark to the objects in the image, indicating whether the watermark is inside, partially overlapping, or near the edge of the objects. Then, briefly describe the watermark's characteristics, such as appearance or texture. Finally, you must evaluate watermark visibility using the following format: 'The watermark visibility is: [invisible/faint/moderate/visible/obvious]'.\n Present all this information in a cohesive and concise paragraph."

    save_json = os.path.dirname(args.result) + '/final_answers.json'
    memory = []
    for path in tqdm(os.listdir(args.result)):
        stego_name = path[:-3] + end
        question = '<|image|>' + query
        with open(os.path.join(args.result, path), 'r') as f:
            line = f.readline()
            answer = line.replace("*", "").strip()
        dis_question = f'''You are an expert in image watermark analysis, specializing in assessing the visibility of watermarks. I will provide a textual description of an image watermark, and your task is to evaluate its visibility based on the description. Choose the most appropriate visibility level from the following options: invisible, faint, moderate, visible, or obvious. Provide only the selected visibility level as your output.\nInput: {answer}'''
        visible_answer = gpt(dis_question, secret_key=args.api_key)
        data_dict = {}
        data_dict['image'] =  data_dict['id'] = stego_name
        data_dict['visibility'] = visible_answer
        data_dict['conversations'] = []
        data_dict['conversations'].append({'from': 'human', 'value': question})
        data_dict['conversations'].append({'from': 'gpt', 'value': answer})
        memory.append(data_dict)

    with open(save_json, 'w') as f:
        json.dump(memory, f, indent=4)
