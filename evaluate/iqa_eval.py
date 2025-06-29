import argparse
import torch
from scipy.special import softmax
from scipy.stats import spearmanr, pearsonr
from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor

from llm.model.modeling_wmark_gpt import MPLUGOwl2LlamaForCausalLM
from llm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llm.conversation import conv_templates, SeparatorStyle
from llm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm
from collections import defaultdict

import os
import re
import random
import warnings

warnings.filterwarnings("ignore")
question_list = [
    "Can you describe the watermark in the image, focusing on its orientation, position relative to objects, appearance, and visibility?",
    "Please explain the direction of the watermark in the image, its relationship to other objects, its appearance, and how visible it is.",
    "Describe the watermark in the image, considering its angle, location in relation to objects, visual characteristics, and overall visibility.",
    "How would you describe the watermark's orientation, its position relative to objects, its texture, and its visibility in the image?",
    "Please provide a detailed description of the watermark in terms of its direction, its placement relative to objects, its appearance, and its visibility.",
    "Could you explain the watermark's direction in the image, how it interacts with objects, its visual style, and how visible it appears?",
    "Describe the watermark's orientation and position in relation to objects in the image, as well as its characteristics and visibility.",
    "Please analyze the watermark's direction, its relative position to objects, its visual properties, and the level of its visibility.",
    "Could you provide a summary of the watermark’s angle, its relationship to objects, its visual features, and how clearly it is visible?",
    "What is the orientation of the watermark in the image, how does it relate to objects, what are its visual attributes, and how noticeable is it?",
    "Please describe the watermark's direction, its placement in relation to objects, its texture or appearance, and how visible it is.",
    "Can you explain the watermark's orientation in the image, its relationship to objects, its visual style, and the visibility level?",
    "How does the watermark appear in terms of direction, placement relative to objects, visual texture, and overall visibility?",
    "What is the positioning of the watermark relative to objects, and how would you describe its direction, appearance, and visibility?",
    "Please describe the watermark’s orientation in the image, how it interacts with objects, its visual properties, and its visibility.",
    "Can you provide details about the watermark’s direction, how it fits with the objects in the image, and its visual characteristics?",
    "What is the watermark’s orientation and location relative to objects, and how would you describe its appearance and visibility?",
    "Could you describe the watermark's angle, its position in the image relative to objects, its style, and how visible it is?",
    "Please explain the watermark’s direction, its position relative to objects, its texture or appearance, and the visibility level.",
    "What can you say about the watermark’s direction in the image, its positioning relative to objects, its visual style, and its clarity?",
    "Could you detail the orientation of the watermark, how it interacts with other objects, its appearance, and its visibility?",
    "Describe the watermark’s positioning in relation to objects, its direction, visual properties, and the visibility of the watermark.",
    "What is the watermark’s direction in the image, and how does it relate to objects in terms of position, appearance, and visibility?",
    "How would you characterize the watermark’s angle, its relation to objects, its visual characteristics, and how visible it is?",
    "Please provide a description of the watermark’s orientation, its position in relation to objects, its texture, and its visibility.",
    "Can you explain the watermark’s positioning, its direction in the image, its appearance, and the level of visibility?",
    "How would you describe the watermark’s direction, its spatial relationship with objects, its style, and its visibility in the image?",
    "Please detail the watermark’s orientation, its relative position to objects, its visual features, and how visible it appears.",
    "Could you summarize the watermark’s orientation, its location relative to objects, its appearance, and how clearly it is visible?",
    "What is the watermark’s direction and placement in the image, and how would you describe its visual style and visibility?"
]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def remove_last_sentence(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    if len(sentences) > 1:
        sentences = sentences[:-1]
    return ' '.join(sentences)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    config.name_or_path = args.model_path
    model = MPLUGOwl2LlamaForCausalLM(config).to(torch.float16).to(args.device)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
    model.eval()

    toks = ['invisible', 'faint', 'moderate', 'visible', 'obvious']
    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)

    with open(args.json_path) as f:
        iqadata = json.load(f)

        memory = {}

        for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(args.json_path.split("/")[-1]))):
            try:
                filename = llddata["image"]
            except:
                filename = llddata["img_path"]

            image = load_image(os.path.join(args.data_dir, filename))
            def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)

            conv_mode = "mplug_owl2"

            query = random.choice(question_list)

            conv = conv_templates[conv_mode].copy()
            inp = query + "\n" + DEFAULT_IMAGE_TOKEN
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)

            with open(os.path.join(args.text_dir, os.path.basename(filename)[:-4]+'.txt'), "r") as f:
                prior_prompt = f.readlines()[0].strip('</s>')
                prior_prompt = remove_last_sentence(prior_prompt)
            # prompt = conv.get_prompt() + ' ' + prior_prompt + 'The watermark visibility is: '
            prompt = conv.get_prompt() + ' ' + 'The watermark visibility is: '
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).to(args.device)

            with torch.inference_mode():
                output_logits = model(input_ids, images=image_tensor)["logits"][:,-1]

            memory[filename] = {'gt_score': llddata['gt_score'], 'logits': {}, 'predict_score': 0.0}
            for tok, id_ in zip(toks, ids_):
                memory[filename]["logits"][tok] = output_logits[0, id_].item()

            score_logits = softmax([memory[filename]['logits'][x] for x in toks])
            score_map = [5, 4, 3, 2, 1]
            score = [a * b for a, b in zip(score_map, score_logits)]
            memory[filename]['predict_score'] = sum(score)

        with open(args.save_path, 'w') as f:
            json.dump(memory, f, indent=4)

        gts = []
        scores = []
        for name in memory:
            gts.append(memory[name]['gt_score'])
            scores.append(memory[name]['predict_score'])
        srcc, _ = spearmanr(gts, scores)
        plcc, _ = pearsonr(gts, scores)
        print(f'SRCC:{srcc}, PLCC:{plcc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="TanSongBai/WMarkGPT-Synthetic")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--text_dir", type=str, default='result/WQA-Synthetic/out')
    parser.add_argument("--data_dir", type=str, default='playground/WQA-Synthetic/stego')
    parser.add_argument("--json_path", type=str, default='playground/WQA-Synthetic/test.json')
    parser.add_argument("--save_path", type=str, default='result/WQA-Synthetic/wmarkgpt_qscore.json')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)