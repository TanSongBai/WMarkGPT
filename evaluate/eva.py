import sys
import os
from pathlib import Path

from timm.models import eva

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import random
import time
import argparse
from PIL import Image
import json
from tqdm import tqdm
from transformers import TextStreamer
from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor

from llm.model.modeling_wmark_gpt import MPLUGOwl2LlamaForCausalLM
from llm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llm.conversation import conv_templates
from llm.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
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

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main(args):
    # image_file = '000000581884_0.23.png' # Image Path
    torch.cuda.set_device(args.gpu)
    image_dir =  args.image_dir
    image_file = args.image_file
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    is_json = True if image_file[-4:] == 'json' else False
    if is_json:
        name_map = {}
        image_files = []
        with open(image_file, 'r') as f:
            image_json = json.load(f)
            for idx, x in enumerate(image_json):
                image_files.append(x['image'])
                name_map[x['image']] = idx
    image_file = image_files if is_json else [image_file]

    model_path = args.model_path

    disable_torch_init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.name_or_path = model_path
    model = MPLUGOwl2LlamaForCausalLM(config).to(torch.float16).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    temperature = 0.7
    max_new_tokens = 512
    model.eval()

    random.seed(time.time())
    random.shuffle(image_files)
    for file in tqdm(image_file):
        if os.path.exists(os.path.join(save_dir, file[:-3]+'txt')):
            continue
        print(file, '\n')
        image = Image.open(os.path.join(image_dir, file)).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        conv = conv_templates["mplug_owl2"].copy()
        query = random.choice(question_list)
        inp = query + "\n" + DEFAULT_IMAGE_TOKEN

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # print(outputs)
        with open(os.path.join(save_dir, file[:-3]+'txt'), 'w') as f:
            f.write(prompt + '\n' + outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_dir", type=str,
                        default='playground/WQA-Synthetic/stego')
    parser.add_argument("--image_file", type=str,
                        default='playground/WQA-Synthetic/test.json')
    parser.add_argument("--save_dir", type=str,
                        default='result/WQA-Synthetic/out')
    parser.add_argument("--model_path", type=str,
                        default="ckpt/synthetic")

    args = parser.parse_args()
    main(args)
