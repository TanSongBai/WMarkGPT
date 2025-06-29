import json
import os.path
from tqdm import tqdm
import numpy as np
from rouge import Rouge
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import argparse
from llm.metrics.gpt_api import gpt


def Rouge_Score(hypothesis, reference):
    rouger = Rouge()
    scores = rouger.get_scores(hypothesis, reference)
    arr_1 = np.array([[item['rouge-1']['r'], item['rouge-1']['p'], item['rouge-1']['f']] for item in scores])
    avg_r_1, avg_p_1, avg_f_1 = np.mean(arr_1[:, 0]), np.mean(arr_1[:, 1]), np.mean(arr_1[:, 2])
    arr_2 = np.array([[item['rouge-2']['r'], item['rouge-2']['p'], item['rouge-2']['f']] for item in scores])
    avg_r_2, avg_p_2, avg_f_2 = np.mean(arr_2[:, 0]), np.mean(arr_2[:, 1]), np.mean(arr_2[:, 2])
    arr_l = np.array([[item['rouge-l']['r'], item['rouge-l']['p'], item['rouge-l']['f']] for item in scores])
    avg_r_l, avg_p_l, avg_f_l = np.mean(arr_l[:, 0]), np.mean(arr_l[:, 1]), np.mean(arr_l[:, 2])
    avg_scores = {
        'rouge-1': {'r': avg_r_1, 'p': avg_p_1, 'f': avg_f_1},
        'rouge-2': {'r': avg_r_2, 'p': avg_p_2, 'f': avg_f_2},
        'rouge-l': {'r': avg_r_l, 'p': avg_p_l, 'f': avg_f_l}
    }
    return avg_scores, arr_l[:, 2][0]


def Bleu_Scores(candidates, references):
    references = np.array(references).reshape(-1, 1)
    references_tokens = [[s.split() for s in sentences] for sentences in references]

    candidates_tokens = [sentence.split() for sentence in [candidates]]

    bleu_scores = [corpus_bleu(references_tokens, candidates_tokens, weights=(n,)) for n in range(1, 5)]
    bleu_1_scores = [sentence_bleu(r, z, weights=(1,)) for r, z in zip(references_tokens, candidates_tokens)]
    ave_scores = {}
    for n, score in enumerate(bleu_scores, start=1):
        ave_scores[f"BLEU-{n}"] = score
    return ave_scores, bleu_1_scores[0]


def Llm_scores(candidates, references, api_key):
    user_prompts = f"""Evaluate the relevance between the following description and ground truth on a scale from 0 to 4.\nHigher scores indicate better relevance. Return only the numeric score.\nDescription: "{candidates}"\nGround Truth: "{references}" """
    score = gpt(user_prompts, secret_key=api_key)
    return float(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers against key labels")

    parser.add_argument("--label_json", default="playground/WQA-Synthetic/test.json",
                        type=str, help="Path to label file")
    parser.add_argument("--answer_json", default="result/WQA-Synthetic/wmarkgpt-7b.json",
                        type=str, help="Path to answers file")
    parser.add_argument("--api_key", default="**-***************",
                        type=str, help="Your api key")

    args = parser.parse_args()

    save_name = os.path.basename(args.answer_json)[:-5] + '-result.json'
    args.result = os.path.join(args.result, save_name)

    with open(args.label_json, 'r') as f:
        labels = json.load(f)

    with open(args.answer_json, 'r') as f:
        answers = json.load(f)

    labels_dir = {}
    answers_dir = {}
    for label in labels:
        labels_dir[label['image']] = {'text': label['conversations'][1]['value'].strip('<\s>'), 'visibility': label['visibility']}
    for answer in answers:
        answers_dir[answer['image']] = {'text': answer['conversations'][1]['value'].strip('<\s>'), 'visibility': answer['visibility']}

    result_dir = {}
    for name in tqdm(labels_dir):
        ROUGE, ROUGE_L = Rouge_Score(answers_dir[name]['text'], labels_dir[name]['text'])
        BLEU, BLEU_1 = Bleu_Scores(answers_dir[name]['text'], labels_dir[name]['text'])
        try:
            score = Llm_scores(answers_dir[name]['text'], labels_dir[name]['text'], api_key=args.api_key)
        except:
            score = 2.5
            print(name, '######', score)
        result_dir[name] = {'ROUGE': ROUGE, 'ROUGE_L': ROUGE_L , 'BLEU': BLEU, 'BLEU_1': BLEU_1, 'LLM_SCORE': score}
    with open(args.result, 'w') as f:
        json.dump(result_dir, f, indent=4)





