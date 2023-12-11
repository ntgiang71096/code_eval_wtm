
from sklearn.metrics import accuracy_score, recall_score
from core import run_eval
from extended_watermark_processor import WatermarkDetector
from initializer import init_tokenizer_and_model, get_gen_function_by_name

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)

from core import run_eval, replit_glaive_prompt
import os
import torch
import argparse
from submitit_utils import str2bool, load_generated_code, load_dataset_groundtruth

import os
import json

TOKEN = ''

torch.manual_seed(1234)

def generate_watermark_code(args, wtm_path):

    args = argparse.Namespace(**vars(args), use_watermark=True)
    print("generate_watermark_code: {}".format(args.use_watermark))
    os.makedirs(wtm_path, exist_ok=True)
    out_path = wtm_path + "/eval.jsonl"

    print("Get model, tokenizer and gen function by model name: {}".format(args.model_name))

    model, tokenizer = init_tokenizer_and_model(args.model_name, args.gpu, args.cache_location)
    gen_function = get_gen_function_by_name(args.model_name) 

    run_eval(
        args, model, tokenizer, args.pass_value, out_path, gen_function, num_samples=args.num_samples
    )

    return model, tokenizer


def get_watermark_detector(gamma, model, tokenizer):
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=gamma, # should match original setting
                                        seeding_scheme="simple_1", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=2.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)
    
    return watermark_detector


def get_detector_prediction(watermark_detector, code_list):
    result = []

    # giang: count blank to count number of generated code which is blank
    count_blank = 0
    for output_text in code_list:
        # print("output_text:{}".format(output_text))
        # print("len: {}".format(len(output_text)))
        # if output_text.strip() == '':
        #     count_blank += 1
        #     continue
        # print(output_text)
        score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
        # print()
        # print()
        # print("Prediction: {},  z_score: {}".format(score_dict['prediction'], score_dict['z_score']))
        # print('-' * 64)
        result.append((score_dict['prediction'], score_dict['z_score']))
        
        # print(score_dict['z_score'], score_dict['prediction'])

    # print("Number of blank completion: {}".format(count_blank)) 
    return result
    


def calculate_pass_k(wtm_path):
    os.system("evaluate_functional_correctness {}/eval.jsonl".format(wtm_path))

def try_param(args):
    # no_wtm_path = 'results/eval_codellama_no_watermark_4122023'

    gamma = args.gamma
    delta = args.delta

    print("Running with specific gamma {} and delta: {}".format(gamma, delta))

    wtm_path = 'results/result_{}_watermark_pass_{}_{}_{}'.format(args.model_name.replace("-", "_").replace(".", "_").replace("/", "_"), args.pass_value, int(gamma * 100), int(delta))

    print("Trying with Gamma: {}; Delta: {}, pass@{}".format(gamma, delta, args.pass_value))
    print("Saving result to folder: {}".format(wtm_path))

    print("Num_samples: {}".format(args.num_samples))

    if os.path.isfile(wtm_path + '/eval.jsonl'):
        print("Watermark result already exist.")
        model, tokenizer = init_tokenizer_and_model(args.model_name, args.gpu, args.cache_location)
    else:
        print("Generating code with watermark...")
        model, tokenizer = generate_watermark_code(args, wtm_path)
    
    wtm_code = load_generated_code(wtm_path)
    
    groundtruth = load_dataset_groundtruth(args.dataset_name)
    
    no_wtm_code = load_generated_code(args.no_wtm_path)

    print("Calculating accuracy...")

    X = wtm_code + no_wtm_code + groundtruth
    if args.num_samples is None:
        y_true = [1] * len(wtm_code) + [0] * (len(no_wtm_code)) + [0] * len(groundtruth)
    else:
        y_true = [1] * len(wtm_code) + [0] * (args.num_samples * 10) + [0] * args.num_samples
    # y_true = [1] * len(wtm_code) 

    y_pred = []

    watermark_detector = get_watermark_detector(gamma, model, tokenizer)
    
    pred_1 = get_detector_prediction(watermark_detector, wtm_code)
    for pred, z_score in pred_1:
        y_pred.append(pred)

    if args.num_samples is None:
        pred_2 = get_detector_prediction(watermark_detector, no_wtm_code)
    else:
        pred_2 = get_detector_prediction(watermark_detector, no_wtm_code[: (args.num_samples * 10)])

    for pred, z_score in pred_2:
        y_pred.append(pred)

    if args.num_samples is None:
        pred_3 = get_detector_prediction(watermark_detector, groundtruth)
    else:
        pred_3 = get_detector_prediction(watermark_detector, groundtruth[:args.num_samples])
    for pred, z_score in pred_3:
        y_pred.append(pred)
    
    print(len(pred_1), len(pred_2), len(pred_3))

    # giang, write prediction to eval.jsonl

    data = []
    with open(wtm_path + '/eval.jsonl', 'r') as file:
            for i, line in enumerate(file):
                json_data = json.loads(line)
                json_data['prediction'] = pred_1[i][0]
                json_data['z_score'] = pred_1[i][1]
                data.append(json_data)

    with open(wtm_path + '/eval_with_prediction.jsonl', 'w') as jsonl_file:
        for json_data in data:
            json_line = json.dumps(json_data)
            jsonl_file.write(json_line + '\n')

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: {}".format(accuracy))

    tnr = recall_score(y_true, y_pred, pos_label = 0) 
    fpr = 1 - tnr
    print("False Positive Rate: {}".format(fpr))

    if accuracy > 0.99 and fpr < 0.01:
        return True, accuracy, fpr
    else:
        return False, accuracy, fpr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        default=None
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default=None
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=-1
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0
    )

    parser.add_argument(
        "--pass_value",
        type=int,
        default=1
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None
    )

    parser.add_argument(
        "--cache_location",
        type=str,
        required=False,
        default=None
    )

    parser.add_argument(
        "--no_wtm_path",
        type=str,
        required=True,
        default=None
    )

    args = parser.parse_args()

    try_param(args)
    
    