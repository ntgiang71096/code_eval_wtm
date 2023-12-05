
from sklearn.metrics import accuracy_score, recall_score
from core import run_eval
from eval_codellama import generate_batch_completion
from extended_watermark_processor import WatermarkDetector
# from codegen.model import VLlmDecoder

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    LlamaForCausalLM,
)

from extended_watermark_processor import WatermarkLogitsProcessor
from transformers import (
    LogitsProcessorList,
)

from core import run_eval, replit_glaive_prompt
import os
import torch
import argparse
from submitit_utils import str2bool
import inspect
from human_eval.data import write_jsonl, read_problems

import os
import json

TOKEN = ''

torch.manual_seed(1234)


def load_human_eval_groundtruth():
    problems = read_problems()

    result = [problems[task_id]['canonical_solution'] for task_id in problems]

    # giang testing for 10 sample first
    # result = result[:10]

    return result


def generate_watermark_code(gamma, delta, wtm_path, pass_value, model, tokenizer, num_samples):
    # giang: this is just to keep the code compatible, not the real ArgumentParser
    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")

    parser.add_argument(
        "--use_watermark",
        type=str2bool,
        default=True,
        help=("Whether to use watermark"),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=gamma,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=delta,
        
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0
    )

    parser.add_argument(
        "--pass_value",
        type=int,
        default=pass_value
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=num_samples
    )


    args = parser.parse_args()    
    os.makedirs(wtm_path, exist_ok=True)

    out_path = wtm_path + "/eval.jsonl"

    run_eval(
        args, model, tokenizer, pass_value, out_path, generate_batch_completion, num_samples=num_samples
    )

    return 


def load_generated_code(file_path):
    problems = read_problems()

    prompt = {task_id:problems[task_id]['prompt'] for task_id in problems}

    result = []
    with open(file_path + '/eval.jsonl', 'r') as file:
            for i, line in enumerate(file):
                # if i % 10 != 0:
                #         continue
                # Remove leading/trailing whitespace and parse the JSON object
                json_data = json.loads(line)
                result.append(json_data['completion'][len(prompt[json_data['task_id']]):])
    
    # print(result)

    return result

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
        print("output_text:{}".format(output_text))
        print("len: {}".format(len(output_text)))
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
    # os.system("python3 process_eval.py --path {} --out_path {}/processed.jsonl --add_prompt".format(wtm_path, wtm_path))
    os.system("evaluate_functional_correctness {}/eval.jsonl".format(wtm_path))

def init_tokenizer_and_model(gpu):
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token


    model = torch.compile(
        LlamaForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto", load_in_4bit=True
        )
    )

    model.eval()

    return model, tokenizer


def try_param(gamma, delta, pass_value, wtm_path, model, tokenizer, num_samples = None):

    no_wtm_path = 'results/eval_codellama_no_watermark_4122023'

    groundtruth = load_human_eval_groundtruth()
    no_wtm_code = load_generated_code(no_wtm_path)

    if os.path.isfile(wtm_path + '/eval.jsonl'):
        print("Watermark result already exist.")
    else:
        print("Generating code with watermark...")
        generate_watermark_code(gamma, delta, wtm_path, pass_value, model, tokenizer, num_samples)
    
    # print(w)
    wtm_code = load_generated_code(wtm_path)

    print("Calculating accuracy...")

    X = wtm_code + no_wtm_code + groundtruth
    if num_samples is None:
        y_true = [1] * len(wtm_code) + [0] * (len(no_wtm_code)) + [0] * len(groundtruth)
    else:
        y_true = [1] * len(wtm_code) + [0] * (num_samples * 10) + [0] * num_samples
    # y_true = [1] * len(wtm_code) 

    y_pred = []

    watermark_detector = get_watermark_detector(gamma, model, tokenizer)

    
    pred_1 = get_detector_prediction(watermark_detector, wtm_code)
    for pred, z_score in pred_1:
        y_pred.append(pred)

    if num_samples is None:
        pred_2 = get_detector_prediction(watermark_detector, no_wtm_code)
    else:
        pred_2 = get_detector_prediction(watermark_detector, no_wtm_code[: (num_samples * 10)])

    for pred, z_score in pred_2:
        y_pred.append(pred)

    if num_samples is None:
        pred_3 = get_detector_prediction(watermark_detector, groundtruth)
    else:
        pred_3 = get_detector_prediction(watermark_detector, groundtruth[:num_samples])
    for pred, z_score in pred_3:
        y_pred.append(pred)
    
    print(len(pred_1), len(pred_2), len(pred_3))
    # y_pred.extend(get_detector_prediction(watermark_detector, wtm_code))
    # y_pred.extend(get_detector_prediction(watermark_detector, no_wtm_code))
    # y_pred.extend(get_detector_prediction(watermark_detector, groundtruth))


    # giang, write prediction to eval.jsonl

    data = []
    with open(wtm_path + '/eval.jsonl', 'r') as file:
            for i, line in enumerate(file):
                # if i % 10 != 0:
                #         continue
                # Remove leading/trailing whitespace and parse the JSON object
                json_data = json.loads(line)
                json_data['prediction'] = pred_1[i][0]
                json_data['z_score'] = pred_1[i][1]
                data.append(json_data)
                # result.append(json_data['completion'])


    with open(wtm_path + '/eval_with_prediction.jsonl', 'w') as jsonl_file:
        for json_data in data:
            json_line = json.dumps(json_data)
            jsonl_file.write(json_line + '\n')


    # data = []
    # with open(no_wtm_path + '/eval.jsonl', 'r') as file:
    #         for i, line in enumerate(file):
    #             # if i % 10 != 0:
    #             #         continue
    #             # Remove leading/trailing whitespace and parse the JSON object
    #             json_data = json.loads(line)
    #             if i < len(pred_2):
    #                 json_data['prediction'] = pred_2[i][0]
    #                 json_data['z_score'] = pred_2[i][1]
    #             data.append(json_data)
    #             # result.append(json_data['completion'])

    # with open(no_wtm_path + '/eval_with_prediction.jsonl', 'w') as jsonl_file:
    #     for json_data in data:
    #         json_line = json.dumps(json_data)
    #         jsonl_file.write(json_line + '\n')

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: {}".format(accuracy))

    tnr = recall_score(y_true, y_pred, pos_label = 0) 
    fpr = 1 - tnr
    print("False Positive Rate: {}".format(fpr))

    # if accuracy >= 0.7:
    #     calculate_pass_k(wtm_path)
    if accuracy > 0.99 and fpr < 0.01:
        return True, accuracy, fpr
    else:
        return False, accuracy, fpr



# groundtruth = load_human_eval_groundtruth()
# print(groundtruth)
# print(len(groundtruth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")
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

    

    args = parser.parse_args()
    
    gamma = args.gamma
    delta = args.delta
    model, tokenizer = init_tokenizer_and_model(args.gpu)

    
    if delta == -1:
        print("Trying to find optimal gamma and delta")
        l = 0
        r = 20
        while l < r:
            
            delta = int((l + r) / 2)
            # delta = 5

            wtm_path = 'results/eval_codellama_watermark_pass_{}_{}_{}'.format(args.pass_value, int(gamma * 100), delta)
            print("Trying with Gamma: {}; Delta: {}, pass@{}".format(gamma, delta, args.pass_value))
            print("Saving result to folder: {}".format(wtm_path))
        
            ok, accuracy, fpr = try_param(gamma = gamma, delta = delta, pass_value=args.pass_value, wtm_path = wtm_path , model=model, tokenizer=tokenizer, num_samples=args.num_samples)
            if ok:
                print("Correct detection. Decreasing delta.")
                r = delta
            else:
                print("Incorrect detection. Increasing delta.")
                l = delta + 1

            print("-" * 64)
        if ok:
            print("Found optimal value: Gamma {}, Delta {}".format(gamma, delta))

    else:
        print("Running with specific gamma {} and delta: {}".format(gamma, delta))

        wtm_path = 'results/eval_codellama_watermark_pass_{}_{}_{}'.format(args.pass_value, int(gamma * 100), int(delta))
        print("Trying with Gamma: {}; Delta: {}, pass@{}".format(gamma, delta, args.pass_value))
        print("Saving result to folder: {}".format(wtm_path))

        print("Num_samples: {}".format(args.num_samples))
        ok, accuracy, fpr = try_param(gamma = gamma, delta = delta, pass_value=args.pass_value, wtm_path = wtm_path , model=model, tokenizer=tokenizer, num_samples=args.num_samples)