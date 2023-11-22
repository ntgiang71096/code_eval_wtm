from extended_watermark_processor import WatermarkLogitsProcessor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    LogitsProcessorList,
)
from core import run_eval, replit_glaive_prompt
import os
import torch
import argparse
from submitit_utils import str2bool
import inspect
import json

torch.manual_seed(1234)

TOKEN = ' '
tokenizer = AutoTokenizer.from_pretrained(
        "sahil2801/replit-code-instruct-glaive",
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )
# model = torch.compile(
#         AutoModelForCausalLM.from_pretrained(
#             "sahil2801/replit-code-instruct-glaive",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#             use_auth_token=TOKEN,
#             init_device="cuda",
#         ).eval()
#     )
# watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
#                                                gamma=0.25,
#                                                delta=8.0,
#                                                seeding_scheme="selfhash") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`

def load_generated_code(file_path):
    result = []
    with open(file_path + '/eval.jsonl', 'r') as file:
            for i, line in enumerate(file):
                # if i % 10 != 0:
                #         continue
                # Remove leading/trailing whitespace and parse the JSON object
                json_data = json.loads(line)
                result.append(json_data['completion'])

    return result

input_text = '''
from typing import List
from typing_extensions import Literal
from pyramid_graphql import ObjectType, String


class Query(ObjectType):
    filter_by_substring = List(String)(
        required=False,
        description=(
            f'Filter list of objects by substring, if given {substring!r}, if empty list is returned'
        )
    )(default_value=[])
    
    def __call__(
        self, info, objects, substring, '''

tokenized_input = tokenizer(input_text, return_tensors='pt')
print(len(tokenized_input['input_ids'][0]))

# code_list = load_generated_code('results/eval_new_watermark_pass_10_40_15')
# for input_text in code_list:
#     # tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
    
#     tokenized_input = tokenizer(input_text, return_tensors='pt')
#     print(len(tokenized_input['input_ids'][0]))
#     # print(tokenized_input.shape)











# input_text = """from typing import List
# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     #  Check if in given list of numbers, are any two numbers closer to each other than
#     # given threshold.
#     # >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     # False
#     # >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
#     # True"""

# tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
# # note that if the model is on cuda, then the input is on cuda
# # and thus the watermarking rng is cuda-based.
# # This is a different generator than the cpu-based rng in pytorch!
# output_tokens = model.generate(**tokenized_input,
#                                logits_processor=LogitsProcessorList([watermark_processor]), max_new_tokens=100, do_sample=True, num_return_sequences=5)
# # if decoder only model, then we need to isolate the
# # newly generated tokens as only those are watermarked, the input/prompt is not
# output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]
# output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)#[0]
# print('0')
# print(output_text[0])
# print('1')
# print(output_text[1])
# print('2')
# print(output_text[2])
# print('3')
# print(output_text[3])
# print('4')
# print(output_text[4])

# developer_wrriten_result = """ for idx, elem in enumerate(numbers):
#         for idx2, elem2 in enumerate(numbers):
#             if idx != idx2:
#                 distance = abs(elem - elem2)
#                 if distance < threshold:
#                     return True
#     return False  """
# from extended_watermark_processor import WatermarkDetector
# watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
#                                         gamma=0.25, # should match original setting
#                                         seeding_scheme="selfhash", # should match original setting
#                                         device=model.device, # must match the original rng device type
#                                         tokenizer=tokenizer,
#                                         z_threshold=4.0,
#                                         normalizers=[],
#                                         ignore_repeated_ngrams=True)
# score_dict = watermark_detector.detect(output_text[0])
# print(score_dict['prediction'], score_dict['z_score'])
# score_dict = watermark_detector.detect(output_text[1])
# print(score_dict['prediction'], score_dict['z_score'])
# score_dict = watermark_detector.detect(output_text[2])
# print(score_dict['prediction'], score_dict['z_score'])
# score_dict = watermark_detector.detect(output_text[3])
# print(score_dict['prediction'], score_dict['z_score'])
# score_dict = watermark_detector.detect(output_text[4])
# print(score_dict['prediction'], score_dict['z_score'])
# score_dict = watermark_detector.detect(developer_wrriten_result)
# print(score_dict['prediction'], score_dict['z_score'])