from extended_watermark_processor import WatermarkLogitsProcessor
from transformers import (
    CodeLlamaTokenizer,
    LlamaForCausalLM,
    LogitsProcessorList,
)
from core import run_eval, replit_glaive_prompt
import os
import torch
import argparse
from submitit_utils import str2bool
import inspect

torch.manual_seed(1234)

TOKEN = ' '
# tokenizer = CodeLlamaTokenizer.from_pretrained(
#         "sahil2801/replit-code-instruct-glaive",
#         trust_remote_code=True,
#         use_auth_token=TOKEN,
#     )
# model = torch.compile(
#         LlamaForCausalLM.from_pretrained(
#             "sahil2801/replit-code-instruct-glaive",
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#             use_auth_token=TOKEN,
#             init_device="cuda",
#         ).eval()
#     )

tokenizer = CodeLlamaTokenizer.from_pretrained(
        "codellama/CodeLlama-7b-hf",
    )

model = torch.compile(
    LlamaForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        torch_dtype=torch.bfloat16,
    )
    .eval()
    .to("cuda:1")
)
watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.5,
                                               delta=8.0,
                                               seeding_scheme="selfhash") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.
input_text = """from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    #  Check if in given list of numbers, are any two numbers closer to each other than
    # given threshold.
    # >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    # False
    # >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    # True"""

tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
# note that if the model is on cuda, then the input is on cuda
# and thus the watermarking rng is cuda-based.
# This is a different generator than the cpu-based rng in pytorch!
output_tokens = model.generate(**tokenized_input,
                               logits_processor=LogitsProcessorList([watermark_processor]), max_new_tokens=512, do_sample=True, num_return_sequences=5)
# if decoder only model, then we need to isolate the
# newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)#[0]
print('0')
print(output_text[0])
print('1')
print(output_text[1])
print('2')
print(output_text[2])
print('3')
print(output_text[3])
print('4')
print(output_text[4])

developer_wrriten_result = """ for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False  """
from extended_watermark_processor import WatermarkDetector
watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="selfhash", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)
score_dict = watermark_detector.detect(output_text[0])
print(score_dict['prediction'], score_dict['z_score'])
score_dict = watermark_detector.detect(output_text[1])
print(score_dict['prediction'], score_dict['z_score'])
score_dict = watermark_detector.detect(output_text[2])
print(score_dict['prediction'], score_dict['z_score'])
score_dict = watermark_detector.detect(output_text[3])
print(score_dict['prediction'], score_dict['z_score'])
score_dict = watermark_detector.detect(output_text[4])
print(score_dict['prediction'], score_dict['z_score'])
score_dict = watermark_detector.detect(developer_wrriten_result)
print(score_dict['prediction'], score_dict['z_score'])