from extended_watermark_processor_backup import WatermarkDetector

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
# from core import run_eval, replit_glaive_prompt
import os
import torch
import argparse
from submitit_utils import str2bool
import json


TOKEN = ""

tokenizer = AutoTokenizer.from_pretrained(
        "sahil2801/replit-code-instruct-glaive",
        trust_remote_code=True,
        use_auth_token=TOKEN,
    )


model = torch.compile(
    AutoModelForCausalLM.from_pretrained(
        "sahil2801/replit-code-instruct-glaive",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=TOKEN,
        init_device="cuda:2",
    ).eval()
)

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.5, # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)

file_path = 'results/replit_glaive_watermark_17112023_3/processed.jsonl'

output_text = ''
with open(file_path, 'r') as file:
            count_nan = 0
            ppls = []
            for i, line in enumerate(file):
                # if i % 10 != 0:
                #         continue
                # Remove leading/trailing whitespace and parse the JSON object
                json_data = json.loads(line)
                output_text = json_data['completion']
                # print(output_text)

                # break

                score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
                # del score_dict['z_score_at_T']
                # print(score_dict['prediction'])
                prediction = score_dict['prediction']
                print(score_dict['z_score'], score_dict['prediction'])
                # if prediction == True:
                #     print(score_dict['z_score'], score_dict['prediction'])
