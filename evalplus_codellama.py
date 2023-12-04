from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import run_eval, replit_glaive_prompt, filter_code, fix_indents
import os
import torch
import argparse
from submitit_utils import str2bool
from torch import cuda

from codegen.model import make_model, VLlmDecoder
from codegen.generate import construct_contract_prompt

from core import run_eval, replit_glaive_prompt, filter_code, fix_indents


# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""

use_cuda = cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")

@torch.inference_mode()
def generate_batch_completion(
    model, tokenizer, prompt: str, batch_size: int, logit_processor_lst, use_watermark
) -> list:
    
    outputs = model.codegen(
                    construct_contract_prompt(
                        prompt=prompt, contract_type='none', contract='none'
                    ),
                    do_sample=True,
                    num_samples=10,
                    logit_processor_lst=logit_processor_lst
                )
    
    outputs = [filter_code(fix_indents(completion)) for completion in outputs]

    # batch_completions = [prompt + batch for batch in batch_completions]
    result = []
    for batch in outputs:
         result.append(fix_indents(prompt) + batch)

    return result
    # return outputs


if __name__ == "__main__":
    model_name = "codellama/CodeLlama-7b-hf"
    batch_size = 10
    temperature = 0.8

    model = VLlmDecoder(name=model_name, batch_size=batch_size, temperature=temperature)


    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")

    parser.add_argument(
        "--use_watermark",
        type=str2bool,
        default=False,
        help=("Whether to use watermark"),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The ratio of blacklist to whitelist tokens when splitting the vocabulary",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary at every step",
    )
    
    parser.add_argument(
        "--out_path",
        type=str,
        default=False,
        help=("Result path"),
    )

    parser.add_argument(
        "--pass_value",
        type=int,
        default=False,
        help=("pass_value either 1 or 10"),
    )
    
    args = parser.parse_args()

    result_path = args.out_path
    
    os.makedirs(result_path, exist_ok=True)
    out_path = result_path + "/eval.jsonl"

    # os.makedirs("results/replit_glaive", exist_ok=True)

    num_samples_per_task = args.pass_value

    run_eval(
        args, model, None, num_samples_per_task, out_path, generate_batch_completion
    )




