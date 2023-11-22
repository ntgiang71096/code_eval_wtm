from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, replit_glaive_prompt
import os
import torch
import argparse
from submitit_utils import str2bool

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int, logit_processor_lst, use_watermark
) -> list:
    
    arg_dict = {1: {"num_return_sequences" : 1, "top_p" : 0.95, "temperature" : 0.2},
                10: {"num_return_sequences" : 10, "top_p" : 0.95, "temperature" : 0.8}}
    
    prompt_input = replit_glaive_prompt(prompt)

    tokenized_input = tokenizer(prompt_input, return_tensors='pt').to(model.device)

    if use_watermark:
        output_tokens = model.generate(**tokenized_input, 
                                    logits_processor=logit_processor_lst,
                                    max_new_tokens=128,
                                    min_new_tokens=10,
                                    do_sample=True,
                                    temperature=arg_dict[batch_size]['temperature'],
                                    top_p=arg_dict[batch_size]['top_p'],
                                    num_return_sequences=arg_dict[batch_size]['num_return_sequences'])
    else:
        output_tokens = model.generate(**tokenized_input, 
                                    max_new_tokens=128,
                                    min_new_tokens=10,
                                    do_sample=True,
                                    temperature=arg_dict[batch_size]['temperature'],
                                    top_p=arg_dict[batch_size]['top_p'],
                                    num_return_sequences=arg_dict[batch_size]['num_return_sequences'])
    
    # print(output_tokens.shape)
    output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]
    # print(output_tokens.shape)
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)#[0]
    return output_text


if __name__ == "__main__":
    # adjust for n = 10 etc

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
            init_device="cuda:0",
        ).eval()
    )


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
        args, model, tokenizer, num_samples_per_task, out_path, generate_batch_completion
    )
