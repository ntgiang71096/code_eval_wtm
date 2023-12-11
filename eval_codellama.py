# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

from transformers import (
    CodeLlamaTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer
)
from core import run_eval, replit_glaive_prompt, filter_code, fix_indents
import os
import torch
import argparse
from submitit_utils import str2bool
from torch import cuda
# import deepspeed



# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""

use_cuda = cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int, logit_processor_lst, use_watermark
) -> list:
    
    arg_dict = {1: {"num_return_sequences" : 1, "top_p" : 0.95, "temperature" : 0.2},
                10: {"num_return_sequences" : 10, "top_p" : 0.95, "temperature" : 0.8},
                40: {"num_return_sequences" : 40, "top_p" : 0.95, "temperature" : 0.8}}

    prompt_input = prompt
    # print(prompt)
    # prompt_input = replit_glaive_prompt(prompt)

    inputs = tokenizer([prompt_input], return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    if use_watermark:
        generated_ids = model.generate(**inputs, 
                                    use_cache=True,
                                    logits_processor=logit_processor_lst,
                                    max_new_tokens=256,
                                    min_new_tokens=10,
                                    do_sample=True,
                                    temperature=arg_dict[batch_size]['temperature'],
                                    top_p=arg_dict[batch_size]['top_p'],
                                    num_return_sequences=arg_dict[batch_size]['num_return_sequences'],
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
    else:
        generated_ids = model.generate(**inputs, 
                                    use_cache=True,
                                    max_new_tokens=256,
                                    min_new_tokens=10,
                                    do_sample=True,
                                    temperature=arg_dict[batch_size]['temperature'],
                                    top_p=arg_dict[batch_size]['top_p'],
                                    num_return_sequences=arg_dict[batch_size]['num_return_sequences'],
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
    
    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )
    filtered_batch_completions = []

    # must do this so the watermark detector doesn't get error because of empty code
    # 5 tokens do not make completion meaningful after all
    for completion in batch_completions:
        filtered_code = filter_code(fix_indents(completion))
        if len(filtered_code.strip()) >= 10:
             filtered_batch_completions.append(filtered_code)
        else:
             filtered_batch_completions.append(completion)

    # batch_completions = [filter_code(fix_indents(completion)) for completion in batch_completions]

    # batch_completions = [prompt + batch for batch in batch_completions]
    result = []
    for completion in filtered_batch_completions:
         result.append(fix_indents(prompt) + " " + completion)

    return result
    # return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc

    # tokenizer = CodeLlamaTokenizer.from_pretrained(
    #     "codellama/CodeLlama-7b-hf", 
    # )

#     ds_model = deepspeed.init_inference(
#     model=model,      # Transformers models
#     mp_size=2,        # Number of GPU
#     dtype=torch.float16, # dtype of the weights (fp16)
#     replace_method="auto", # Lets DS autmatically identify the layer to replace
#     replace_with_kernel_inject=True, # replace the model with the kernel injector
# )
    # model = torch.compile(
    #     AutoModelForCausalLM.from_pretrained(
    #         "sahil2801/replit-code-instruct-glaive",
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True,
    #         use_auth_token=TOKEN,
    #         init_device="cuda:0",
    #     ).eval()
    # )


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
    
    parser.add_argument(
        "--cache_location",
        type=str,
        required=False,
        default=None
    )
    
    args = parser.parse_args()

    if args.cache_location is not None:
        print("Cache dir is set to: {}".format(args.cache_location))
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", cache_dir=args.cache_location)

        model = torch.compile(
            LlamaForCausalLM.from_pretrained(
                "codellama/CodeLlama-7b-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                load_in_4bit=True,
                cache_dir=args.cache_location
            )
        )
    else:
        print("Default cache dir is used")
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

        model = torch.compile(
            LlamaForCausalLM.from_pretrained(
                "codellama/CodeLlama-7b-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                load_in_4bit=True
            )
        )

    if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    
    print("Running codellama/CodeLlama-7b-hf...")
    print("model.get_memory_footprint(): {}".format(model.get_memory_footprint()))

    # model.to(device)
    model.eval()

    result_path = args.out_path
    
    os.makedirs(result_path, exist_ok=True)
    out_path = result_path + "/eval.jsonl"

    # os.makedirs("results/replit_glaive", exist_ok=True)

    num_samples_per_task = args.pass_value

    run_eval(
        args, model, tokenizer, num_samples_per_task, out_path, generate_batch_completion
    )
