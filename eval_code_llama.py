from transformers import (
    CodeLlamaTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    # AutoTokenizer
)
import argparse
from core import filter_code, run_eval, fix_indents
import os
import torch
from submitit_utils import str2bool
# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""

torch.manual_seed(1234)

@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size, logit_processor_lst, use_watermark, **gen_kwargs
) -> list[str]:
    # input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    if use_watermark:
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=batch_size,
            logits_processor=logit_processor_lst,
            **gen_kwargs
        )
    else:
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,  # model has no pad token
            num_return_sequences=batch_size
        )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "results/code_llama/eval.jsonl"
    os.makedirs("results/code_llama", exist_ok=True)

    tokenizer = CodeLlamaTokenizer.from_pretrained(
        "codellama/CodeLlama-7b-hf",
    )

    model = torch.compile(
        LlamaForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )

    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")

    parser.add_argument(
        "--use_watermark",
        type=str2bool,
        default=False,
        help=("Whether to use watermark"),
    )

    parser.add_argument(
        "--initial_seed",
        type=int,
        default=1234,
        help=("The initial seed to use in the blacklist randomization process.", 
              "Is unused if the process is markov generally. Can be None."),
    )
    parser.add_argument(
        "--dynamic_seed",
        type=str,
        default="markov_1",
        choices=[None, "initial", "markov_1"],
        help="The seeding procedure to use when sampling the blacklist at each step.",
    )
    parser.add_argument(
        "--bl_proportion",
        type=float,
        default=0.5,
        help="The ratio of blacklist to whitelist tokens when splitting the vocabulary",
    )
    parser.add_argument(
        "--bl_logit_bias",
        type=float,
        default=1.0,
        help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary at every step",
    )
    parser.add_argument(
        "--bl_type",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="The type of blacklisting being performed.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        # default=8,
        help="ngram size to force the model not to generate, can't be too small or model is handicapped, too large and blows up in complexity.",
    )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        default=False,
        help="Whether to use early stopping, only for beam search.",
    )
    
    parser.add_argument(
        "--store_bl_ids",
        type=str2bool,
        default=False,
        help=("Whether to store all the blacklists while generating with bl processor. "),
    )
    parser.add_argument(
        "--store_spike_ents",
        type=str2bool,
        default=False,
        help=("Whether to store the spike entropies while generating with bl processor. "),
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="The temperature to use when generating using multinom sampling",
    )
  
    parser.add_argument(
        "--all_gas_no_eos",
        type=str2bool,
        default=False,
        help=("Whether to weight the EOS token as -inf"),
    )
    
    parser.add_argument(
        "--out_path",
        type=str,
        default=False,
        help=("Result path"),
    )
    
    args = parser.parse_args()

    result_path = args.out_path
    
    os.makedirs(result_path, exist_ok=True)
    out_path = result_path + "/eval.jsonl"

    # os.makedirs("results/replit_glaive", exist_ok=True)

    run_eval(
        args, model, tokenizer, num_samples_per_task, out_path, generate_batch_completion
    )

    # run_eval(
    #     model,
    #     tokenizer,
    #     num_samples_per_task,
    #     out_path,
    #     generate_batch_completion,
    #     True,
    # )
