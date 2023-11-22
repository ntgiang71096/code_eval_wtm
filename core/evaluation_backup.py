from human_eval.data import write_jsonl, read_problems
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    LogitsProcessorList
)
from tqdm import tqdm
import itertools
import typing

from tokenizers import Tokenizer

from functools import partial

from watermark import BlacklistLogitsProcessor

import torch

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list
]


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def split_batch(samples: list, size=4):
    mini_batches = []

    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches


def run_eval(
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    format_tabs: bool = False,
):
    logit_processor_lst = [], 
    gen_kwargs = {}

    if args.use_watermark == True:
        print("Using watermark")
        # Giang: constructing pipeline for watermark

        all_token_ids = list(tokenizer.get_vocab().values())
        vocab_size = len(all_token_ids)
        print(f"Vocabulary size: {vocab_size}")

        init_seed = args.initial_seed
        dyna_seed=args.dynamic_seed # type not value
        bl_proportion = args.bl_proportion
        bl_logit_bias = args.bl_logit_bias
        bl_type = args.bl_type
        n_beams = args.num_beams
        early_stopping = args.early_stopping
        no_repeat_ngram_size = args.no_repeat_ngram_size
        store_bl_ids = args.store_bl_ids
        store_spike_ents = args.store_spike_ents

        bl_processor = BlacklistLogitsProcessor(bad_words_ids=None, 
                                                store_bl_ids=store_bl_ids, 
                                                store_spike_ents=store_spike_ents, 
                                                eos_token_id=tokenizer.eos_token_id, 
                                                vocab=all_token_ids, 
                                                vocab_size=vocab_size, 
                                                bl_proportion=bl_proportion,
                                                bl_logit_bias=bl_logit_bias,
                                                bl_type=bl_type, 
                                                initial_seed=init_seed, 
                                                dynamic_seed=dyna_seed)                                           

        logit_processor_lst = LogitsProcessorList([bl_processor])

        
        # Greedy and basic beam search, default
        gen_kwargs = dict(
            num_beams=n_beams,
        )
        if n_beams > 1:
            # these are only for beam search repetition correction
            if no_repeat_ngram_size > 0:
                gen_kwargs.update(dict(no_repeat_ngram_size=no_repeat_ngram_size))
            gen_kwargs.update(dict(early_stopping=early_stopping))

        if args.use_sampling:
            gen_kwargs.update(dict(do_sample=True,
                                    top_k=0,
                                    temperature=args.sampling_temp))
        if args.all_gas_no_eos:
            gen_kwargs.update(dict(suppress_tokens=[tokenizer.eos_token_id]))
        
        # generate_without_blacklist = partial(
        #     model.generate,
        #     **gen_kwargs
        # )
        # generate_with_blacklist = partial(
        #     model.generate,
        #     logits_processor=logit_processor_lst, 
        #     **gen_kwargs
        # )

        # hf_model_name = args.model_name

        # Construct the data filtering/sampling scheme partials
        # token_kwargs = dict(
        #     tokenizer=tokenizer,
        #     model=model,
        # )
        # if args.input_truncation_strategy == "prompt_length":
        #     token_kwargs.update(dict(min_prompt_tokens=min_prompt_tokens))
        # elif args.input_truncation_strategy == "completion_length":
        #     token_kwargs.update(dict(max_new_tokens=max_new_tokens))
        # else:
        #     ValueError(f"Unknown input truncation strategy {args.input_truncation_strategy}")


        # giang: create function for tokenizing prompt
                
        # tokenize_prompts = partial(
        #     tokenize_for_generation,
        #     **token_kwargs
        # )

        # input_check_kwargs = dict(
        #     # min_sample_len = min_prompt_tokens + max_new_tokens,
        #     min_sample_len = args.min_sample_tokens, # first line is a bug sometimes with large amounts
        # )
        # if args.input_filtering_strategy == "prompt_length":
        #     input_check_kwargs.update(dict(min_prompt_len = min_prompt_tokens,
        #                                    min_completion_len = 0))
        # elif args.input_filtering_strategy == "completion_length":
        #     input_check_kwargs.update(dict(min_prompt_len = 0,
        #                                    min_completion_len = max_new_tokens))
        # elif args.input_filtering_strategy == "prompt_and_completion_length":
        #     input_check_kwargs.update(dict(min_prompt_len = min_prompt_tokens,
        #                                    min_completion_len = max_new_tokens))
        # else:
        #     ValueError(f"Unknown input filtering strategy {args.input_filtering_strategy}")

        # # giang: create function to check input length

        # input_check = partial(
        #     check_input_lengths,
        #     **input_check_kwargs
        # )

        # if args.output_filtering_strategy == "max_new_tokens":
        #     output_kwargs = dict(min_output_len = max_new_tokens)
        # elif args.output_filtering_strategy == "no_filter":
        #     output_kwargs = dict(min_output_len = 0)
        # else:
        #     ValueError(f"Unknown output filtering strategy {args.output_filtering_strategy}")

        # # giang: create function to check output length

        # output_check = partial(
        #     check_output_lengths,
        #     **output_kwargs
        # )

        # Giang: finish constructing pipeline

    else:
        print("Not using watermark")

    problems = read_problems()
    # problems = dict(itertools.islice(problems.items(), 20))
    samples = []
    pbar = tqdm(total=len(problems))

    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task, logit_processor_lst, args.use_watermark, **gen_kwargs
        )

        for sample in batch_completions:
            print(sample)
            print("-" * 64)
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(1)

        break
    
    write_jsonl(out_path, samples)
