from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)
import torch

from codegen.model import SantaCoder, HFTorchDecoder
from eval_codellama import generate_batch_completion as codelama_gen
from evalplus_polycoder import generate_batch_completion as polycoder_gen
from evalplus_santacoder import generate_batch_completion as santacoder_gen


CODELLAMA_MODEL_NAME = "codellama/CodeLlama-7b-hf"
SANTACODER_MODEL_NAME = "bigcode/santacoder"
POLYCODER_MODEL_NAME = "NinedayWang/PolyCoder-2.7B"

def init_codellama_model(gpu, cache_location):
    if cache_location is not None:
        print("Cache dir is set to: {}".format(cache_location))
    else:
        print("Default cache dir is used")
        tokenizer = AutoTokenizer.from_pretrained(CODELLAMA_MODEL_NAME, cache_dir=cache_location)

        model = torch.compile(
            LlamaForCausalLM.from_pretrained(
                CODELLAMA_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                load_in_4bit=True,
                cache_dir=cache_location
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

    return model, tokenizer


def init_santacoder(gpu, cache_location):
    batch_size = 10
    temperature = 0.8

    model = SantaCoder(
            batch_size=batch_size, name=SANTACODER_MODEL_NAME, temperature=temperature, gpu=gpu, cache_location=cache_location, load_in_8bit=False
        )

    print("Finish loading model")

    return model, model.tokenizer 


def init_polycoder(gpu, cache_location):
    batch_size = 10
    temperature = 0.8


    model = HFTorchDecoder(
            batch_size=batch_size,
            name=POLYCODER_MODEL_NAME,
            temperature=temperature,
            gpu=gpu,
            load_in_8bit=True
        )

    print("Finish loading model")

    return model, model.tokenizer


def init_tokenizer_and_model(model_name, gpu, cache_location):
    print("Model name: {}".format(model_name))
    if model_name == CODELLAMA_MODEL_NAME:
        return init_codellama_model(gpu, cache_location)
    elif model_name == SANTACODER_MODEL_NAME:
        return init_santacoder(gpu, cache_location)

    elif model_name == POLYCODER_MODEL_NAME:
        return init_polycoder(gpu, cache_location)

    else:
        raise Exception("Model name is not correct. Select between codellama/CodeLlama-7b-hf, bigcode/santacoder, NinedayWang/PolyCoder-2.7B")
    

def get_gen_function_by_name(model_name):
    if model_name == CODELLAMA_MODEL_NAME:
        return codelama_gen
    elif model_name == SANTACODER_MODEL_NAME:
        return santacoder_gen
    elif model_name == POLYCODER_MODEL_NAME:
        return polycoder_gen
    else:
        raise Exception("Model name is not valid")