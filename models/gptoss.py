import torch
import numpy as np
from jax import grad,vmap
from tqdm import tqdm
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from data.serialize import serialize_arr, deserialize_str, SerializerSettings
from loguru import logger
import os

os.environ["CUDA_HOME"] = "/data2/InstallFolder"
os.environ["PATH"] = f"/data2/InstallFolder/bin:{os.environ['PATH']}"
os.environ["LD_LIBRARY_PATH"] = f"/data2/InstallFolder/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["C_INCLUDE_PATH"] = f"/data2/InstallFolder/include:{os.environ.get('C_INCLUDE_PATH', '')}"
os.environ["CPLUS_INCLUDE_PATH"] = f"/data2/InstallFolder/include:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"


DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
MODEL_PATH = "/data2/amir/HF_home/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"

loaded = {}

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        # device_map = "auto",
        # dtype="auto",
        local_files_only=True
    )
    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_model_and_tokenizer(model_name, cache_model=False):
    if model_name in loaded:
        return loaded[model_name]
    tokenizer = get_tokenizer()
   
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",device_map="cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        # offload_state_dict=True,      # Offload weights not in use
        # offload_folder="./offload",  # Offload to disk if needed
        max_memory={0: "45GB", 1: "45GB"}  # Respect available memory
    )
    model.eval()
    if cache_model:
        loaded[model_name] = model, tokenizer
    return model, tokenizer

def tokenize_fn(str, model):
    tokenizer = get_tokenizer()
    return tokenizer(str)

def gptoss_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1, cache_model=True):
    """ Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM 
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
        cache_model: whether to cache the model and tokenizer for faster repeated calls
    Returns: NLL/D
    """
    model, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    full_series = input_str + target_str
    
    batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        logger.debug(batch.keys())
        out = model(**batch)

    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out['logits'][:,:,bad_tokens] = -100

    input_ids = batch['input_ids'][0][1:]
    input_ids = input_ids.to('cpu')
    logprobs = torch.nn.functional.log_softmax(out['logits'], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].float().cpu().numpy()
    # logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()


    tokens = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=False, 
        clean_up_tokenization_spaces=False
    )
    
    input_len = len(tokenizer([input_str], return_tensors="pt",)['input_ids'][0])
    input_len = input_len - 2 # remove the BOS token

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum()/len(target_arr)

    #print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()

    return transformed_nll-avg_logdet_dydx

def gptoss_completion_fn(
    model,
    input_str,
    steps,
    settings,
    batch_size=5,
    num_samples=20,
    temp=0.9, 
    top_p=0.9,
    cache_model=True,
    **kwargs
):
    
    if kwargs.get('extra_input', None) is not None:
        input_str = kwargs['extra_input'] + input_str
    if kwargs.get('chatgpt_sys_message', None) is not None:
        input_str =  kwargs['chatgpt_sys_message'] + input_str

    
    avg_tokens_per_step = len(tokenize_fn(input_str, model)['input_ids']) / len(input_str.split(settings.time_sep))
    max_tokens = int(avg_tokens_per_step*steps)

    model, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    gen_strs = []
    logger.debug(f"Batch size: {batch_size}")
    logger.debug(f"Num samples: {num_samples}")
    logger.debug(f"num_samples // batchsize: {num_samples // batch_size}")

    for _ in tqdm(range(num_samples // batch_size)):
        logger.debug(f'input_str: {input_str}')
        batch = tokenizer(
            [input_str],
            return_tensors="pt",
        ).to(model.device)

        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        # batch = {k: v.cpu() for k, v in batch.items()}
        num_input_ids = batch['input_ids'].shape[1]

        # good_tokens_str = list("0123456789" + settings.time_sep)
        # good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
        # good_tokens += [tokenizer.eos_token_id]
        good_tokens = [
            tok_id for tok, tok_id in tokenizer.get_vocab().items()
            if tok.strip().isdigit() or tok == settings.time_sep
        ]

        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp, 
            top_p=top_p, 
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )
        gen_strs += tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        gen_strs = [s for s in gen_strs if s.strip()]
        logger.debug(f"Generated strings: {gen_strs}")

    return gen_strs
