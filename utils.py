from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import einops
import textwrap
from typing import Literal
import plotly.express as px
from functools import partial
import dataclasses
from IPython.display import display, HTML
import gc
import pandas as pd
from safetensors.torch import load_file
import torch
import torch.nn as nn

from sae_lens import SAE

from enum import Enum


class MODEL_SIZE(Enum):
    M270 = "270m"
    B1 = "1b"
    B3 = "3b"
    B7 = "7b"


# load model and tokenizer
torch.set_grad_enabled(False)  # avoid blowing up mem
if torch.backends.mps.is_available():
    device = "mps"


def get_test_components(model_size: MODEL_SIZE, instructions_tuned: bool = False):
    model_name = f"google/gemma-{model_size.value}-{'it' if instructions_tuned else 'pt'}"
    sae_name = f"gemma-scope-2-{model_size.value}-{'it' if instructions_tuned else 'pt'}-transcoders-all"


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sae = SAE.from_pretrained(
        release=sae_name,  # see other options in sae_lens/pretrained_saes.yaml
        sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
        device=device,
    )

    return model, tokenizer, sae

def process_data(model, tokenizer, sae, prompts, device: str = "cuda"):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[8]  # get hidden states at layer 8
    transcodes = sae.encode(hidden_states)
    return inputs


