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
    model_name = f"google/gemma-3-{model_size.value}-{'it' if instructions_tuned else 'pt'}"
    sae_name = f"google/gemma-scope-2-{model_size.value}-{'it' if instructions_tuned else 'pt'}"

    from sae_lens import SAE

    release = "gemma-scope-2-270m-it-res"
    sae_id = "layer_12_width_262k_l0_medium_seed_1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sae = SAE.from_pretrained(release, sae_id)
    return model, tokenizer, sae


if __name__ == "__main__":

    get_test_components(MODEL_SIZE.M270, instructions_tuned=True)
