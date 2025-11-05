from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_path):
    """Load a causal LM and tokenizer from the specified path or identifier."""

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "gemma-2-2b" in str(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="sequential", attn_implementation="eager"
        )
    elif "gemma" in str(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", attn_implementation="eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    return model, tokenizer
