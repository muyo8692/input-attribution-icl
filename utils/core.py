from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import copy


SUPPORTED_MODELS = {
    "llama-2-7b",
    "llama-2-13b",
    "gemma-2-2b",
    "gemma-2-9b",
    "gemma-2-27b",
    "llama-2-7b-ft",
    "llama-2-13b-ft",
    "llama-3.1-8b-ft",
    "gemma-2-2b-ft",
    "gemma-2-9b-ft",
    "gemma-2-27b-ft",
    "gemma-2-27b-it",
    "mistral-7b",
    "mistral-7b-ft",
}


RULE_TO_DATASET_DIR: Dict[str, str] = {
    "distinct": "linear_or_distinct",
    "linear": "linear_or_distinct",
    "add": "add_or_multiply",
    "multiply": "add_or_multiply",
    "associative_recall": "associative_recall",
    "verb": "verb_object",
    "object": "verb_object",
    "tense": "tense_article",
    "article": "tense_article",
    "title": "pos_title",
    "pos": "pos_title",
}


def check_arguments(*, model_name: str, rule: str) -> None:
    """Validate model and rule identifiers against supported options."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    if rule not in RULE_TO_DATASET_DIR:
        raise ValueError(f"Unsupported rule: {rule}")


def locate_model_path(model_name: str, rule: str | None = None) -> Path:
    """Return the path or model identifier used for loading weights."""

    base_model_map = {
        "llama-2-7b": "meta-llama/Llama-2-7b-hf",
        "llama-2-13b": "meta-llama/Llama-2-13b-hf",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-27b": "google/gemma-2-27b",
        "gemma-2-27b-it": "google/gemma-2-27b-it",
        "mistral-7b": "mistralai/Mistral-7B-v0.3",
    }

    if "ft" not in model_name:
        return Path(base_model_map[model_name])

    if rule is None or rule not in RULE_TO_DATASET_DIR:
        raise ValueError(
            "Fine-tuned model lookup requires a supported rule identifier."
        )

    suffix = RULE_TO_DATASET_DIR[rule]
    return (
        Path("./work00/muyo/example_saliency/fine-tuned-models")
        / f"{model_name}-lora"
        / suffix
    )


def initialize_save_path(
    *,
    toy: bool,
    parent_path: Path | str,
    exp_date: str,
    exp_name: str,
    rule: str,
    model_name: str,
    num_shots: int,
) -> Path:
    """Create (if needed) and return the directory for saving outputs."""

    parent = Path(parent_path)
    run_name = f"{exp_name}_toy" if toy else exp_name
    save_path = parent / exp_date / run_name / rule / model_name / f"{num_shots}_shots"
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def load_task_list(rule: str, num_shots: int, no_disamb: bool) -> List[Dict[str, Any]]:
    """Return the list of evaluation tasks for the given rule and shot count."""

    if rule not in RULE_TO_DATASET_DIR:
        raise ValueError(f"Unsupported rule: {rule}")

    effective_shots = 10 if num_shots == 0 else num_shots
    tasks_root = Path(__file__).resolve().parents[1] / "tasks"
    dataset_dir = tasks_root / RULE_TO_DATASET_DIR[rule] / f"{effective_shots}_shots"
    file_path = dataset_dir / f"generated_{rule}_data.json"

    with open(file_path, "r", encoding="utf-8") as handle:
        task_list = json.load(handle)["tasks"]

    if no_disamb:
        task_list_no_disamb = copy.deepcopy(task_list)
        for task, task_work in zip(task_list, task_list_no_disamb):
            exemplars = task["exemplars"]
            disamb_index = int(task["exemplar_index_for_diff"])
            disamb_line = exemplars.split("\n")[disamb_index]
            task_work["exemplars"] = exemplars.replace(disamb_line + "\n", "")
        return task_list_no_disamb

    if num_shots == 0:
        task_list_zero_shot = copy.deepcopy(task_list)
        for task in task_list_zero_shot:
            task["exemplars"] = ""
        return task_list_zero_shot

    return task_list
