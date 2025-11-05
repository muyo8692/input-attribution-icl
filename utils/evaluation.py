"""
Utilities for evaluating model performance and attribution methods.

This module provides data structures and functions for calculating,
aggregating, and saving evaluation metrics for both model accuracy and
attribution methods.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

from .core import RULE_TO_DATASET_DIR


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class FormatMetrics:
    def __init__(self) -> None:
        self.total = 0
        self.correct_space = 0
        self.correct_newline = 0

    def increment_correct_space(self) -> None:
        self.correct_space += 1

    def increment_correct_newline(self) -> None:
        self.correct_newline += 1

    def increment_total(self) -> None:
        self.total += 1

    @property
    def space_proportion(self) -> float:
        return self.correct_space / self.total if self.total else 0

    @property
    def newline_proportion(self) -> float:
        return self.correct_newline / self.total if self.total else 0


class PredictionMetrics:
    def __init__(self, num_shots: int) -> None:
        self.total = 0
        self.correct = 0
        self.prob = 0
        self.place = 0
        self.foil_prob = 0
        self.position_counts = [0 for _ in range(num_shots)]
        self.num_shots = num_shots

    def increment_correct(self, position: int) -> None:
        self.correct += 1
        if self.num_shots != 0:
            self.position_counts[position] += 1

    def increment_prob(self, current_prob) -> None:
        self.prob += current_prob

    def increment_foil_prob(self, current_foil_prob) -> None:
        self.foil_prob += current_foil_prob

    def increment_place(self, current_place) -> None:
        self.place += current_place

    def increment_total(self) -> None:
        self.total += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0

    def position_accuracy(self, position: int) -> float:
        return (
            self.position_counts[position] / (self.total / self.num_shots)
            if self.total
            else 0
        )

    @property
    def average_prob(self) -> float:
        return self.prob / self.total if self.total else 0

    @property
    def average_place(self) -> float:
        return self.place / self.total if self.total else 0

    @property
    def average_foil_prob(self) -> float:
        return self.foil_prob / self.total if self.total else 0


@dataclass
class MethodResult:
    mean: Dict[str, Any] = field(
        default_factory=lambda: MethodResult._initialize_nested_dict()
    )
    last_step: Dict[str, Any] = field(
        default_factory=lambda: MethodResult._initialize_nested_dict()
    )

    @staticmethod
    def _initialize_nested_dict() -> Dict[str, Any]:
        return {
            "add": {
                "divide": {"base": 0, "contrast": 0},
                "not_divide": {"base": 0, "contrast": 0},
            },
            "multiply": {
                "divide": {"base": 0, "contrast": 0},
                "not_divide": {"base": 0, "contrast": 0},
            },
        }


@dataclass
class MethodMetrics:
    methods: Dict[str, MethodResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Correct-task utilities
# ---------------------------------------------------------------------------


def load_model_correct_task_list(
    save_path: Path, model_name: str, rule: str, num_shots: int
) -> List[Dict[str, Any]]:
    if rule not in RULE_TO_DATASET_DIR:
        raise ValueError(f"Unsupported rule: {rule}")

    base_dir = save_path.parents[2] / "correct_tasks" / model_name
    folder = RULE_TO_DATASET_DIR[rule]
    file_path = base_dir / folder / f"{num_shots}_shots" / f"correct_{rule}_data.json"

    with open(file_path, "r", encoding="utf-8") as handle:
        data_dict = json.load(handle)
    return data_dict["tasks"]


def save_correct_instance(
    *,
    save_path: Path,
    correct_tasks_dict: Dict[str, Any],
    rule: str,
    num_shots: int,
) -> None:
    if rule not in RULE_TO_DATASET_DIR:
        raise ValueError(f"Unsupported rule: {rule}")

    file_path = save_path / RULE_TO_DATASET_DIR[rule] / f"{num_shots}_shots"
    file_path.mkdir(parents=True, exist_ok=True)

    with open(file_path / f"correct_{rule}_data.json", "w", encoding="utf-8") as handle:
        json.dump(correct_tasks_dict, handle, ensure_ascii=False, indent=4)


def extract_index_distribution(
    save_path: Path, rule: str, num_shots: int, method: str
) -> None:
    folder = RULE_TO_DATASET_DIR[rule]
    file_path = save_path / folder / f"{num_shots}_shots"

    exemplar_counts = {i: 0 for i in range(num_shots)}

    with open(file_path / f"correct_{rule}_data.json", "r", encoding="utf-8") as handle:
        correct_instance_dict = json.load(handle)

    for instance in correct_instance_dict["tasks"]:
        exemplar_index = instance["exemplars_index_for_diff"]
        exemplar_counts[exemplar_index] += 1

    with open(
        file_path / f"{method}_{rule}_exemplars_index_distribution.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(exemplar_counts, handle, indent=4)


def extract_mutual_instance_for_all_methods(
    save_path: Path,
    model_name: str,
    rule: str,
    num_shots: int,
    methods: List[str],
) -> None:
    folder = RULE_TO_DATASET_DIR[rule]
    correct_instance_dict_per_method = {method: None for method in methods}

    for method in methods:
        file_path = (
            save_path.parents[2]
            / f"{method}_correct_tasks"
            / model_name
            / folder
            / f"{num_shots}_shots"
        )

        with open(
            file_path / f"correct_{rule}_data.json", "r", encoding="utf-8"
        ) as handle:
            correct_instance_dict = json.load(handle)
            correct_instance_dict_per_method[method] = correct_instance_dict["tasks"]

    log_dict: Dict[str, Any] = {}
    first_key = next(iter(correct_instance_dict_per_method))
    mutual_dicts = set(
        tuple(sorted(d.items())) for d in correct_instance_dict_per_method[first_key]
    )
    log_dict["initial_count"] = len(mutual_dicts)

    for key, task_list in correct_instance_dict_per_method.items():
        if key == first_key:
            continue
        current_set = set(tuple(sorted(d.items())) for d in task_list)
        mutual_dicts &= current_set
        log_dict[f"after_{key}"] = len(mutual_dicts)

    mutual_dicts_list = [dict(d) for d in mutual_dicts]

    output_dir = save_path.parents[2] / "mutual_correct_tasks" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mutual_dicts_list:
        with open(
            output_dir / f"no_mutual_{rule}_dicts.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(log_dict, handle, indent=4)
    else:
        payload = {"len": len(mutual_dicts_list), "tasks": mutual_dicts_list}
        with open(
            output_dir / f"mutual_{rule}_dicts.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=4)


def sample_instance_and_reorder_exemplars(
    mutual_correct_task_dict: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int]:
    random.seed(68)

    instance_to_reorder = random.choice(mutual_correct_task_dict["tasks"])
    original_diff_index = instance_to_reorder["exemplars_index_for_diff"]

    exemplars_list = instance_to_reorder["exemplars"].split("\n")[:-1]
    if len(exemplars_list) != 10:
        raise AssertionError("Expected 10 exemplars for reorder experiments.")

    value_to_reposition = exemplars_list[original_diff_index]
    exemplars_list.remove(value_to_reposition)

    resulting_lists = []
    for i in range(len(exemplars_list) + 1):
        new_list = exemplars_list[:]
        new_list.insert(i, value_to_reposition)

        resulting_dict = {
            "exemplars": "\n".join(new_list) + "\n",
            "exemplars_index_for_diff": i,
            "foil_answer": instance_to_reorder["foil_answer"],
            "model_answer": instance_to_reorder["model_answer"],
            "target_answer": instance_to_reorder["target_answer"],
            "target_question": instance_to_reorder["target_question"],
        }
        resulting_lists.append(resulting_dict)

    return resulting_lists, original_diff_index


# ---------------------------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------------------------


def get_model_output(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int,
    model_name: str,
):
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    if "ft" not in model_name:
        new_line_token_id = tokenizer("\n")["input_ids"][-1]
    elif "gemma" in model_name or "mistral" in model_name:
        new_line_token_id = tokenizer("\n")["input_ids"][-1]
    else:
        new_line_token_id = tokenizer.eos_token_id

    if "ft" in model_name and "llama-3.1" in model_name:
        outputs = model.generate(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=new_line_token_id,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **inputs,
        )
    elif model_name == "gemma-2-27b-it":
        outputs = model.generate(
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_scores=True,
            return_dict_in_generate=True,
            **inputs,
        )
    elif "mistral" in model_name:
        outputs = model.generate(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=new_line_token_id,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **inputs,
        )
    else:
        outputs = model.generate(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=new_line_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **inputs,
        )

    model_output = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
    if model_name == "gemma-2-27b-it":
        try:
            model_answer = model_output.split("So the answer is: ")[-1].rstrip("\n")
        except IndexError:
            print(f"Wrong Model output: {model_output}")
            model_answer = None
    else:
        model_answer = model_output.split(": ")[-1].split("\n")[0]

    target_answer_prob = -999
    foil_answer_prob = -999

    return model_output, model_answer, target_answer_prob, foil_answer_prob


def check_output_format_func(model_full_output: str) -> Tuple[str, str]:
    if model_full_output.split(":")[-1].startswith(" "):
        space_flag = "yes"
    else:
        space_flag = "no"
    if model_full_output.endswith("\n"):
        newline_flag = "yes"
    else:
        newline_flag = "no"

    return space_flag, newline_flag


def calculate_accuracy_and_save_correct_tasks(
    *,
    task_list: List[Dict[str, Any]],
    max_index: int,
    prediction_metrics: PredictionMetrics,
    format_metrics: FormatMetrics,
    model,
    tokenizer,
    model_name: str,
    correct_tasks_list: List[Dict[str, Any]],
    save_path: Path,
    rule: str,
    num_shots: int,
) -> None:
    flag = False
    for _, instance in tqdm(
        enumerate(task_list[:max_index]),
        total=len(task_list[:max_index]),
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        prediction_metrics.increment_total()
        format_metrics.increment_total()
        correct_instance_dict: Dict[str, Any] = {}

        input_text = instance["exemplars"] + instance["target_question"]
        target_answer = instance["target_answer"]
        foil_answer = instance["foil_answer"]
        correct_exemplar_index = instance["exemplar_index_for_diff"]

        (
            model_full_output,
            model_answer,
            target_answer_prob,
            foil_answer_prob,
        ) = get_model_output(
            model=model,
            tokenizer=tokenizer,
            question=input_text,
            max_new_tokens=20,
            model_name=model_name,
        )

        prediction_metrics.increment_prob(target_answer_prob)
        prediction_metrics.increment_foil_prob(foil_answer_prob)

        if len(model_answer) > len(target_answer) and model_answer.startswith(
            target_answer
        ):
            model_answer = model_answer[: len(target_answer)]
        if model_answer.startswith("**"):
            model_answer = model_answer.split("**")[1]

        is_space, is_newline = check_output_format_func(model_full_output)
        if is_space == "yes":
            format_metrics.increment_correct_space()
        if is_newline == "yes":
            format_metrics.increment_correct_newline()

        if model_answer == target_answer:
            if num_shots != 0:
                prediction_metrics.increment_correct(correct_exemplar_index)
            else:
                prediction_metrics.increment_correct(0)

            correct_instance_dict["exemplars"] = instance["exemplars"]
            correct_instance_dict["target_question"] = instance["target_question"]
            correct_instance_dict["target_answer"] = instance["target_answer"]
            correct_instance_dict["foil_answer"] = foil_answer
            correct_instance_dict["exemplars_index_for_diff"] = correct_exemplar_index
            correct_instance_dict["model_answer"] = model_answer

            correct_tasks_list.append(correct_instance_dict)
        elif not flag:
            full_output_ids = tokenizer(model_full_output)["input_ids"]
            print(
                f"Output format checker:\n{tokenizer.convert_ids_to_tokens(full_output_ids)}"
            )
            print(f"Model output: {model_full_output}")
            print(f"Extracted answer: {model_answer}")
            flag = True

    correct_tasks_dict = {"tasks": correct_tasks_list}
    correct_instance_save_path = save_path.parents[2] / "correct_tasks" / model_name
    save_correct_instance(
        save_path=correct_instance_save_path,
        correct_tasks_dict=correct_tasks_dict,
        rule=rule,
        num_shots=num_shots,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
