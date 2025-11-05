"""
Core workflow definitions for ICL attribution experiments.

This module orchestrates the evaluation process by chaining together
sub-tasks like model loading, accuracy evaluation, and attribution analysis.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

import polars as pl

from utils.attribution_utils import evaluate_attributions_for_instances
from utils.core import (
    check_arguments,
    initialize_save_path,
    load_task_list,
    locate_model_path,
)
from utils.evaluation import (
    FormatMetrics,
    MethodMetrics,
    MethodResult,
    PredictionMetrics,
    calculate_accuracy_and_save_correct_tasks,
    extract_index_distribution,
    extract_mutual_instance_for_all_methods,
    load_model_correct_task_list,
    save_correct_instance,
    sample_instance_and_reorder_exemplars,
)
from utils.model import load_model

from .config import EvaluationConfig


def load_model_bundle(model_name: str, rule: str, model_path: Path) -> tuple:
    """
    Resolve the correct checkpoint directory and load the model/tokenizer.

    For fine-tuned models, this function searches for specific subdirectory
    names to locate the best available checkpoint.
    """

    if "ft" in model_name:
        candidate_dirs = ["merged", "zero_loss_model", "best_model"]
        for subdir in candidate_dirs:
            candidate_path = model_path / subdir
            if candidate_path.exists():
                if subdir == "zero_loss_model":
                    print(f"Loading zero-loss checkpoint from {candidate_path}")
                elif subdir == "best_model":
                    print(f"Loading best checkpoint from {candidate_path}")
                return load_model(candidate_path)
        raise FileNotFoundError(
            f"No merged/zero_loss/best checkpoint found under {model_path}."
        )
    return load_model(model_path)


def evaluate_model_accuracy(
    *,
    rule: str,
    num_shots: int,
    no_disambiguation: bool,
    model,
    tokenizer,
    model_name: str,
    save_path: Path,
):
    """Evaluate model accuracy over the configured task list and persist metrics."""

    task_list = load_task_list(
        rule=rule, num_shots=num_shots, no_disamb=no_disambiguation
    )
    max_index = len(task_list)

    if "cot" in rule:
        if model_name != "gemma-2-27b-it":
            raise ValueError(
                "Chain-of-thought prompts are only supported for gemma-2-27b-it"
            )
        task_list = _append_cot_scaffold(task_list)

    prediction_metrics = PredictionMetrics(num_shots)
    format_metrics = FormatMetrics()
    correct_tasks_list = []

    calculate_accuracy_and_save_correct_tasks(
        task_list=task_list,
        max_index=max_index,
        prediction_metrics=prediction_metrics,
        format_metrics=format_metrics,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        correct_tasks_list=correct_tasks_list,
        save_path=save_path,
        rule=rule,
        num_shots=num_shots,
    )

    print("*" * 50)
    print(f"Accuracy: {prediction_metrics.accuracy}")
    print("*" * 50)


def evaluate_method_attribution(
    *,
    rule: str,
    num_shots: int,
    toy: bool,
    exemplar_reorder: bool,
    model,
    tokenizer,
    model_name: str,
    save_path: Path,
):
    """Run attribution methods and persist aggregated metrics and raw scores."""

    methods = ["input_x_gradient", "gradient_norm", "exemplar_erasure"]
    print(f"Attribution methods: {methods}")

    if exemplar_reorder:
        if num_shots != 10:
            raise ValueError("Exemplar reorder is only available for 10-shot tasks.")

        extract_mutual_instance_for_all_methods(
            save_path=save_path,
            model_name=model_name,
            rule=rule,
            num_shots=num_shots,
            methods=methods,
        )

        mutual_correct_task_path = (
            save_path.parents[2] / "mutual_correct_tasks" / model_name
        )
        mutual_json = mutual_correct_task_path / f"mutual_{rule}_dicts.json"

        if not mutual_json.exists():
            print(
                f"No mutual correct tasks for {model_name} in {rule}; skipping attribution."
            )
            return

        with open(mutual_json, "r", encoding="utf-8") as handle:
            mutual_correct_task_dict = json.load(handle)

        reordered_model_correct_task_list, original_diff_index = (
            sample_instance_and_reorder_exemplars(mutual_correct_task_dict)
        )

        if len(reordered_model_correct_task_list) != 10:
            raise AssertionError(
                "Reordered exemplar list must contain 10 tasks for attribution aggregation."
            )

        for task in reordered_model_correct_task_list:
            total_questions = 1
            method_metrics = MethodMetrics(
                {method: MethodResult() for method in methods}
            )
            method_correct_tasks_dict = {method: [] for method in methods}

            evaluate_attributions_for_instances(
                model_correct_task_list=[task],
                rule=rule,
                tokenizer=tokenizer,
                model=model,
                methods=methods,
                method_metrics=method_metrics,
                num_shots=num_shots,
                method_correct_tasks_dict=method_correct_tasks_dict,
                save_path=save_path,
                model_name=model_name,
            )

            for method in methods:
                pass

            with open(
                save_path.parents[2] / "exemplar_reorder" / "original_diff_index.json",
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {"original_diff_index": original_diff_index}, handle, indent=4
                )
        return

    model_correct_task_list = load_model_correct_task_list(
        save_path=save_path,
        model_name=model_name,
        rule=rule,
        num_shots=num_shots,
    )

    if toy:
        model_correct_task_list = random.sample(model_correct_task_list, 50)

    total_questions = len(model_correct_task_list)
    method_metrics = MethodMetrics({method: MethodResult() for method in methods})
    method_correct_tasks_dict = {method: [] for method in methods}

    calculated_correct_tasks_dict, raw_scores_df = evaluate_attributions_for_instances(
        model_correct_task_list=model_correct_task_list,
        rule=rule,
        tokenizer=tokenizer,
        model=model,
        methods=methods,
        method_metrics=method_metrics,
        num_shots=num_shots,
        method_correct_tasks_dict=method_correct_tasks_dict,
        save_path=save_path,
        model_name=model_name,
    )

    raw_scores_path = (
        save_path.parents[4] / f"{model_name}_{rule}_{num_shots}_raw_scores.parquet"
    )
    if isinstance(raw_scores_df, pl.DataFrame):
        raw_scores_df.write_parquet(raw_scores_path)

    for method in methods:
        method_correct_instance_save_path = (
            save_path.parents[2] / f"{method}_correct_tasks" / model_name
        )
        save_correct_instance(
            save_path=method_correct_instance_save_path,
            correct_tasks_dict={"tasks": calculated_correct_tasks_dict[method]},
            rule=rule,
            num_shots=num_shots,
        )
        extract_index_distribution(
            save_path=method_correct_instance_save_path,
            rule=rule,
            num_shots=num_shots,
            method=method,
        )


def run_evaluation(config: EvaluationConfig) -> None:
    """
    Entry point used by the CLI to execute an evaluation run.

    Initializes the configuration, loads the model, and runs the selected
    evaluation workflows (model accuracy and/or method attribution).
    """

    if not (config.model_accuracy or config.method_accuracy):
        raise ValueError(
            "Enable at least one of --model-accuracy or --method-accuracy."
        )

    check_arguments(model_name=config.model_name, rule=config.rule)

    model_path = locate_model_path(model_name=config.model_name, rule=config.rule)
    results_root = config.resolved_output_dir()

    save_path = initialize_save_path(
        toy=config.toy,
        parent_path=results_root,
        exp_date=config.exp_date,
        exp_name=config.run_identifier,
        rule=config.rule,
        model_name=config.model_name,
        num_shots=config.num_shots,
    )

    print("Configuration summary")
    print("-" * 50)
    print(f"Rule: {config.rule}")
    print(f"Num shots: {config.num_shots}")
    print(f"Model name: {config.model_name}")
    print(f"Model path: {model_path}")
    print(f"Result save path: {save_path}")
    print(f"Model accuracy: {config.model_accuracy}")
    print(f"Method accuracy: {config.method_accuracy}")
    print(f"No disambiguation exemplar: {config.no_disambiguation}")
    print(f"Exemplar reorder: {config.exemplar_reorder}")
    print("-" * 50)

    random.seed(68)

    model, tokenizer = load_model_bundle(
        model_name=config.model_name,
        rule=config.rule,
        model_path=model_path,
    )

    if config.model_accuracy:
        evaluate_model_accuracy(
            rule=config.rule,
            num_shots=config.num_shots,
            no_disambiguation=config.no_disambiguation,
            model=model,
            tokenizer=tokenizer,
            model_name=config.model_name,
            save_path=save_path,
        )

    if config.method_accuracy:
        evaluate_method_attribution(
            rule=config.rule,
            num_shots=config.num_shots,
            toy=config.toy,
            exemplar_reorder=config.exemplar_reorder,
            model=model,
            tokenizer=tokenizer,
            model_name=config.model_name,
            save_path=save_path,
        )

    print("Evaluation complete.\n")


def _append_cot_scaffold(task_list: list[dict]) -> list[dict]:
    """Augment tasks with a chain-of-thought scaffold."""

    augmented = []
    for instance in task_list:
        exemplars, target_question = _craft_chat_template(
            instance["exemplars"], instance["target_question"]
        )
        instance["exemplars"] = exemplars
        instance["target_question"] = target_question
        augmented.append(instance)
    return augmented


def _craft_chat_template(examples: str, target_question: str) -> Tuple[str, str]:
    cot_instruction = (
        "\n\nSolve this problem step by step, Generate the content of <ANSWER> after "
        "'So the answer is: '"
    )
    crafted_target_questions = (
        target_question
        + " <ANSWER>"
        + cot_instruction
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    crafted_examples = "<start_of_turn>user\n" + examples
    return crafted_examples, crafted_target_questions
