"""
Core module for calculating saliency and attribution metrics.

This module provides functions to calculate attribution scores
for model predictions. It supports multiple methods such as
input_x_gradient, gradient_norm, erasure, exemplar_erasure, embedding cosine similarity,
and gradient cosine similarity.

The main entry point in this file is the function
`evaluate_attributions_for_instances`, which is used by the main script.
"""

import json
import warnings

import numpy as np
import torch

from lm_saliency import (
    erasure_scores,
    exemplar_based_erasure,
    input_x_gradient,
    l1_grad_norm,
    saliency,
)
from tqdm import tqdm
import polars as pl

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def create_ids_list(tokenizer, input_text_for_attribution, correct_text, foil_text):
    """
    Tokenizes input text and returns lists of input IDs, attention masks,
    correct token IDs, and foil token IDs for attribution calculations.
    """
    original_input_ids = tokenizer(input_text_for_attribution)["input_ids"]
    original_attention_mask = tokenizer(input_text_for_attribution)["attention_mask"]
    original_correct_ids = tokenizer(correct_text, add_special_tokens=False)[
        "input_ids"
    ]
    original_foil_ids = tokenizer(foil_text, add_special_tokens=False)["input_ids"]

    # If tokenizations differ in length, adjust by prepending a space.
    if len(original_correct_ids) != len(original_foil_ids):
        if len(original_correct_ids) > len(original_foil_ids):
            correct_text = " " + correct_text
        else:
            foil_text = " " + foil_text
        original_correct_ids = tokenizer(correct_text, add_special_tokens=False)[
            "input_ids"
        ]
        original_foil_ids = tokenizer(foil_text, add_special_tokens=False)["input_ids"]

    # For single-token cases, return one-item lists.
    if len(original_correct_ids) == 1:
        return (
            [original_input_ids],
            [original_attention_mask],
            [original_correct_ids],
            [original_foil_ids],
        )

    # Otherwise, build lists for each tokenization step.
    input_ids_list = [original_input_ids]
    attention_masks_list = [original_attention_mask]
    correct_ids_list = [original_correct_ids[0]]
    foil_ids_list = [original_foil_ids[0]]

    # Use the space token ID from tokenizing "a "
    space_id = tokenizer("a ")["input_ids"][-1]
    for i in range(1, len(original_correct_ids)):
        input_ids = original_input_ids[:-1] + original_correct_ids[:i] + [space_id]
        attention_mask = original_attention_mask + [1] * i
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        correct_ids_list.append(original_correct_ids[i])
        foil_ids_list.append(original_foil_ids[i])

    # Ensure all lists have matching lengths.
    assert (
        len(input_ids_list)
        == len(attention_masks_list)
        == len(correct_ids_list)
        == len(foil_ids_list)
    ), "Mismatch in lengths of token lists."
    assert len(input_ids_list) == len(original_correct_ids), (
        "Input IDs list length incorrect."
    )
    return input_ids_list, attention_masks_list, correct_ids_list, foil_ids_list


def slice_array_and_calculate_mean(array_list):
    """
    Slices all arrays in `array_list` to the minimum length,
    stacks them, and returns:
      - the minimum length,
      - the mean array,
      - the array from the last step.
    """
    lengths = [array.shape[0] for array in array_list]
    min_length = min(lengths)
    sliced_arrays = [array[:min_length] for array in array_list]
    stacked_array = np.stack(sliced_arrays, axis=0)
    mean_array = np.mean(stacked_array, axis=0)
    last_step_array = sliced_arrays[-1]
    return min_length, mean_array, last_step_array


def calculate_embedding_norm(model, tokenizer, text):
    """
    Calculates the norm of the token embeddings for the given text.
    """
    input_ids = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**input_ids, output_hidden_states=True)
        embeddings = outputs.hidden_states[0]
    return torch.norm(embeddings, dim=-1)


# ---------------------------------------------------------------------------
# Saliency and Attribution Calculations
# ---------------------------------------------------------------------------


def calculate_saliencies_results(
    input_ids_list,
    attention_masks_list,
    correct_ids_list,
    foil_ids_list,
    model,
    all_possible_foil_ids=None,
):
    """
    Calculates saliency and embedding results for both base and contrast cases.
    Returns dictionaries for mean and last step values as well as raw lists.
    """
    base_saliencies_list, base_embeddings_list = [], []
    contrast_saliencies_list, contrast_embeddings_list = [], []

    for input_ids, attention_mask, correct_id, foil_id in zip(
        input_ids_list, attention_masks_list, correct_ids_list, foil_ids_list
    ):
        base_saliency, base_embedding = saliency(
            model=model,
            input_ids=input_ids,
            input_mask=attention_mask,
            correct=correct_id,
            foil=None,
            all_possible_foil_ids=None,
        )
        contrast_saliency, contrast_embedding = saliency(
            model=model,
            input_ids=input_ids,
            input_mask=attention_mask,
            correct=correct_id,
            foil=foil_id,
            all_possible_foil_ids=all_possible_foil_ids,
        )
        base_saliencies_list.append(base_saliency)
        base_embeddings_list.append(base_embedding)
        contrast_saliencies_list.append(contrast_saliency)
        contrast_embeddings_list.append(contrast_embedding)

    # Compute mean and last step arrays.
    _, mean_base_saliencies, last_base = slice_array_and_calculate_mean(
        base_saliencies_list
    )
    _, mean_base_embeddings, last_base_emb = slice_array_and_calculate_mean(
        base_embeddings_list
    )
    _, mean_contrast_saliencies, last_contrast = slice_array_and_calculate_mean(
        contrast_saliencies_list
    )
    _, mean_contrast_embeddings, last_contrast_emb = slice_array_and_calculate_mean(
        contrast_embeddings_list
    )

    # Sanity-check: assume all arrays in the first step have the same length.
    length = len(base_saliencies_list[0])
    assert (
        length
        == len(base_embeddings_list[0])
        == len(contrast_saliencies_list[0])
        == len(contrast_embeddings_list[0])
    ), "Inconsistent lengths in saliency and embedding arrays."

    base_saliencies_dict = {"mean": mean_base_saliencies, "last_step": last_base}
    base_embeddings_dict = {"mean": mean_base_embeddings, "last_step": last_base_emb}
    contrast_saliencies_dict = {
        "mean": mean_contrast_saliencies,
        "last_step": last_contrast,
    }
    contrast_embeddings_dict = {
        "mean": mean_contrast_embeddings,
        "last_step": last_contrast_emb,
    }

    return (
        base_saliencies_dict,
        base_embeddings_dict,
        contrast_saliencies_dict,
        contrast_embeddings_dict,
        base_embeddings_list,
        base_saliencies_list,
        contrast_saliencies_list,
    )


def compute_input_gradient_scores(
    base_saliencies_dict,
    base_embeddings_dict,
    contrast_saliencies_dict,
    contrast_embeddings_dict,
):
    """Return normalized input × gradient attribution scores."""
    result = {}
    for strategy in ["mean", "last_step"]:
        result[strategy] = {
            "base": input_x_gradient(
                grads=base_saliencies_dict[strategy],
                embds=base_embeddings_dict[strategy],
                normalize=True,
            ),
            "contrast": input_x_gradient(
                grads=contrast_saliencies_dict[strategy],
                embds=contrast_embeddings_dict[strategy],
                normalize=True,
            ),
        }
    return result


def compute_gradient_norm_scores(base_saliencies_dict, contrast_saliencies_dict):
    """Return normalized L1 gradient norm scores for base and contrast saliencies."""
    result = {}
    for strategy in ["mean", "last_step"]:
        result[strategy] = {
            "base": l1_grad_norm(
                grads=base_saliencies_dict[strategy],
                normalize=True,
            ),
            "contrast": l1_grad_norm(
                grads=contrast_saliencies_dict[strategy],
                normalize=True,
            ),
        }
    return result


def compute_erasure_scores(
    input_ids_list, attention_masks_list, correct_id_list, foil_id_list, model
):
    """Compute erasure attribution scores for base and contrast prompts."""
    base_results, contrast_results = [], []
    for input_ids, attention_mask, correct_id, foil_id in zip(
        input_ids_list, attention_masks_list, correct_id_list, foil_id_list
    ):
        base_score = erasure_scores(
            model=model,
            input_ids=input_ids,
            input_mask=attention_mask,
            correct=correct_id,
            foil=None,
            remove=False,
            normalize=True,
        )
        contrast_score = erasure_scores(
            model=model,
            input_ids=input_ids,
            input_mask=attention_mask,
            correct=correct_id,
            foil=foil_id,
            remove=False,
            normalize=True,
        )
        base_results.append(base_score)
        contrast_results.append(contrast_score)

    _, mean_base, last_base = slice_array_and_calculate_mean(base_results)
    _, mean_contrast, last_contrast = slice_array_and_calculate_mean(contrast_results)

    result = {
        "mean": {"base": mean_base, "contrast": mean_contrast},
        "last_step": {"base": last_base, "contrast": last_contrast},
    }
    return result


def compute_exemplar_erasure_scores(
    input_ids_list,
    attention_masks_list,
    correct_id_list,
    foil_id_list,
    model,
    tokenizer,
):
    """Compute exemplar erasure scores for base and contrast prompts."""
    base_results, contrast_results = [], []
    for input_ids, attention_mask, correct_id, foil_id in zip(
        input_ids_list, attention_masks_list, correct_id_list, foil_id_list
    ):
        base_score = exemplar_based_erasure(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            input_mask=attention_mask,
            correct=correct_id,
            foil=None,
            remove=False,
            normalize=True,
        )
        contrast_score = exemplar_based_erasure(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            input_mask=attention_mask,
            correct=correct_id,
            foil=foil_id,
            remove=False,
            normalize=True,
        )
        base_results.append(base_score)
        contrast_results.append(contrast_score)

    _, mean_base, last_base = slice_array_and_calculate_mean(base_results)
    _, mean_contrast, last_contrast = slice_array_and_calculate_mean(contrast_results)

    result = {
        "mean": {"base": mean_base, "contrast": mean_contrast},
        "last_step": {"base": last_base, "contrast": last_contrast},
    }
    return result


def cosine_similarity(a, b):
    """
    Calculates the cosine similarity between two vectors.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def compute_embedding_cosine_similarity(
    base_embeddings_list,
    input_ids_list,
    tokenizer,
    correct_exemplar_index,
    method_metrics,
):
    """
    Calculates cosine similarity between the target question embedding and exemplar embeddings.
    Updates `method_metrics` if the predicted exemplar index matches `correct_exemplar_index`.
    """
    base_embeddings = base_embeddings_list[0]
    input_ids = input_ids_list[0][:-1]
    assert len(base_embeddings) == len(input_ids), (
        "Mismatch between base embeddings and input IDs length."
    )

    newline_token_id = tokenizer("\n")["input_ids"][-1]
    newline_indices = [
        i for i, token_id in enumerate(input_ids) if token_id == newline_token_id
    ]
    newline_indices.append(len(input_ids) - 1)

    # Determine token ranges for each exemplar.
    exemplars_ranges = []
    for i, idx in enumerate(newline_indices):
        if i == 0:
            exemplars_ranges.append((1, idx + 1))
        else:
            exemplars_ranges.append((newline_indices[i - 1] + 1, idx + 1))

    exemplars_embeddings = [base_embeddings[r[0] : r[1]] for r in exemplars_ranges]
    target_embedding = exemplars_embeddings[-1]
    cosine_similarities = [
        cosine_similarity(np.mean(target_embedding, axis=0), np.mean(e, axis=0))
        for e in exemplars_embeddings
    ]
    predicted_index = np.argmax(cosine_similarities[:-1])

    # Update metrics.
    if predicted_index == correct_exemplar_index:
        for strategy in ["mean", "last_step"]:
            for op in ["add", "multiply"]:
                for div in ["divide", "not_divide"]:
                    for base_or_contrast in ["base", "contrast"]:
                        method_metrics.__dict__[strategy][op][div][
                            base_or_contrast
                        ] += 1

    return predicted_index == correct_exemplar_index


def compute_gradient_cosine_similarity(
    base_saliencies_list,
    contrast_saliencies_list,
    input_ids_list,
    tokenizer,
    correct_exemplar_index,
    method_metrics,
):
    """
    Calculates cosine similarity using gradient saliency values.
    Updates `method_metrics` if the predicted exemplar index matches `correct_exemplar_index`.
    """
    base_saliencies = base_saliencies_list[0]
    contrast_saliencies = contrast_saliencies_list[0]
    input_ids = input_ids_list[0][:-1]
    assert len(base_saliencies) == len(input_ids) == len(contrast_saliencies), (
        "Mismatch in lengths for gradient cosine similarity."
    )

    for base_or_contrast in ["base", "contrast"]:
        saliencies = (
            base_saliencies if base_or_contrast == "base" else contrast_saliencies
        )
        newline_token_id = tokenizer("\n")["input_ids"][-1]
        newline_indices = [
            i for i, token_id in enumerate(input_ids) if token_id == newline_token_id
        ]
        newline_indices.append(len(input_ids) - 1)

        exemplars_ranges = []
        for i, idx in enumerate(newline_indices):
            if i == 0:
                exemplars_ranges.append((1, idx + 1))
            else:
                exemplars_ranges.append((newline_indices[i - 1] + 1, idx + 1))

        exemplars_saliencies = [saliencies[r[0] : r[1]] for r in exemplars_ranges]
        target_saliencies = exemplars_saliencies[-1]
        cosine_similarities = [
            cosine_similarity(np.mean(target_saliencies, axis=0), np.mean(e, axis=0))
            for e in exemplars_saliencies
        ]
        predicted_index = np.argmax(cosine_similarities[:-1])

        if predicted_index == correct_exemplar_index:
            for strategy in ["mean", "last_step"]:
                for op in ["add", "multiply"]:
                    for div in ["divide", "not_divide"]:
                        method_metrics.__dict__[strategy][op][div][
                            base_or_contrast
                        ] += 1

    return predicted_index == correct_exemplar_index


def dispatch_attribution_method(
    attribution_method,
    base_saliencies_dict,
    base_embeddings_dict,
    contrast_saliencies_dict,
    contrast_embeddings_dict,
    input_ids_list,
    attention_masks_list,
    correct_id_list,
    foil_id_list,
    model,
    tokenizer,
):
    """Dispatch attribution score computations based on the requested method."""
    if attribution_method == "input_x_gradient":
        return compute_input_gradient_scores(
            base_saliencies_dict,
            base_embeddings_dict,
            contrast_saliencies_dict,
            contrast_embeddings_dict,
        )
    elif attribution_method == "gradient_norm":
        return compute_gradient_norm_scores(
            base_saliencies_dict, contrast_saliencies_dict
        )
    elif attribution_method == "erasure":
        return compute_erasure_scores(
            input_ids_list, attention_masks_list, correct_id_list, foil_id_list, model
        )
    elif attribution_method == "exemplar_erasure":
        return compute_exemplar_erasure_scores(
            input_ids_list,
            attention_masks_list,
            correct_id_list,
            foil_id_list,
            model,
            tokenizer,
        )
    # For methods such as "embedding_cos_similarity" or "gradient_cos_similarity",
    # the attribution is computed directly by their corresponding functions.
    return None


def summarize_exemplar_erasure(
    method_result_dict, correct_exemplar_index, method_metrics, num_shots
):
    """Update metrics for exemplar erasure and report correctness."""
    max_exemplar_index_dict = {"correct_exemplar_index": correct_exemplar_index}
    for strategy, score_dict in method_result_dict.items():
        max_exemplar_index_dict[strategy] = {}
        is_base_correct = False
        is_contrast_correct = False

        for base_or_contrast, score in score_dict.items():
            assert len(score) == num_shots, (
                "Number of exemplars does not match num_shots."
            )
            predicted_index = np.argmax(score)
            max_exemplar_index_dict[strategy][base_or_contrast] = predicted_index
            if predicted_index == correct_exemplar_index:
                if strategy == "last_step":
                    if base_or_contrast == "base":
                        is_base_correct = True
                    elif base_or_contrast == "contrast":
                        is_contrast_correct = True
                for op in ["add", "multiply"]:
                    for div in ["divide", "not_divide"]:
                        method_metrics.__dict__[strategy][op][div][
                            base_or_contrast
                        ] += 1

    return is_base_correct and is_contrast_correct


def infer_exemplar_token_spans(correct_instance, tokenizer, method_result_dict=None):
    """Return exemplar token index spans and token counts within the prompt."""
    exemplars_text = correct_instance["exemplars"]
    if method_result_dict is not None:
        tokenized = tokenizer(exemplars_text + correct_instance["target_question"])[
            "input_ids"
        ]
        assert len(tokenized) == len(method_result_dict["mean"]["base"]), (
            "Tokenized exemplars and target question length mismatch."
        )
    exemplar_input_ids = tokenizer(exemplars_text)["input_ids"]
    newline_token_id = tokenizer("\n")["input_ids"][-1]
    assert exemplar_input_ids[-1] == newline_token_id, (
        "Exemplar input_ids do not end with newline token."
    )

    newline_indices = [
        i
        for i, token_id in enumerate(exemplar_input_ids)
        if token_id == newline_token_id
    ]
    exemplars_ranges = []
    for i, idx in enumerate(newline_indices):
        if i == 0:
            exemplars_ranges.append((1, idx + 1))
        else:
            exemplars_ranges.append((newline_indices[i - 1] + 1, idx + 1))
    tokens_length = [len(exemplar_input_ids[r[0] : r[1]]) for r in exemplars_ranges]

    return exemplars_ranges, tokens_length


def aggregate_scores_by_exemplar(
    score_array,
    exemplars_index_range_list,
    num_shots,
    input_ids_list,
    tokenizer,
    base_or_contrast,
):
    """Aggregate token-level attribution scores at the exemplar level."""
    added_scores, multiplied_scores = [], []
    tokens_per_exemplar, tokens_score_per_exemplar = [], []

    for r in exemplars_index_range_list:
        add_score = np.sum(score_array[r[0] : r[1]])
        multiply_score = np.prod(score_array[r[0] : r[1]])
        added_scores.append(add_score)
        multiplied_scores.append(multiply_score)

        if base_or_contrast == "contrast":
            tokens = tokenizer.convert_ids_to_tokens(input_ids_list[0][r[0] : r[1]])
            tokens_per_exemplar.append(tokens)
            tokens_score_per_exemplar.append(score_array[r[0] : r[1]])
        else:
            tokens_per_exemplar = None
            tokens_score_per_exemplar = None

    assert len(added_scores) == len(multiplied_scores), (
        "Mismatch in per-exemplar score lengths."
    )
    assert len(added_scores) == num_shots, (
        "Number of exemplars does not match num_shots."
    )
    norm = np.linalg.norm(multiplied_scores, ord=1)
    multiplied_scores = (multiplied_scores / norm).tolist()
    return (
        added_scores,
        multiplied_scores,
        tokens_per_exemplar,
        tokens_score_per_exemplar,
    )


def normalize_scores_by_length(score_list, exemplars_tokens_length_list):
    """Normalize exemplar scores by token counts and return both variants."""
    divided, not_divided = [], []
    for score, length in zip(score_list, exemplars_tokens_length_list):
        divided.append(score / length)
        not_divided.append(score)
    return divided, not_divided


def evaluate_exemplar_predictions(
    method_result_dict,
    exemplars_index_range_list,
    exemplars_tokens_length_list,
    correct_exemplar_index,
    method_metrics,
    num_shots,
    input_ids_list,
    tokenizer,
    method,
    save_path,
):
    """Aggregate exemplar-level scores and update metrics for prediction accuracy."""
    is_base_correct = False
    is_contrast_correct = False
    chosen_exemplar_index = None

    for strategy, score_dict in method_result_dict.items():
        for base_or_contrast, score in score_dict.items():
            (
                added_scores,
                multiplied_scores,
                tokens_per_exemplar,
                tokens_score_per_exemplar,
            ) = aggregate_scores_by_exemplar(
                score_array=score,
                exemplars_index_range_list=exemplars_index_range_list,
                num_shots=num_shots,
                input_ids_list=input_ids_list,
                tokenizer=tokenizer,
                base_or_contrast=base_or_contrast,
            )
            added_divided, added_not_divided = normalize_scores_by_length(
                added_scores, exemplars_tokens_length_list
            )
            added_max_index = np.argmax(added_divided)
            if added_max_index == correct_exemplar_index:
                method_metrics.__dict__[strategy]["add"]["divide"][
                    base_or_contrast
                ] += 1

            added_not_divided_max_index = np.argmax(added_not_divided)
            chosen_exemplar_index = added_not_divided_max_index
            if added_not_divided_max_index == correct_exemplar_index:
                method_metrics.__dict__[strategy]["add"]["not_divide"][
                    base_or_contrast
                ] += 1
                if base_or_contrast == "base":
                    is_base_correct = True
                elif base_or_contrast == "contrast":
                    is_contrast_correct = True

            multiplied_divided, multiplied_not_divided = normalize_scores_by_length(
                multiplied_scores, exemplars_tokens_length_list
            )
            multiplied_divided_max_index = np.argmax(multiplied_divided)
            if multiplied_divided_max_index == correct_exemplar_index:
                method_metrics.__dict__[strategy]["multiply"]["divide"][
                    base_or_contrast
                ] += 1

            multiplied_not_divided_max_index = np.argmax(multiplied_not_divided)
            if multiplied_not_divided_max_index == correct_exemplar_index:
                method_metrics.__dict__[strategy]["multiply"]["not_divide"][
                    base_or_contrast
                ] += 1

    return (
        True if is_base_correct and is_contrast_correct else False,
        tokens_per_exemplar,
        tokens_score_per_exemplar,
        chosen_exemplar_index,
    )


def encode_nested_dicts(record):
    """Convert nested dictionaries to JSON strings for serialization."""
    return {k: json.dumps(v) if isinstance(v, dict) else v for k, v in record.items()}


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def evaluate_attributions_for_instances(
    model_correct_task_list,
    rule,
    tokenizer,
    model,
    methods,
    method_metrics,
    num_shots,
    method_correct_tasks_dict,
    save_path,
    model_name,
):
    """Evaluate attribution scores for each instance and collect metrics."""
    gn_only_plot_counter = 0
    ixg_norm_larger_counter = 0
    norm_hypothesis_list = [
        ["Model", "Rule", "IxG larger count", "Total count", "Percentage"]
    ]
    raw_data_list = []
    used_signatures = set()

    for question_idx, correct_instance in tqdm(
        enumerate(model_correct_task_list),
        desc="Calculating attributions",
        total=len(model_correct_task_list),
    ):
        raw_attribution_score = {}
        # Build the attribution input text.
        input_text = (
            correct_instance["exemplars"] + correct_instance["target_question"] + " "
        )
        correct_text = correct_instance["model_answer"]
        foil_text = correct_instance["foil_answer"]

        signature = (
            rule
            + str(question_idx)
            + correct_instance["target_answer"]
            + foil_text
            + str(correct_instance["exemplars_index_for_diff"])
        )
        if signature in used_signatures:
            raise ValueError("Signature already used.")
        used_signatures.add(signature)
        raw_attribution_score["signature"] = signature
        raw_attribution_score["eureka_idx"] = correct_instance[
            "exemplars_index_for_diff"
        ]

        assert correct_instance["target_answer"] == correct_instance["model_answer"], (
            "The correct answer is not the model's answer."
        )

        all_possible_foil_ids = None
        if rule == "associative_recall":
            all_possible_foil_numbers = [i for i in range(10) if i != int(correct_text)]
            all_possible_foil_ids = [
                tokenizer(str(i))["input_ids"][-1] for i in all_possible_foil_numbers
            ]

        # Create token lists for attribution.
        input_ids_list, attention_masks_list, correct_id_list, foil_id_list = (
            create_ids_list(tokenizer, input_text, correct_text, foil_text)
        )
        assert (all_possible_foil_ids is None) or (rule == "associative_recall"), (
            "Invalid foil IDs for rule."
        )

        (
            base_saliencies_dict,
            base_embeddings_dict,
            contrast_saliencies_dict,
            contrast_embeddings_dict,
            base_embeddings_list,
            base_saliencies_list,
            contrast_saliencies_list,
        ) = calculate_saliencies_results(
            input_ids_list,
            attention_masks_list,
            correct_id_list,
            foil_id_list,
            model,
            all_possible_foil_ids,
        )

        # Initialize flags and chosen exemplar indices for later comparison.
        gn_correct_flag = False
        ixg_correct_flag = False
        gn_chosen_index = None
        ixg_chosen_index = None

        # Process each attribution method.
        for method in methods:
            raw_attribution_score[method] = {}
            method_result_dict = dispatch_attribution_method(
                attribution_method=method,
                base_saliencies_dict=base_saliencies_dict,
                base_embeddings_dict=base_embeddings_dict,
                contrast_saliencies_dict=contrast_saliencies_dict,
                contrast_embeddings_dict=contrast_embeddings_dict,
                input_ids_list=input_ids_list,
                attention_masks_list=attention_masks_list,
                correct_id_list=correct_id_list,
                foil_id_list=foil_id_list,
                model=model,
                tokenizer=tokenizer,
            )
            correct_exemplar_index = correct_instance["exemplars_index_for_diff"]

            if method == "exemplar_erasure":
                is_method_correct = summarize_exemplar_erasure(
                    method_result_dict,
                    correct_exemplar_index,
                    method_metrics.methods[method],
                    num_shots,
                )
                ie_scores_list = method_result_dict["mean"]["contrast"].tolist()
                for i, scores in enumerate(ie_scores_list):
                    raw_attribution_score[method][f"sentence_{i}"] = {"scores": scores}
                if is_method_correct:
                    method_correct_tasks_dict[method].append(correct_instance)

            elif method == "embedding_cos_similarity":
                is_method_correct = compute_embedding_cosine_similarity(
                    base_embeddings_list,
                    input_ids_list,
                    tokenizer,
                    correct_exemplar_index,
                    method_metrics.methods[method],
                )
                if is_method_correct:
                    method_correct_tasks_dict[method].append(correct_instance)

            elif method == "gradient_cos_similarity":
                is_method_correct = compute_gradient_cosine_similarity(
                    base_saliencies_list,
                    contrast_saliencies_list,
                    input_ids_list,
                    tokenizer,
                    correct_exemplar_index,
                    method_metrics.methods[method],
                )
                if is_method_correct:
                    method_correct_tasks_dict[method].append(correct_instance)

            else:
                # For methods like "gradient_norm" or "input_x_gradient", calculate per-exemplar scores.
                exemplars_index_range_list, exemplars_tokens_length_list = (
                    infer_exemplar_token_spans(
                        correct_instance, tokenizer, method_result_dict
                    )
                )
                (
                    is_method_correct,
                    tokens_per_exemplar,
                    tokens_score_per_exemplar,
                    chosen_exemplar_index,
                ) = evaluate_exemplar_predictions(
                    method_result_dict,
                    exemplars_index_range_list,
                    exemplars_tokens_length_list,
                    correct_exemplar_index,
                    method_metrics.methods[method],
                    num_shots,
                    input_ids_list,
                    tokenizer,
                    method,
                    save_path,
                )
                if is_method_correct:
                    method_correct_tasks_dict[method].append(correct_instance)

                if (
                    tokens_score_per_exemplar is not None
                    and tokens_per_exemplar is not None
                ):
                    for i, (tokens_list, scores_list) in enumerate(
                        zip(tokens_per_exemplar, tokens_score_per_exemplar)
                    ):
                        raw_attribution_score[method][f"sentence_{i}"] = {
                            "tokens": tokens_list,
                            "scores": scores_list.astype(float).tolist(),
                        }
                else:
                    # If tokens_score_per_exemplar or tokens_per_exemplar is None,
                    # we might want to log this or handle it,
                    # for now, we just ensure the method key exists in raw_attribution_score
                    if method not in raw_attribution_score:
                        raw_attribution_score[method] = {}

                if method == "gradient_norm":
                    gn_chosen_index = chosen_exemplar_index
                    if gn_chosen_index == correct_exemplar_index:
                        gn_correct_flag = True
                if method == "input_x_gradient":
                    ixg_chosen_index = chosen_exemplar_index
                    if ixg_chosen_index == correct_exemplar_index:
                        ixg_correct_flag = True

                # In cases where gradient_norm is correct but input_x_gradient is not,
                # perform an additional norm comparison.
                if gn_correct_flag and not ixg_correct_flag:
                    if gn_chosen_index is None or ixg_chosen_index is None:
                        # This continue will skip the rest of the loop for the current method
                        # but the raw_attribution_score for the method would have been populated above
                        continue
                    assert gn_chosen_index == correct_exemplar_index
                    assert ixg_chosen_index != correct_exemplar_index
                    gn_only_plot_counter += 1

                    token_norms = calculate_embedding_norm(model, tokenizer, input_text)
                    exemplars_norms = [
                        token_norms[0][r[0] : r[1]] for r in exemplars_index_range_list
                    ]
                    avg_norm = [torch.mean(n) for n in exemplars_norms]
                    ixg_avg_norm = avg_norm[ixg_chosen_index]
                    correct_avg_norm = avg_norm[correct_exemplar_index]
                    if correct_avg_norm < ixg_avg_norm:
                        ixg_norm_larger_counter += 1

        raw_data_list.append(raw_attribution_score)

    if gn_only_plot_counter > 0:
        norm_hypothesis_list.append(
            [
                model_name,
                rule,
                ixg_norm_larger_counter,
                gn_only_plot_counter,
                ixg_norm_larger_counter / gn_only_plot_counter,
            ]
        )
        import csv

        csv_path = save_path.parents[2] / "norm_hypothesis.csv"
        mode = "w" if not csv_path.exists() else "a"
        with open(csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerows(norm_hypothesis_list)
            else:
                writer.writerows(norm_hypothesis_list[1:])

    records = [encode_nested_dicts(record) for record in raw_data_list]
    df = pl.DataFrame(records)

    return method_correct_tasks_dict, df
