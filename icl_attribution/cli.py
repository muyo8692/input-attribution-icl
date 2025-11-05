"""
Command-line interface for running ICL attribution experiments.

This script provides a CLI to configure and run attribution evaluations
based on the settings defined in `config.py` and `workflows.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fire
from dotenv import load_dotenv

from .config import EvaluationConfig
from .workflows import run_evaluation


def main(
    exp_date: str,
    exp_name: str,
    rule: str,
    model_name: str,
    num_shots: int,
    toy: bool = False,
    model_accuracy: bool = False,
    method_accuracy: bool = False,
    no_disambiguation: bool = False,
    exemplar_reorder: bool = False,
    output_dir: str = "./outputs",
    run_name: Optional[str] = None,
) -> None:
    """
    Main entry point for the ICL attribution CLI.

    Parses command-line arguments, creates an EvaluationConfig, and runs
    the evaluation workflow.
    """
    load_dotenv()

    if not (model_accuracy or method_accuracy):
        # Preserve previous behaviour: attribution run by default.
        method_accuracy = True

    config = EvaluationConfig(
        exp_date=exp_date,
        exp_name=exp_name,
        rule=rule,
        model_name=model_name,
        num_shots=num_shots,
        toy=toy,
        model_accuracy=model_accuracy,
        method_accuracy=method_accuracy,
        no_disambiguation=no_disambiguation,
        exemplar_reorder=exemplar_reorder,
        output_dir=Path(output_dir),
        run_name=run_name,
    )

    run_evaluation(config)


if __name__ == "__main__":
    fire.Fire(main)
