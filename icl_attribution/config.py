from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class EvaluationConfig:
    """Runtime configuration for an evaluation run.

    Attributes mirror the arguments exposed by the public CLI. Paths resolve to
    the current working directory unless absolute values are supplied.
    """

    exp_date: str
    exp_name: str
    rule: str
    model_name: str
    num_shots: int
    toy: bool = False
    model_accuracy: bool = False
    method_accuracy: bool = True
    no_disambiguation: bool = False
    exemplar_reorder: bool = False
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    run_name: Optional[str] = None

    def resolved_output_dir(self) -> Path:
        """Return the output directory as an absolute path and ensure it exists."""

        resolved = self.output_dir.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @property
    def run_identifier(self) -> str:
        """Unique identifier for this run, defaulting to the experiment name."""

        return self.run_name or self.exp_name
