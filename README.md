# [Input Attribution for In-Context Learning (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.1092/)

By [Mengyu Ye](https://muyo8692.com), [Tatsuki Kuribayashi](https://kuribayashi4.github.io/),
[Goro Kobayashi](https://sites.google.com/view/goro-kobayashi/), and [Jun Suzuki](https://www.fai.cds.tohoku.ac.jp/members/js/)

[![arXiv](https://img.shields.io/badge/arXiv-2412.15628-red.svg)](https://arxiv.org/abs/2412.15628)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the code for our ACL 2025 Findings paper, "Can Input Attributions Explain Inductive Reasoning in In-Context Learning?". It provides a lightweight evaluation pipeline, curated datasets, and attribution analyses for all experiments reported in the paper.

## Setup

### Environment

We use [uv](https://github.com/astral-sh/uv) as our project management tool. To set up the environment, run the following commands:

```bash
git clone https://github.com/muyo8692/input-attribution-icl.git
cd input-attribution-icl
uv sync
```

### API Keys

Refer to `.env_template` to set up your OpenAI API key, which is required for automatic evaluation.

## Datasets

This repository contains all datasets required to reproduce the experiments from the paper. The data is located in the `tasks/` directory and organized by task family:

- `linear_or_distinct`
- `add_or_multiply`
- `associative_recall`
- `verb_object`
- `tense_article`
- `pos_title`

Within each task directory, you will find subdirectories for `10_shots/`, `50_shots/`, and `100_shots/` evaluations. Each JSON file in these directories contains a list of task instances under the `tasks` key.

## Running Evaluations

To run a quick smoke test and verify your setup, execute the evaluation script from the `scripts/` directory:

```bash
bash scripts/run_evaluation.sh
```

This will run a small-scale experiment using the `--toy` flag. To run a full evaluation, you can modify the command inside `scripts/run_evaluation.sh` with your desired arguments (e.g., model, task rule, number of shots) and remove the `--toy` flag.

The key arguments for the CLI are:

- `--model-accuracy`: Compute model accuracy for each task. This is required before running attribution experiments.
- `--method-accuracy`: Run attribution experiments.
- `--toy`: Restrict attribution to a 50-instance subset for quick tests.
- `--output-dir`: Customize the directory where results are saved (defaults to `./outputs`).

## Citation

```
@inproceedings{ye-etal-2025-input,
    title = "Can Input Attributions Explain Inductive Reasoning in In-Context Learning?",
    author = "Ye, Mengyu and Kuribayashi, Tatsuki and Kobayashi, Goro and Suzuki, Jun",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    year = "2025",
    url = "https://aclanthology.org/2025.findings-acl.1092/",
}
```
