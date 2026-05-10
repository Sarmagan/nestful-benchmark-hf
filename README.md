# Local NESTFUL Runs

README for running local Hugging Face models on NESTFUL benchmark and tracking baseline results.  
Model is fixed to `Qwen/Qwen3-8B` in `src/run_hf_local.py`.

Relevant benchmark references:

- Paper: **NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls**
  - [https://arxiv.org/abs/2409.03797v3](https://arxiv.org/abs/2409.03797v3)
- Dataset on Hugging Face:
  - [https://huggingface.co/datasets/ibm-research/nestful](https://huggingface.co/datasets/ibm-research/nestful)

## 1) Benchmark Data Notes

- Main evaluation data is under `data_v2`.
- `data_v2/nestful_data.jsonl` has 1861 evaluation examples.
- `data_v2/executable_functions` contains executable tool implementations used by scorer win-rate evaluation.

## 2) Environment

```bash
conda activate armagan
cd /s0/armagan/projects/nestful
```

## 3) Main Runner (Generate + Score)

`src/run_hf_local.py` does both:

1. Generate model outputs
2. Save `output.jsonl`
3. Run NESTFUL scoring and print final metrics

Prompting in `src/run_hf_local.py` is self-contained (your own prompt text).
For the ICL baseline, examples are read from `src/icl_examples.json`.

Default behavior: first `100` examples from `data_v2/nestful_data.jsonl` for faster iteration.
Use `--max_examples -1` for full benchmark.

### Vanilla Baseline (No ICL)

```bash
python -u src/run_hf_local.py \
  --run_name "qwen3-8b-8bit-vanilla" \
  --baseline "vanilla" \
  --dataset "data_v2/nestful_data.jsonl" \
  --executable_func_dir "data_v2/executable_functions" \
  --save_directory "results" \
  --temperature 0.0 \
  --max_new_tokens 1024
```

### ICL Baseline (Separate from Vanilla)

```bash
python -u src/run_hf_local.py \
  --run_name "qwen3-8b-8bit-icl3" \
  --baseline "icl" \
  --dataset "data_v2/nestful_data.jsonl" \
  --executable_func_dir "data_v2/executable_functions" \
  --save_directory "results" \
  --icl_count 3 \
  --temperature 0.0 \
  --max_new_tokens 1024
```

### CoT Baseline (chain-of-thought, no ICL)

The model may write brief reasoning first, then a single JSON list of tool calls (same parsing as other local baselines: whole-text JSON or a `[...]` substring, then `Action:` blocks if needed).

```bash
python -u src/run_hf_local.py \
  --run_name "qwen3-8b-8bit-cot" \
  --baseline "cot" \
  --dataset "data_v2/nestful_data.jsonl" \
  --executable_func_dir "data_v2/executable_functions" \
  --save_directory "results" \
  --temperature 0.0 \
  --max_new_tokens 1024
```

### ReAct Baseline (reason + action blocks, no ICL)

Implements ReAct-style `Thought / Action / Observation` prompting.  
For strict benchmark parsing, generated `Action:` JSON objects are normalized into one JSON list before scoring.

```bash
python -u src/run_hf_local.py \
  --run_name "qwen3-8b-8bit-react" \
  --baseline "react" \
  --dataset "data_v2/nestful_data.jsonl" \
  --executable_func_dir "data_v2/executable_functions" \
  --save_directory "results" \
  --temperature 0.0 \
  --max_new_tokens 1024
```

## 4) Optional Variants

### Disable 8-bit Quantization

```bash
python -u src/run_hf_local.py \
  --run_name "qwen3-8b-fp" \
  --baseline "vanilla" \
  --disable_8bit
```

### Full Benchmark

```bash
python -u src/run_hf_local.py \
  --run_name "qwen3-8b-8bit-vanilla-full" \
  --baseline "vanilla" \
  --max_examples -1
```

## 5) Output Locations

- Vanilla:
  - `results/nestful_0/qwen3-8b-8bit-vanilla/output.jsonl`
- CoT:
  - `results/nestful_0/qwen3-8b-8bit-cot/output.jsonl` (example path; matches your `--run_name`)
- ReAct:
  - `results/nestful_0/qwen3-8b-8bit-react/output.jsonl` (example path; matches your `--run_name`)
- ICL:
  - `results/nestful_3/qwen3-8b-8bit-icl3/output.jsonl`

Printed metrics:

- `Parsing Errors`
- `F1 Intent`
- `F1 Slot`
- `Partial Match Accuracy`
- `Full Match Accuracy`
- `Win Rate`

Metric definitions:

- `F1 Intent`: F1 score for tool (API) generation quality.
- `F1 Slot`: F1 score for parameter (argument-value pair) generation quality.
- `Partial Match Accuracy`: Partial Sequence Matching, measuring how many predicted APIs with their argument-value pairs match the gold API sequence.
- `Full Match Accuracy`: Full Sequence Matching, evaluating whether the model produces the exact full gold API sequence.
- `Win Rate`: Whether the predicted APIs, when executed, produce the gold answer.
- `Parsing Errors`: Number of samples where model output could not be parsed into valid tool-call format.

## 6) Results Table

100 samples


| Baseline         | Parsing Errors | F1 Intent | F1 Slot | Partial Acc | Full Acc | Win Rate |
| ---------------- | -------------- | --------- | ------- | ----------- | -------- | -------- |
| Vanilla          | 9              | 0.923     | 0.404   | 0.181       | 0.050    | 0.340    |
| CoT              | 12             | 0.784     | 0.381   | 0.173       | 0.040    | 0.450    |
| ICL (3 examples) | 2              | 0.919     | 0.517   | 0.276       | 0.180    | 0.380    |


