import argparse
import copy
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

from scorer import calculate_scores, print_result
from utils import read_json, read_jsonlines, write_jsonlines

MODEL_ID = "Qwen/Qwen3-8B"
hf_logging.set_verbosity_error()

COMMON_FORMAT_RULES = (
    "Return only a JSON list of function calls. "
    "Each item must be {\"name\": str, \"label\": \"$var_n\", \"arguments\": {...}}. "
    "Even if there is only one call, still return a list with one object: [{...}]. "
    "Use labels like $var_1, $var_2, ... and references like \"$var_1.result$\" for nested steps. "
    "Do not include any explanation."
)

COT_JSON_RULES = (
    "After your reasoning, output exactly one JSON list of function calls and nothing after it. "
    "Each item must be {\"name\": str, \"label\": \"$var_n\", \"arguments\": {...}}. "
    "Even if there is only one call, still return a list with one object: [{...}]. "
    "Use labels like $var_1, $var_2, ... and references like \"$var_1.result$\" for nested steps. "
    "Do not wrap the JSON in markdown fences."
)

SYSTEM_PROMPT_BY_BASELINE = {
    "vanilla": (
        "You are a tool-calling assistant. "
        + COMMON_FORMAT_RULES
    ),
    "icl": (
        "You are a tool-calling assistant. Use the provided examples while planning. "
        + COMMON_FORMAT_RULES
    ),
    "cot": (
        "You are a tool-calling assistant. "
        "First, think through the problem step by step in plain text (brief steps). "
        "Then output your final answer. "
        + COT_JSON_RULES
    ),
}


def build_icl_string(icl_examples: List[Dict], icl_count: int) -> str:
    rows = []
    for idx, ex in enumerate(icl_examples[:icl_count], start=1):
        rows.append(
            f"# Example-{idx}\n"
            f"Question: {ex['input']}\n"
            f"Output: {json.dumps(ex['output'])}"
        )
    return "\n\n".join(rows)


def build_messages(sample: Dict, baseline: str, icl_str: str) -> List[Dict]:
    system_prompt = (
        f"{SYSTEM_PROMPT_BY_BASELINE[baseline]}\n\n"
        f"Available tools:\n{json.dumps(sample['tools'])}\n\n"
    )
    if icl_str:
        system_prompt += f"Examples:\n{icl_str}\n\n"

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": sample["input"]},
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--baseline", type=str, default="vanilla", choices=["vanilla", "icl", "cot"])
    parser.add_argument("--dataset", type=str, default="data_v2/nestful_data.jsonl")
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--save_directory", type=str, default="results")
    parser.add_argument("--executable_func_dir", type=str, default="data_v2/executable_functions")
    parser.add_argument("--icl_count", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--disable_8bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    use_8bit = not args.disable_8bit
    baseline = args.baseline
    run_name = args.run_name or f"qwen3-8b-8bit-{baseline}"

    data = read_jsonlines(args.dataset)
    if args.max_examples > 0:
        data = data[: args.max_examples]
    icl_str = ""
    effective_icl_count = 0
    if baseline == "icl":
        icl_examples = read_json("src/icl_examples.json")
        icl_str = build_icl_string(icl_examples, args.icl_count)
        effective_icl_count = args.icl_count

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        **model_kwargs,
    )
    model.eval()

    outputs = []
    for sample in tqdm(data):
        messages = build_messages(sample, baseline, icl_str)
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        model_inputs = {"input_ids": input_ids.to(model.device)}
        if tokenizer.pad_token_id is not None:
            model_inputs["attention_mask"] = (model_inputs["input_ids"] != tokenizer.pad_token_id).long()
        do_sample = args.temperature > 0

        with torch.no_grad():
            generation_config = copy.deepcopy(model.generation_config)
            generation_config.do_sample = do_sample
            if do_sample:
                generation_config.temperature = args.temperature
                generation_config.top_p = args.top_p
            else:
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None

            generate_kwargs = dict(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = model.generate(**generate_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()

        outputs.append(
            {
                "sample_id": sample["sample_id"],
                "input": sample["input"],
                "output": json.dumps(sample["output"]),
                "gold_answer": json.dumps(sample["gold_answer"]),
                "tools": json.dumps(sample["tools"]),
                "generated_text": generated_text,
            }
        )

    result_dir = os.path.join(args.save_directory, f"nestful_{effective_icl_count}", run_name)
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, "output.jsonl")
    write_jsonlines(outputs, output_path)
    print(f"Saved generations to: {output_path}")

    score_model_name = "cot-agent" if baseline == "cot" else f"{baseline}-agent"
    result = calculate_scores(outputs, score_model_name, args.executable_func_dir)
    print_result(result, score_model_name)


if __name__ == "__main__":
    main()
