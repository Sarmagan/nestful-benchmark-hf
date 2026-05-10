import json
import re


def ground_seq_nested_repsonse(api_list):
    def check_label_in_slot(label, slot_v):
        if slot_v.startswith("$var"):
            if "." in slot_v:
                lbl_slot = slot_v.split(".", 1)[0].replace("$", "")
                if lbl_slot == label:
                    return True
        return False

    label_api_map = {}
    for api in api_list:
        if api["name"] == "varResult":
            continue
        if api["name"] == "var_result":
            continue
        if "label" in api:
            lbl = api["label"].replace("$", "")
            label_api_map[lbl] = api["name"]

    grounded_api_list = []

    for api in api_list:
        if api["name"] == "var_result":
            continue
        temp_arguments = {}
        if "arguments" in api:
            arg_dict = api["arguments"]
        elif "parameters" in api:
            arg_dict = api["parameters"]
        else:
            arg_dict = {}
        for s_n, s_v in arg_dict.items():
            for l, a in label_api_map.items():
                if type(s_v) == str and check_label_in_slot(l, s_v):
                    s_v = s_v.replace(l, a)
                elif type(s_v) == list:
                    new_s_v = []
                    for v in s_v:
                        if type(v) == str and check_label_in_slot(l, v):
                            v = v.replace(l, a)
                        elif type(v) == dict and check_label_in_slot(l, json.dumps(v)):
                            v = json.loads(json.dumps(v).replace(l, a))
                        new_s_v.append(v)
                    s_v = new_s_v
            temp_arguments[s_n] = s_v

        grounded_api_list.append(
            {
                "name": api["name"],
                "arguments": temp_arguments,
            }
        )
    return grounded_api_list


def parse_agent_output(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []

    gold_dict_list = json.loads(item["output"])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = [json.dumps(func) for func in ground_seq_nested_repsonse(gold_dict_list)]

    def normalize_calls(call_list):
        normalized_calls = []
        for idx, func in enumerate(call_list, start=1):
            if not isinstance(func, dict):
                continue
            name = func.get("name")
            if not name:
                continue
            arguments = func.get("arguments", func.get("parameters", {}))
            if not isinstance(arguments, dict):
                arguments = {}
            label = func.get("label", f"$var_{idx}")
            normalized_calls.append(
                {
                    "name": name,
                    "label": label,
                    "arguments": arguments,
                }
            )
        return normalized_calls

    def parse_first_json_list(text):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                parsed = parsed["tool_calls"]
            elif isinstance(parsed, dict) and "name" in parsed:
                # Recover single-call outputs emitted as one JSON object.
                parsed = [parsed]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return None

    def parse_json_objects_sequence(text):
        """
        Recover object-per-line outputs such as:
        { ... }
        { ... }
        """
        decoder = json.JSONDecoder()
        i = 0
        recovered = []
        while i < len(text):
            while i < len(text) and text[i] not in "{[":
                i += 1
            if i >= len(text):
                break

            try:
                obj, consumed = decoder.raw_decode(text[i:])
            except Exception:
                i += 1
                continue

            i += consumed
            if isinstance(obj, dict):
                if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                    recovered.extend(obj["tool_calls"])
                elif "name" in obj:
                    recovered.append(obj)
            elif isinstance(obj, list):
                for entry in obj:
                    if isinstance(entry, dict) and "name" in entry:
                        recovered.append(entry)
        return recovered if recovered else None

    def parse_action_blocks(text):
        action_dicts = []
        # Capture "Action:" blocks and parse the first JSON object after each.
        blocks = re.findall(
            r"Action:\s*([\s\S]*?)(?=\n(?:Thought:|Observation:|Final Answer:|Action:)|\Z)",
            text,
        )
        for b in blocks:
            start = b.find("{")
            if start == -1:
                continue
            depth = 0
            end = None
            for i, ch in enumerate(b[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end is None:
                continue
            candidate = b[start : end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                        action_dicts.extend(obj["tool_calls"])
                    else:
                        action_dicts.append(obj)
            except Exception:
                continue
        return action_dicts

    try:
        generated_text = (
            item["generated_text"]
            .replace("```json", "")
            .replace("```", "")
            .replace("<|tool_call|>", "")
            .replace("<tool_call>", "")
            .replace("</tool_call>", "")
            .replace("<think>", "")
            .replace("</think>", "")
            .strip()
        )

        parsed = parse_first_json_list(generated_text)
        if parsed is None:
            parsed = parse_action_blocks(generated_text)
        if not parsed:
            parsed = parse_json_objects_sequence(generated_text)
        if not parsed:
            raise ValueError("Could not parse tool calls from agent output")

        pred_dict_list = normalize_calls(parsed)
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = [json.dumps(func) for func in ground_seq_nested_repsonse(pred_dict_list)]
    except Exception:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )
