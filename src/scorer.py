import os, json, argparse, statistics, sys, signal, importlib
from tqdm import tqdm
from utils import *
from output_parsers import *
from sklearn.metrics import accuracy_score

def handler(signum, frame): 
    raise TimeoutError("Time limit exceeded!") 

# Local Hugging Face Qwen runs (run_hf_local.py).
QWEN_LOCAL_MODEL_NAMES = (
    "vanilla-agent",
    "icl-agent",
    "cot-agent",
)

def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

def get_basic_func_list(basic_func_file_path):
    lines = read_file(basic_func_file_path)
    func_names = []
    for l in lines:
        if l.startswith("def "):
            func_str = l.strip().replace("def ","")
            func_name = func_str[:func_str.index('(')]
            func_names.append(func_name)
    return func_names

def calculate_ans(func_calls, spec_lib, executable_func_dir):
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(10)

    basic_func_file_name = "basic_functions.py"
    basic_func_file_path = os.path.join(executable_func_dir, basic_func_file_name)
    basic_func_list = get_basic_func_list(basic_func_file_path)
    func_file_dict = read_json(os.path.join(executable_func_dir, "func_file_map.json"))

    try:
        variable_result_map = {}
        for idx, f in enumerate(func_calls):
            label = f["label"].replace("$", "")
            output_params = [s for s in spec_lib if s["name"] == f["name"]][0]["output_parameters"]
            output_params = list(output_params.keys())
            arg_values = []
            arg_val_list = []
            for k, v in f["arguments"].items():
                if  type(v) == str and v.startswith("$") and v.endswith("$"):
                    v = v[1:-1]
                    v_l = v.split(".",1)[0]
                    out_param = v.split(".",1)[1]
                    if v_l not in variable_result_map:
                        return False
                    # Many model outputs use ".result" regardless of the tool's actual
                    # output key (e.g. "output_0"). Fall back to the sole output value.
                    if out_param in variable_result_map[v_l]:
                        v = variable_result_map[v_l][out_param]
                    elif out_param == "result" and len(variable_result_map[v_l]) == 1:
                        v = next(iter(variable_result_map[v_l].values()))
                    else:
                        return False
                elif  type(v) == str and v.startswith("$var"):
                    v = v[1:]
                    v_l = v.split(".",1)[0]
                    out_param = v.split(".",1)[1]
                    if v_l not in variable_result_map:
                        return False
                    if out_param in variable_result_map[v_l]:
                        v = variable_result_map[v_l][out_param]
                    elif out_param == "result" and len(variable_result_map[v_l]) == 1:
                        v = next(iter(variable_result_map[v_l].values()))
                    else:
                        return False
                arg_val_list.append(v)
                arg_values.append(str(v))

            func_str = f"{f['name']}({','.join(arg_values)})"
            if f["name"] in basic_func_list:
                file_name = basic_func_file_name
            else:
                file_name = func_file_dict[f["name"]]
            file_path = os.path.join(executable_func_dir, file_name)

            spec = importlib.util.spec_from_file_location(file_name, file_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[file_name] = mod
            spec.loader.exec_module(mod)
            func = getattr(mod, f['name'])
            try:
                res = func(*arg_val_list)
            except:
                return False

            if len(output_params) == 1:  
                variable_result_map[label] = {
                    output_params[0]: res
                }
            else:
                return False

        final_var = func_calls[-1]["label"].replace("$", "")
        final_ans = next(iter(variable_result_map[final_var].values()))
        return final_ans
    
    except TimeoutError: 
        print("The program timed out!") 
        signal.alarm(0)
        return False
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        return False
    
def calculate_win_score(pred_func_calls, gold_ans, tools, executable_func_dir):
    if not pred_func_calls:
        return False
    
    gold_ans = json.loads(gold_ans)
    tools = json.loads(tools)
    pred_ans = calculate_ans(pred_func_calls, tools, executable_func_dir)
   
    if type(gold_ans) == float and type(pred_ans) == float:
        dec_no = len(str(gold_ans).split(".")[1])
        pred_ans = round(pred_ans, dec_no)
    if pred_ans == gold_ans:
        return True
    else:
        if type(pred_ans) == tuple and type(gold_ans) == list:
            pred_ans = listit(pred_ans)
            if pred_ans == gold_ans:
                return True
        return False


def get_alter_traj_scores(win_rate, acc_comb):
    print(len(win_rate))
    print(len(acc_comb))
    alt_traj_lst = []
    for wr, ac in zip(win_rate, acc_comb):
        if wr and ac < 1:
            alt_traj_lst.append(True)
        else:
            alt_traj_lst.append(False)
    sc = sum(alt_traj_lst) / len(alt_traj_lst)
    print(sc)
    return sc

def calculate_scores(predictions, model_name, executable_func_dir, intents_only=False, win_rate_flag=True, alt_traj_flag = True):
    gold_output_intent = []
    pred_output_intent = []
    gold_output_slot = []
    pred_output_slot = []
    p_intent, r_intent, f1_intent, p_slot, r_slot, f1_slot = None, None, None, None, None, None
    num_errors_parsing_pred_intent = 0
    num_errors_parsing_gold_intent = 0
    num_errors_parsing_pred_slot = 0
    num_errors_parsing_gold_slot = 0
    all_accuracy_combined = []
    all_num_times_full_score = 0
    win_rate_list = []

    num_pred_examples_w_parsing_errors = 0
    for item in tqdm(predictions):
        pred_has_parsing_errors = False
        pred_func_calls, gold_func_calls = [], []
        if model_name in QWEN_LOCAL_MODEL_NAMES:
            pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list, num_errors_parsing_pred_intent, pred_has_parsing_errors = parse_agent_output(
                item, num_errors_parsing_pred_intent
            )
        else:
            raise Exception(f"model not handled: {model_name!r} (expected one of {list(QWEN_LOCAL_MODEL_NAMES)})")

        gold_apis_names, pred_apis_names = [], []
        for f in pred_func_calls:
            if not f: continue
            try:
                if f.strip() == '{"name": "dummy", "arguments": {}}':
                    continue
                f = json.loads(f.replace('<|endoftext|>', '').strip())
                pred_apis_names.append(str(f['name']))
            except:
                num_errors_parsing_pred_intent += 1
                pred_has_parsing_errors = True
                pass
        for f in gold_func_calls:
            if not f: continue
            try:
                f = json.loads(f.replace('<|endoftext|>', '').strip())
                gold_apis_names.append(str(f['name']))
            except: 
                num_errors_parsing_gold_intent += 1
                pass
            
        gold_output_intent.append(gold_apis_names)
        pred_output_intent.append(pred_apis_names)
        if not intents_only:
            pred_api_map, gold_api_map = {}, {}
            for f in pred_func_calls:
                if f.strip() == '{"name": "dummy", "arguments": {}}':
                    continue
                try:
                    if not f: continue
                    f = json.loads(f.replace('<|endoftext|>', '').strip())
                    if type(f) != dict or 'name' not in f:
                        raise Exception
                    api_name = f['name']
                    pred_api_map[api_name] = []
                    for arg, val in f['arguments'].items():
                        if type(val) == str and val.startswith("$") and not val.endswith("$") :
                            val = val + "$"
                        pred_api_map[f['name']].append(f'{arg} = {val}')
                except:
                    num_errors_parsing_pred_slot += 1
                    pred_has_parsing_errors = True
                    pass
            for f in gold_func_calls:
                if not f: continue
                try:
                    f = json.loads(f.replace('<|endoftext|>', '').strip())
                    gold_api_map[f['name']] = []
                    for arg, val in f['arguments'].items():
                        gold_api_map[f['name']].append(f'{arg} = {val}')
                except: 
                    num_errors_parsing_gold_slot += 1
                    pass
            for key in set(pred_api_map.keys()).union(gold_api_map.keys()):
                if key in pred_api_map:
                    pred_output_slot.append(pred_api_map[key])
                else:
                    pred_output_slot.append([])
                if key in gold_api_map:
                    gold_output_slot.append(gold_api_map[key])
                else:
                    gold_output_slot.append([])

        if pred_has_parsing_errors:
            num_pred_examples_w_parsing_errors += 1

        api_with_args_gold = []
        for f in gold_func_calls:
            f = json.loads(f.replace('<|endoftext|>', '').strip())
            f_name = str(f['name'])
            args = ", ".join(sorted([f"{key} = {val}"  for key, val in f['arguments'].items()]))
            api_with_args_gold.append(f'{f_name}({args})')

        api_with_args_pred = []
        for f in pred_func_calls:
            try:
                f = json.loads(f.replace('<|endoftext|>', '').strip())
                f_name = str(f['name'])
                try:
                    arg_list = []
                    for key, val in f['arguments'].items():
                        if type(val) == str and val.startswith("$") and not val.endswith("$") :
                            val = val + "$"
                        arg_list.append(f"{key} = {val}")
                    args = ", ".join(sorted(arg_list))
                except:
                    args = {}
                api_with_args_pred.append(f'{f_name}({args})')
            except:
                continue
        api_with_args_gold, api_with_args_pred = post_process_api_with_args(api_with_args_gold, api_with_args_pred)

        try:
            accuracy_combined = accuracy_score(api_with_args_gold, api_with_args_pred)
        except:
            accuracy_combined = 0.0
        if accuracy_combined == 1:
            all_num_times_full_score += 1
        all_accuracy_combined.append(accuracy_combined)

        ## WinRate
        if win_rate_flag:
            win_score = calculate_win_score(pred_dict_list, item["gold_answer"], item["tools"], executable_func_dir)
            win_rate_list.append(win_score)

    p_intent, r_intent, f1_intent = compute_score_sklearn(gold_output_intent, pred_output_intent)
    p_slot, r_slot, f1_slot = compute_score_sklearn(gold_output_slot, pred_output_slot)

    return {
        "p_intent": "{:.3f}".format(p_intent),
        "r_intent": "{:.3f}".format(r_intent),
        "f1_intent": "{:.3f}".format(f1_intent),
        "p_slot": "{:.3f}".format(p_slot) if p_slot != None else '',
        "r_slot": "{:.3f}".format(r_slot) if r_slot != None else '',
        "f1_slot": "{:.3f}".format(f1_slot) if f1_slot != None else '',
        'num_examples': len(predictions),
        'accuracy_combined': "{:.3f}".format(statistics.mean(all_accuracy_combined)),
        'percentage_times_full_score': "{:.3f}".format(all_num_times_full_score/len(predictions)),
        'win_rate': "{:.3f}".format(sum(win_rate_list) / len(win_rate_list)) if win_rate_flag else "no",
        'num_errors_parsing_pred_intent': num_errors_parsing_pred_intent,
        'num_errors_parsing_gold_intent': num_errors_parsing_gold_intent,
        'num_errors_parsing_pred_slot': num_errors_parsing_pred_slot,
        'num_errors_parsing_gold_slot': num_errors_parsing_gold_slot,
        'num_pred_examples_w_parsing_errors': num_pred_examples_w_parsing_errors
    }    

def print_result(result, model):
    print("\n###################################")
    print(f"##### {model} #####")
    print("###################################")
    print(f"Total Samples: {result['num_examples']}")
    print(f"Parsing Errors: {result['num_pred_examples_w_parsing_errors']}")
    print(f"F1 Intent: {result['f1_intent']}")
    print(f"F1 Slot: {result['f1_slot']}")
    print(f"Partial Match Accuracy: {result['accuracy_combined']}")
    print(f"Full Match Accuracy: {result['percentage_times_full_score']}")
    print(f"Win Rate: {result['win_rate']}")
    print("-"*100)

def calculate_weighted_avg_score(res_lst):
    f1_intent_total = 0
    f1_slot_total = 0
    par_acc_total = 0
    full_acc_total = 0
    alt_traj_total = 0
    win_total = 0
    examples_total = 0
    parsing_errors_total = 0
    for res in res_lst:
        f1_intent_total += res["num_examples"] * float(res['f1_intent'])
        f1_slot_total += res["num_examples"] * float(res['f1_slot'])
        par_acc_total += res["num_examples"] * float(res['accuracy_combined'])
        full_acc_total += res["num_examples"] * float(res['percentage_times_full_score'])
        alt_traj_total += res["num_examples"] * float(res['alternate_trajectory_score'])
        win_total += res["num_examples"] * float(res['win_rate'])
        examples_total += res["num_examples"]
        parsing_errors_total += res["num_pred_examples_w_parsing_errors"]
    avg_res = {
        "num_examples": examples_total,
        "num_pred_examples_w_parsing_errors": parsing_errors_total, 
        "f1_intent": "{:.3f}".format(f1_intent_total / examples_total),
        "f1_slot": "{:.3f}".format(f1_slot_total / examples_total),
        "accuracy_combined": "{:.3f}".format(par_acc_total / examples_total),
        "percentage_times_full_score": "{:.3f}".format(full_acc_total / examples_total),
        "win_rate": "{:.3f}".format(win_total / examples_total),
        "alternate_trajectory_score": "{:.3f}".format(alt_traj_total / examples_total),
    }
    return avg_res
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--result_file_path", type=str)
    parser.add_argument("--executable_func_dir", type=str)
    args = parser.parse_args()

    data = read_jsonlines(args.result_file_path)
    result = calculate_scores(data, args.model_name, args.executable_func_dir)
    print_result(result, args.model_name)