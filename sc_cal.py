import math
import os
import operator
import json
import re

import numpy as np
import argparse
from math_utils import delete_extra_zero, _strip_string
# from math_equivalence import is_equiv
import statistics
import random
from tqdm import tqdm, trange
from copy import deepcopy

def extract_answer(completion):
    INVALID_ANS = "[invalid]"
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def extract_last_number(s):
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[-1])
    else:
        return None


def extract_last_letter_answer(generated_answer):
    answer_text = generated_answer.lower().split('the answer is')[-1]
    answer_text = ''.join(re.split(r'[^A-Za-z]', answer_text))
    return answer_text


def extract_strategy_answer(generated_answer):
    if 'the answer is yes' in generated_answer.lower():
        return "Yes"
    elif 'the answer is no' in generated_answer.lower():
        return "No"
    else:
        if "uncertain" in generated_answer.lower() or "unknown" in generated_answer.lower():
            return ""
        judge = generated_answer.strip("A: ").split(",")[0]
        if judge == "Yes" or judge == "No":
            return judge
        
        judge2 = generated_answer.lower()

        if "yes" in judge2 and "no" not in judge2:
            #print(generated_answer)
            return "Yes"
        if "no" in judge2 and "yes" not in judge2:
            #print(generated_answer)
            return "No"
        #print(generated_answer)
        return ""


def extract_coin_flip_answer(generated_answer):
    if 'the answer is yes' in generated_answer.lower():
        return "yes"
    elif 'the answer is no' in generated_answer.lower():
        return "no"
    else:
        return ""


def extract_common_answer(generated_answer):
    answer_text = generated_answer.split('the answer is')[-1]
    _ = answer_text
    p = re.compile(r'[(](.*)[)]', re.S)
    answer_text = re.findall(p, answer_text)
    if answer_text:
        return answer_text[0].upper()
    else:
        return ""


def extract_math_answer(pred_str):
    try:
        if 'boxed' in pred_str:
            ans = pred_str.split('boxed')[-1]
            if (ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0): break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            a = _strip_string(a)
            pred = a
        elif ('the answer is ' in pred_str):
            pred = pred_str.split('the answer is ')[-1].strip()
        elif ('The answer is ' in pred_str):
            pred = pred_str.split('The answer is ')[-1].strip()
        else:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if (len(pred) >= 1):
                pred = pred[-1]
            else:
                pred=''
            if pred != "":
                if pred[-1] == ".":
                    pred = pred[:-1]
    except:
        pred = ""
    return pred

def extract_gsm8k_answer(pred_str):
    pred_str = re.sub(r'(\d),(\d)', r'\1\2', pred_str)
    pattern = '-?\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if (len(pred) >= 1):
        # print(pred_str)
        pred = pred[-1]
    else:
        pred = ''
    try:
        if pred != '' and math.floor(float(pred)) == float(pred):
            pred = str(int(float(pred)))
    except:
        a = 1
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="result/MATH_gpt_4_samples_n_50_temp_0.5.jsonl")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    input_path = args.input_path
    n = args.n
    repeat = args.repeat
    dataset = input_path.split("_gpt_")[0].split("/")[-1]
    #dataset = ["MATH", "GSM8K"]
    
    #dataset = dataset[1]


    correct_distribute = []
    wrong_distribute = []
    accuracy_list = []
    accuracy_list_level = [[] for i in range(5)]
    accuracy_list_subject = {}
    accuracy_list_detail = {}
    with open(input_path, "r") as f:
        data = f.readlines()
        f.close()

    all_dict = {}


    pre = []
    pre_answer = []
    pre_batch = []
    count = 0
    unmatch_count = 0
    for line in data:
        tem = json.loads(line)
        pre.append(tem)
        if dataset == "MATH":
            pre_answer.append(tem['answer'])
        elif dataset == "GSM8K":
            pre_answer.append(extract_answer(tem['answer']))
        else:
            pre_answer.append(tem['answer'])
        shu_batch = tem['completion']
        if dataset == "MATH":
            predict_batch = [extract_math_answer(item) for item in shu_batch]
        elif dataset == "GSM8K":
            predict_batch = [extract_gsm8k_answer(item) for item in shu_batch]
        elif dataset == "common":
            predict_batch = [extract_common_answer(item) for item in shu_batch]
        elif dataset == "coin_flip":
            predict_batch = [extract_coin_flip_answer(item) for item in shu_batch]
        elif dataset == "last_letter":
            predict_batch = [extract_last_letter_answer(item) for item in shu_batch]
        elif dataset == "strategy":
            predict_batch = [extract_strategy_answer(item) for item in shu_batch]

        for item in predict_batch:
            if item == '':
                unmatch_count += 1
        pre_batch.append(deepcopy(predict_batch))

    print("unmatch count={}".format(unmatch_count))

    for repeat_num in tqdm(range(repeat)):
        for sc_num in [n]:
            result_list = []
            result_list_level = [[] for _ in range(5)]
            result_list_subject = {}
            result_list_detail = {}
            num_list = []
            right_nums = 0
            n_nums = 0
            count = 0
            for i in range(len(pre)):
                flag = False
                pre_dict = {}
                tem = pre[i]
                answer = pre_answer[i]
               
                shu_batch = tem['completion']
                if dataset == "MATH":
                    hard_level = tem['hard_level']
                    subject = tem['subject']
                    if subject not in result_list_subject.keys():
                        result_list_subject[subject] = []
                        result_list_detail[subject] = [[] for i in range(5)]

                predict_batch = pre_batch[i]
                #print(len(predict_batch))
                if len(predict_batch) < sc_num:
                    count += 1
                    continue
                predict_batch = random.sample(predict_batch, sc_num)
                for item in predict_batch:
                    if item in pre_dict.keys():
                        pre_dict[item] += 1
                    else:
                        pre_dict[item] = 1
            
                sc_pre = max(pre_dict, key=pre_dict.get)
                result_list.append(sc_pre==answer)
                if dataset == "MATH":
                    result_list_level[hard_level-1].append(sc_pre==answer)
                    result_list_subject[subject].append(sc_pre==answer)
                    result_list_detail[subject][hard_level-1].append(sc_pre==answer)

            accuracy_list.append(np.array(result_list).mean())
            if dataset == "MATH":
                for k in range(5):
                    accuracy_list_level[k].append(np.array(result_list_level[k]).mean())
                for key in result_list_subject.keys():
                    if key not in accuracy_list_subject.keys():
                        accuracy_list_subject[key] = []
                        accuracy_list_detail[key] = [[] for i in range(5)]

                    accuracy_list_subject[key].append(np.array(result_list_subject[key]).mean())
                    for i in range(5):
                        accuracy_list_detail[key][i].append(np.array(result_list_detail[key][i]).mean())

    
    #print(accuracy_list)
    print("mean_acc = {}".format(np.array(accuracy_list).mean()))
    #print("total wrong_num = {}".format(count))

    if dataset == "MATH":
        print("---------------------- level -----------------------")
        for level in range(5):
            print("level{} mean_acc = {}".format(level+1, np.array(accuracy_list_level[level]).mean()))

        print("---------------------- subject -----------------------")
        for key in accuracy_list_subject.keys():
            print("{} mean_acc = {}".format(key, np.array(accuracy_list_subject[key]).mean()))

        print("--------------------- detail -----------------------")
        for key in accuracy_list_detail.keys():
            print(key)
            for level in range(5):
                print("         " + "level{} mean_acc = {}".format(level + 1, np.array(accuracy_list_detail[key][level]).mean()))

