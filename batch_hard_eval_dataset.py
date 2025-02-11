import json
import time

import openai
from typing import Dict, Iterable

import os
from os import PathLike
import gzip
from math_utils import _strip_string
import math
import re
from copy import deepcopy
import random
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from collections import Counter
# from permsc import KemenyOptimalAggregator, sum_kendall_tau, ranks_from_preferences
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


class Get:
    def __init__(self):
        self.prompt = ""

    def calc(self, questions, instruction="", temp=0, n=1, model=3.5, rank_prompt="short", task=""):
        openai.api_type = "please input your type" 
        openai.api_base = "please input your api"
        openai.api_version = "2023-05-15"
        if model == 3.5:
            openai.api_key = "please input your key"
            id = "gpt-35-turbo"
        elif model == 4:
            openai.api_key = "please input your key"
            id = "gpt-4"
        else:
            openai.api_key = "please input your key"
            id = "gpt-4"


        if rank_prompt == "long":
            instruction = "You will be given a batch of {} questions. Your task is to rank them from easy to hard based on their difficulty level. You should carefully horizontally compare the given questions in order to assign a suitable ranking place to each question. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed. Questions to be evaluated:\n".format(len(questions))
            post_instruction = "Evaluation Form (Answer by starting with \"I will do my best to provide individual analysis for each sample. Analysis:\" to analyze the given questions  as concise as possible (Attention: Don’t give your ranking during this step). After analyzing all the questions, please give all the ranking place (from easy to hard) in order following the template \"Ranking: [Q{{number of the easiest question}},...,Q{{number of the hardest question}}]\".- Difficulty:"
        else:
            instruction = "Your task is to rank the given questions from easy to hard based on their difficulty level. Questions to be evaluated:\n"
            post_instruction = "The output format should be a comma-separated list containing the Q{number of corresponding question}. Do not give any explain. Difficulty Ranking result (from easy to hard):"
        cur = []
        # for tem in task_prompts:
        #     cur.append({"role": "user", "content": tem["Q"]})
        #     cur.append({"role": "assistant", "content": tem["A"]})
        query = ""
        for i in range(len(questions)):
            query += "Q" + str(i+1) + ": " + questions[i] + "\n"

        count = 0
        while True:
            messages = [{"role": "system", "content": instruction}] + cur + [{"role": "user", "content": query + post_instruction}]
            # print(messages)
            try:
                response = openai.ChatCompletion.create(
                    deployment_id=id,
                    messages=messages,
                    top_p=0.9,
                    n=n,
                    temperature=temp,
                    request_timeout=300
                )
                res = [tem["message"]["content"] for tem in response["choices"]]
                return res
            except Exception as e:
                if count > 5:
                    return [" "]
                print("An error occurred:", e)
                count += 1
                time.sleep(18)

def extract_all_number(s):
    numbers = re.findall(r'Q(\d+)', s)
    if numbers:
        return [int(number) for number in numbers]
    else:
        return []
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


def extract_math_answer(pred_str):
    # if ('The answer is ' in pred_str):
    #    pred = pred_str.split('The answer is ')[-1].strip()
    # elif ('the answer is ' in pred_str):
    #    pred = pred_str.split('the answer is ')[-1].strip()
    # elif 'boxed' in pred_str:
    #    ans = pred_str.split('boxed')[-1]
    #    if (len(ans) and ans[0] == '{'):
    #        stack = 1
    #        a = ''
    #        for c in ans[1:]:
    #            if (c == '{'):
    #                stack += 1
    #                a += c
    #            elif (c == '}'):
    #                stack -= 1
    #                if (stack == 0): break
    #                a += c
    #            else:
    #                a += c
    #    else:
    #        a = ans.split('$')[0].strip()
    #    a = _strip_string(a)
    #    pred = a

    # else:
    #    pattern = '-?\d*\.?\d+'
    #    pred = re.findall(pattern, pred_str)
    #    if (len(pred) >= 1):
    #        # print(pred_str)
    #        pred = pred[-1]
    #    else:
    #        pred = ''
    # if pred != "":
    #    if pred[-1] == ".":
    #        pred = pred[:-1]
    #    if pred[-1] == "/":
    #        pred = pred[:-1]
    # pred = _strip_string(pred)
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
                pred = ''
            if pred != "":
                if pred[-1] == ".":
                    pred = pred[:-1]
    except:
        pred = ""
    return pred


def extract_gsm8k_answer(pred_str):
    pred_str = re.sub(r'(\d),(\d)', r'\1\2', pred_str)
    if ('The answer is ' in pred_str):
       pred_str = pred_str.split('The answer is ')[-1].strip()
    elif ('the answer is ' in pred_str):
       pred_str = pred_str.split('the answer is ')[-1].strip()

    if 'boxed' in pred_str and 1==2:
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
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]

    else:
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


def find_math_answer(s):
    assert ('boxed' in s)
    # s = s.replace(",", "")
    ans = s.split('boxed')[-1]
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

    if "\\text{" in a:
        a = re.search(r'\\text\{(.*?)\}', a)
        if a:
            a = a.group(1)

    return a


def ESC(input_list):
    esc_count = 1
    esc_num = 0
    while True:
        if esc_num + 4 >= len(input_list):
            break
        sub = input_list[esc_num]
        flag = 1
        for i in range(1, 5):
            if sub != input_list[esc_num+i]:
                flag = 0
                break

        if flag == 1:
            break
        else:
            esc_count += 1
            esc_num += 5

    return esc_count

def cal_spear(result_list1, result_list2):
    spearman = []
    pearson = []
    for i in range(len(result_list1)):
        pearson_corr, _ = pearsonr(result_list1[i], result_list2[i])
        spearman_corr, _ = spearmanr(result_list1[i], result_list2[i])
        spearman.append(spearman_corr)
        pearson.append(pearson_corr)

    return sum(spearman) / len(spearman), sum(pearson) / len(pearson)

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def sort_and_record(A):
    indexed_A = list(enumerate(A))
    sorted_A = sorted(indexed_A, key=lambda x: x[1])
    sorted_indexes = [i[0] for i in sorted_A]
    sorted_A = [i[1] for i in sorted_A]
    return sorted_A, sorted_indexes


def divide_into_batches(sorted_A, batch_size):
    avg_len = len(sorted_A) // batch_size
    out = [[] for i in range(batch_size)]
    index = 0
    while True:
        if index >= batch_size:
            break
        if index == batch_size - 1:
            out[index].extend(sorted_A[index*avg_len:])
        else:
            out[index].extend(sorted_A[index*avg_len:(index+1)*avg_len])

        index += 1

    return out


def batchify(input_list, batch_size, strategy="random", score_list=None):
    flag = True
    score = []
    for i in range(len(score_list)):
        try:
            score.append(sum(score_list[i])/len(score_list[i]))
        except:
            flag = False

    if flag == False:
        strategy = "random"

    if strategy == "random" or strategy=="permsc":
        index_list = list(range(len(input_list)))
        random.shuffle(index_list)
        batches = [index_list[i:i+batch_size] for i in range(0, len(input_list), batch_size)]
        out = []
        for j in range(len(batches)):
            batch = batches[j]
            sub = []
            if len(batch) < batch_size:
                for k in range(batch_size-len(batch)):
                    batch.append(batches[j-1][k])

            for i in batch:
                sub.append((i, input_list[i]))
            out.append(deepcopy(sub))
    else:
        sorted_score, sorted_indexes = sort_and_record(score)
        sorted_indexes_batchfy = divide_into_batches(sorted_indexes, batch_size)
        for i in range(len(sorted_indexes_batchfy)):
            random.shuffle(sorted_indexes_batchfy[i])

        out = []
        for j in range(len(sorted_indexes_batchfy[0])):
            sub = []
            for i in range(batch_size):
                sub.append((sorted_indexes_batchfy[i][j], input_list[sorted_indexes_batchfy[i][j]]))
            out.append(deepcopy(sub))

        index = 0
        for j in range(len(sorted_indexes_batchfy[0]), len(sorted_indexes_batchfy[-1])):
            sub = []
            for i in range(batch_size):
                if len(sorted_indexes_batchfy[i]) == len(sorted_indexes_batchfy[0]):
                    sub.append((sorted_indexes_batchfy[i][index], input_list[sorted_indexes_batchfy[i][index]]))
                else:
                    sub.append((sorted_indexes_batchfy[i][j], input_list[sorted_indexes_batchfy[i][j]]))
            out.append(deepcopy(sub))
            index += 1
        # batch diverse strategy
    
    return out

def mean(nums):
    return sum(nums) / len(nums)


def shuffle_list_n_times(input_list, n):
    shuffled_lists = []
    index_transforms = []

    for _ in range(n):
        index_list = list(range(len(input_list)))
        random.shuffle(index_list)
        index_transforms.append(index_list)

        shuffled_list = [input_list[i] for i in index_list]
        shuffled_lists.append(shuffled_list)

    return shuffled_lists, index_transforms

def process_task(k, Gen, perm_questions, temp, model, rank_prompt, dataset):
    gen_preference = Gen.calc(questions=perm_questions[k], temp=temp, n=1, model=model, rank_prompt=rank_prompt, task=dataset)[:1]
    return extract_all_number(gen_preference[0].split("Ranking:")[-1])

def remove_short_lists(input_list, index_list, length):
    out_list = []
    out_index_list = []
    for i in range(len(input_list)):
        if len(input_list[i]) == length:
            out_list.append(input_list[i])
            out_index_list.append(index_list[i])

    return out_list, out_index_list

if __name__ == "__main__":
    Gen = Get()
    model = 4
    temp = 0
    n = 1
    samples = []
    i = 0
    batch_size = 8
    iteration = 10
    strategy = ["random", "diverse", "permsc"]
    dataset = ["GSM8K", "MATH", "MATH_all", "coin_flip", "last_letter", "strategy", "common"]
    rank_prompt = ["short", "long"]
    # dataset_sub = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    esc_result = []
    rank_result = []
    strategy = strategy[0]
    dataset = dataset[2]
    rank_prompt = rank_prompt[0]
    # dataset_sub = dataset_sub[0]
    ori_sc_result = []
    asc_result = []

    label_all = []
    predict = []

    mismatch_count = 0
    count = 0
    error_count = 0
    # if dataset == "MATH":
    #     output_path = "result/{}_{}_gpt_{}_eval_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, batch_size, iteration, n, temp)
    # elif dataset == "GSM8K":
    #     output_path = "result/{}_gpt_{}_eval_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(dataset, model, batch_size, iteration, n, temp)

    input_map = {"GSM8K": "result/gsm8k_gpt_4_samples_n_50_temp_0.8.jsonl",
                 "coin_flip": "coin_flip.jsonl",
                 "last_letter": "last_letter.jsonl",
                 "strategy": "strategy.jsonl",
                 "common": "common.jsonl",
                 "MATH_all": "MATH_all.jsonl"}

    if dataset in ["GSM8K", "coin_flip", "last_letter", "strategy", "common", "MATH_all"]:
        input_path = input_map[dataset]

        with open(input_path, "r") as f:
            data = f.readlines()
            f.close()

        pre = []
        answer = []
        pre_batch = []
        question = []

        for line in data:
            tem = json.loads(line)
            pre.append(tem)
            if dataset == "MATH_all":
                question.append(tem["problem"])
            else:
                question.append(tem["input"])
            if dataset == "GSM8K":
                answer.append(extract_answer(tem['answer']))
            elif dataset == "MATH_all":
                answer.append(tem['answer'])
            else:
                answer.append(extract_answer(tem['output']))
            # shu_batch = tem['completion']
            # predict_batch = [extract_gsm8k_answer(item) for item in shu_batch]
            # pre_batch.append(deepcopy(predict_batch))

        eval_all = [[] for i in range(len(pre))]
        for iter_num in tqdm(range(iteration)):
            samples = []
            random_set = batchify(question, batch_size, strategy, eval_all)
            # pre_batch_sub = pre_batch[i*batch_size:(i+1)*batch_size]
            # pre_answer_sub = pre_answer[i*batch_size:(i+1)*batch_size]
            #
            # pre_sub = pre[i*batch_size:(i+1)*batch_size]
            # question_sub = question[i*batch_size:(i+1)*batch_size]
            error_gen = 0
            for i in tqdm(range(len(random_set))):
                subset = random_set[i]
                subset_question = []
                subset_index = []
                for item in subset:
                    subset_question.append(item[1])
                    subset_index.append(item[0])
                if strategy=="permsc":
                    perm_questions, perm_index = shuffle_list_n_times(subset_question, n)
                    perm_preference, perm_preference_map = [], []
                    # for k in range(len(perm_questions)):
                    #     gen_preference = Gen.calc(questions=perm_questions[k], temp=temp, n=1, model=model, rank_prompt=rank_prompt, task=dataset)[:1]
                    #     perm_preference.append(extract_all_number(gen_preference[0].split("Ranking:")[-1]))

                    # parallel implement
                    with ProcessPoolExecutor() as executor:
                        futures = [executor.submit(process_task, k, Gen, perm_questions, temp, model, rank_prompt, dataset) for k in range(len(perm_questions))]

                    perm_preference = [future.result() for future in futures]
                    pre_len = len(perm_preference)
                    perm_preference, perm_index = remove_short_lists(perm_preference, perm_index, batch_size)
                    post_len = len(perm_preference)
                    error_gen += pre_len - post_len

                    for j in range(len(perm_preference)):
                        preference_completion_map_sub = [0 for k in range(len(perm_preference[j]))]
                        try:
                            for index in range(len(preference_completion_map_sub)):
                                preference_completion_map_sub[index] = perm_index[j][perm_preference[j][index] - 1] # turn 1-8 to 0-7
                            perm_preference_map.append(preference_completion_map_sub)
                        except:
                            error_count += 1

                    perm_preference_map = np.array(perm_preference_map)
                    preference_optimal = KemenyOptimalAggregator().aggregate(perm_preference_map, verbose=False)
                    rank_optimal = ranks_from_preferences(preference_optimal)
                    for j in range(len(rank_optimal)):
                        eval_all[subset_index[j]].append(rank_optimal[j])


                else:
                    rank = Gen.calc(questions=subset_question, temp=temp, n=n, model=model, rank_prompt=rank_prompt, task=dataset)[:n]
                    # samples.append(dict(questions=subset_random, completion=rank, label=sub_label_random))
                    # write_jsonl("result/{}_{}_gpt_{}_eval_test_num_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, test_num, n, temp), samples)

                    rank_completion = []
                    for j in range(len(rank)):
                        sub = extract_all_number(rank[j].split("Ranking:")[-1])
                        if len(sub) == batch_size:
                            rank_completion.append(extract_all_number(rank[j].split("Ranking:")[-1]))
                        else:
                            error_gen += 1
                    rank_completion_map = []
                    for j in range(len(rank_completion)):
                        rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
                        try:
                            for index in range(len(rank_completion_map_sub)):
                                rank_completion_map_sub[rank_completion[j][index] - 1] = index + 1
                            rank_completion_map.append(rank_completion_map_sub)
                        except:
                            error_count += 1

                    rank_completion_map = list(map(mean, zip(*rank_completion_map)))
                    # rank_completion_map = [item / n for item in rank_completion_map]

                    for j in range(len(rank_completion_map)):
                        eval_all[subset_index[j]].append(rank_completion_map[j])

            error = 0
            eval_all_sub = []
            for tem in eval_all:
                try:
                    eval_all_sub.append(sum(tem) / len(tem))
                except:
                    eval_all_sub.append(0)
                    error += 1

            print("gen error:{}".format(error_gen))
            print("sum error:{}".format(error))
            samples.append(dict(questions=question, eval=eval_all_sub, answer=answer))
            write_jsonl(
                "result/{}_gpt_{}_eval_strategy_{}_prompt_{}_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(dataset, model,
                                                                                                   strategy, rank_prompt, batch_size,
                                                                                                   iter_num + 1, n,
                                                                                                   temp), samples)
            # samples.append(dict(questions=question_sub, completion=rank))
            # write_jsonl("result/{}_gpt_{}_eval_batchsize_{}_n_{}_temp_{}.jsonl".format(dataset, model, batch_size, n, temp), samples)

        print("error count:{}".format(error_count))
        print("done!")



    if dataset == "MATH":
        input_path = "MATH.jsonl"
        ori_data = []
        final_eval = []
        index = 0
        with open(input_path, "r") as f:
            data = f.readlines()
            f.close()

        type_dict = {}

        for line in data:
            tem = json.loads(line)
            ori_data.append(deepcopy(tem))
            type = tem["subject"].replace(" ", "_")
            if type not in type_dict.keys():
                type_dict[type] = {"hard_level": [], "problem": [], "answer": [], "eval_all":[], "index":[]}

            type_dict[type]["problem"].append(tem["problem"])
            type_dict[type]["answer"].append(tem["answer"])
            # shu_batch = tem['generated_answer']
            type_dict[type]["hard_level"].append(tem["level"])
            type_dict[type]["index"].append(index)

            # predict_batch = [extract_math_answer(item) for item in shu_batch]
            # type_dict[type]["generated_answer"].append(deepcopy(predict_batch))
            type_dict[type]["eval_all"].append([])
            final_eval.append(0)
            index += 1


        for key in type_dict.keys():
            print(key)
            count = 0
            error_count = 0
            answer = type_dict[key]["answer"]
            # pre_batch = type_dict[key]["generated_answer"]
            question = type_dict[key]["problem"]
            eval_all = type_dict[key]["eval_all"]
            hard_level = type_dict[key]["hard_level"]
            sub_index = type_dict[key]["index"]

            for iter_num in tqdm(range(iteration)):
                samples = []
                random_set = batchify(question, batch_size, strategy, eval_all)
                # pre_batch_sub = pre_batch[i*batch_size:(i+1)*batch_size]
                # pre_answer_sub = pre_answer[i*batch_size:(i+1)*batch_size]
                #
                # pre_sub = pre[i*batch_size:(i+1)*batch_size]
                # question_sub = question[i*batch_size:(i+1)*batch_size]
                error_gen = 0
                for i in tqdm(range(len(random_set))):
                    subset = random_set[i]
                    subset_question = []
                    subset_index = []
                    for item in subset:
                        subset_question.append(item[1])
                        subset_index.append(item[0])

                    if strategy == "permsc":
                        perm_questions, perm_index = shuffle_list_n_times(subset_question, n)
                        perm_preference, perm_preference_map = [], []
                        # for k in range(len(perm_questions)):
                        #     gen_preference = Gen.calc(questions=perm_questions[k], temp=temp, n=1, model=model, rank_prompt=rank_prompt,
                        #                               task=dataset)[:1]
                        #     perm_preference.append(extract_all_number(gen_preference[0].split("Ranking:")[-1]))

                        with ProcessPoolExecutor() as executor:
                            futures = [
                                executor.submit(process_task, k, Gen, perm_questions, temp, model, rank_prompt, dataset)
                                for k in range(len(perm_questions))]

                        perm_preference = [future.result() for future in futures]
                        pre_len = len(perm_preference)
                        perm_preference, perm_index = remove_short_lists(perm_preference, perm_index, batch_size)
                        post_len = len(perm_preference)
                        error_gen += pre_len - post_len

                        for j in range(len(perm_preference)):
                            preference_completion_map_sub = [0 for k in range(len(perm_preference[j]))]
                            try:
                                for index in range(len(preference_completion_map_sub)):
                                    preference_completion_map_sub[index] = perm_index[j][perm_preference[j][index] - 1]  # turn 1-8 to 0-7
                                perm_preference_map.append(preference_completion_map_sub)
                            except:
                                error_count += 1

                        perm_preference_map = np.array(perm_preference_map)
                        preference_optimal = KemenyOptimalAggregator().aggregate(perm_preference_map, verbose=False)
                        rank_optimal = ranks_from_preferences(preference_optimal)
                        for j in range(len(rank_optimal)):
                            eval_all[subset_index[j]].append(rank_optimal[j])

                    else:
                        rank = Gen.calc(questions=subset_question, temp=temp, n=n, model=model, rank_prompt=rank_prompt, task=dataset)[:n]
                        # samples.append(dict(questions=subset_random, completion=rank, label=sub_label_random))
                        # write_jsonl("result/{}_{}_gpt_{}_eval_test_num_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, test_num, n, temp), samples)

                        rank_completion = []
                        for j in range(len(rank)):
                            sub = extract_all_number(rank[j].split("Ranking:")[-1])
                            if len(sub) == batch_size:
                                rank_completion.append(extract_all_number(rank[j].split("Ranking:")[-1]))
                            else:
                                error_gen += 1
                        rank_completion_map = []
                        for j in range(len(rank_completion)):
                            rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
                            try:
                                for index in range(len(rank_completion_map_sub)):
                                    rank_completion_map_sub[rank_completion[j][index] - 1] = index
                                rank_completion_map.append(rank_completion_map_sub)
                            except:
                                error_count += 1

                        rank_completion_map = list(map(mean, zip(*rank_completion_map)))
                        # rank_completion_map = [item / n for item in rank_completion_map]

                        for j in range(len(rank_completion_map)):
                            eval_all[subset_index[j]].append(rank_completion_map[j])

                error = 0
                eval_all_sub = []
                for tem in eval_all:
                    try:
                        eval_all_sub.append(sum(tem) / len(tem))
                    except:
                        eval_all_sub.append(0)
                        error += 1

                print("gen error:{}".format(error_gen))
                print("sum error:{}".format(error))
                samples.append(dict(type=key, questions=question, eval=eval_all_sub, answer=answer, hard_level=hard_level))
                write_jsonl(
                    "result/{}_{}_gpt_{}_eval_strategy_{}_prompt_{}_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(dataset, key, model,
                                                                                                       strategy, rank_prompt, batch_size,
                                                                                                       iter_num + 1, n,
                                                                                                       temp), samples)
                # samples.append(dict(questions=question_sub, completion=rank))
                # write_jsonl("result/{}_gpt_{}_eval_batchsize_{}_n_{}_temp_{}.jsonl".format(dataset, model, batch_size, n, temp), samples)

            print("error count:{}".format(error_count))
            print("done!")
            for i in range(len(eval_all)):
                ori_data[sub_index[i]]["eval"] = sum(eval_all[i]) / len(eval_all[i])


        out_path = "result/{}_gpt_{}_eval_strategy_{}_prompt_{}_batchsize_{}_iter_{}_n_{}_temp_{}.jsonl".format(dataset, model,
                                                                                                       strategy, rank_prompt, batch_size,
                                                                                                       iteration, n,
                                                                                                       temp)
        with open(out_path, 'w') as f:
            for item in ori_data:
                f.write(json.dumps(item) + '\n')
            f.close()

        print("saved!")



    #
    # if dataset == "MATH":
    #     if os.path.exists(output_path):
    #         with open(output_path, "r") as f:
    #             for line in f:
    #                 tem = json.loads(line)
    #
    #                 rank_completion = []
    #                 for j in range(len(tem["completion"])):
    #                     rank_completion.append(extract_all_number(tem["completion"][j].split("Ranking:")[-1]))
    #
    #                 rank_completion_map = []
    #                 for j in range(len(rank_completion)):
    #                     rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
    #                     for index in range(len(rank_completion_map_sub)):
    #                         rank_completion_map_sub[rank_completion[j][index] - 1] = index + 1
    #                     rank_completion_map.append(rank_completion_map_sub)
    #
    #                 rank_completion_map = list(map(sum, zip(*rank_completion_map)))
    #                 rank_completion_map = [item/n for item in rank_completion_map]
    #
    #                 predict.append(rank_completion_map)
    #                 label_all.append(tem["label"])
    #             f.close()
    #     else:
    #         input_path = "MATH/{}".format(dataset_sub)
    #         json_files = [pos_json for pos_json in os.listdir(input_path) if pos_json.endswith('.json')]
    #
    #         data = []
    #         for json_file in json_files:
    #             json_file_path = os.path.join(input_path, json_file)
    #             with open(json_file_path, 'r') as j:
    #                 data.append(json.load(j))
    #         hard_level = {"1": [], "2": [], "3": [], "4": [], "5": []}
    #         for item in data:
    #             sub_level = item["level"][-1]
    #             hard_level[sub_level].append(item["problem"])
    #
    #
    #         for i in tqdm(range(repeat)):
    #             subset = []
    #             sub_label = []
    #             for key in ["1", "2", "3", "4", "5"]:
    #                 subset += random.sample(hard_level[key], test_num)
    #                 sub_label += [int(key) for j in range(test_num)]
    #
    #             random_indices = random.sample(range(len(subset)), len(subset))
    #             subset_random = [0 for j in range(len(subset))]
    #             sub_label_random = [0 for j in range(len(subset))]
    #
    #             for j in range(len(random_indices)):
    #                 subset_random[j] = subset[random_indices[j]]
    #                 sub_label_random[j] = sub_label[random_indices[j]]
    #
    #             rank = Gen.calc(questions=subset_random, temp=temp, n=n, model=model, rank_prompt=rank_prompt, task=dataset)[:n]
    #             samples.append(dict(questions=subset_random, completion=rank, label=sub_label_random))
    #             write_jsonl("result/{}_{}_gpt_{}_eval_test_num_{}_n_{}_temp_{}.jsonl".format(dataset, dataset_sub, model, test_num, n, temp), samples)
    #
    #             rank_completion = []
    #             for j in range(len(rank)):
    #                 rank_completion.append(extract_all_number(rank[j].split("Ranking:")[-1]))
    #
    #             rank_completion_map = []
    #             for j in range(len(rank_completion)):
    #                 rank_completion_map_sub = [0 for k in range(len(rank_completion[j]))]
    #                 for index in range(len(rank_completion_map_sub)):
    #                     rank_completion_map_sub[rank_completion[j][index]-1] = index + 1
    #                 rank_completion_map.append(rank_completion_map_sub)
    #
    #             rank_completion_map = list(map(sum, zip(*rank_completion_map)))
    #             rank_completion_map = [item/n for item in rank_completion_map]
    #
    #             predict.append(rank_completion_map)
    #             label_all.append(sub_label_random)



        # spearman_avg, pearson_avg = cal_spear(predict, label_all)
        #
        # print("spearman: {}".format(spearman_avg))
        # print("pearson: {}".format(pearson_avg))

