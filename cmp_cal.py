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
from stopping_criterias import BetaStoppingCriteria
from typing import List, Dict
from collections import Counter
from collections import Counter
from scipy import integrate, stats

from fast_chat.model import load_model, get_conversation_template, add_model_args

class StoppingCriterias:

    def __init__(self, *args, **kwargs):

        ...

    def should_stop(self, *args, **kwargs) -> Dict:
        ...

class MyStoppingCriteria(StoppingCriterias):

    def __init__(self, easy_conf_thresh: float = 0.95, hard_conf_thresh = 0.50) -> None:
        super().__init__()
        self.easy_conf_thresh = easy_conf_thresh
        self.hard_conf_thresh = hard_conf_thresh

    def should_stop(self, answers: List, easy_conf_thresh: int = None, hard_conf_thresh: int = None, verbose: bool = False) -> Dict:

        if easy_conf_thresh is None: easy_conf_thresh = self.easy_conf_thresh
        if hard_conf_thresh is None: hard_conf_thresh = self.hard_conf_thresh

        most_common = Counter(answers).most_common(2)
        if len(most_common) == 1:
            a, b = most_common[0][1], 0
        else:
            a, b = most_common[0][1], most_common[1][1]
        a = float(a)
        b = float(b)

        return_dict = {
            'most_common': most_common[0][0],
            'prob': -1,
            'easy_stop': False,
            'hard_stop': False,
        }

        try:
            prob = integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0.5, 1)[0] / \
                   integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0, 1)[0]
        except Exception as e:
            # print error message
            print(f"Error during numerical integration: {e}")
            return_dict['easy_stop'] = False
            return_dict['prob'] = -1
            return return_dict
        return_dict['prob'] = prob
        return_dict['easy_stop'] = prob >= easy_conf_thresh
        return_dict['hard_stop'] = prob <= hard_conf_thresh
        return return_dict

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
            # print(generated_answer)
            return "Yes"
        if "no" in judge2 and "yes" not in judge2:
            # print(generated_answer)
            return "No"
        # print(generated_answer)
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


def ESC(input_list, tokens, window_size=5):
    esc_count = 1
    esc_num = 0
    while True:
        if esc_num + window_size - 1 >= len(input_list):
            flag = 0
            break
        sub = input_list[esc_num]
        flag = 1
        for i in range(1, window_size):
            if sub != input_list[esc_num+i]:
                flag = 0
                break

        if flag == 1:
            esc_num += window_size
            break
        else:
            esc_count += 1
            esc_num += window_size

    if flag == 0:
        return input_list, tokens
    else:
        out = {}
        out["input"] = tokens["input"]
        out["output"] = tokens["output"][:esc_num]
        return input_list[:esc_num], out

def ASC(input_list, tokens, threshold=0.95):
    stop_judge = BetaStoppingCriteria(threshold)
    stop_position = len(input_list)
    for i in range(len(input_list)):
        judge_result = stop_judge.should_stop(input_list[:i+1])
        if judge_result["stop"]:
            stop_position = i+1
            break

    out = {}
    out["input"] = tokens["input"]
    out["output"] = tokens["output"][:stop_position]
    return input_list[:stop_position], out




def DSC_hard(input_list, tokens, easy_threshold=0.95, hard_threshold=0.50, window_size=5):
    easy_stop_judge = MyStoppingCriteria(easy_threshold, hard_threshold)
    stop_position = len(input_list)
    Flag = False
    for i in range(len(input_list)//window_size):
        judge_result = easy_stop_judge.should_stop(input_list[:(i+1)*window_size])
        if judge_result["easy_stop"]:
            Flag = True
            stop_position = (i+1)*window_size
            break
        if judge_result["hard_stop"] and i>=len(input_list)//(2*window_size):
            Flag = True
            stop_position = (i+1)*window_size
            break

    out = {}
    out["input"] = tokens["input"]
    if Flag:
        out["output"] = tokens["output"][:stop_position]
        return input_list[:stop_position], out
    else:
        out["output"] = tokens["output"]
        return input_list, out

def DSC(input_list, greedy_list, tokens, greedy_tokens, allocate, easy_threshold=0.95, hard_threshold=0.50, window_size=5):
    stop_position = len(input_list)
    Flag = False
    if allocate == 1:
        out = {}
        out["input"] = greedy_tokens["input"]
        out["output"] = greedy_tokens["output"]
        return greedy_list, out
    else:
        easy_stop_judge = MyStoppingCriteria(easy_threshold, 0)

        for i in range(len(input_list)//window_size):
            judge_result = easy_stop_judge.should_stop(input_list[:(i+1)*window_size])
            if judge_result["easy_stop"]:
                Flag = True
                stop_position = (i+1)*window_size
                break
            #if judge_result["hard_stop"] and i >= len(input_list)//(window_size*2):
            #    Flag = True
            #    stop_position = (i+1)*window_size
            #    break

        out = {}
        out["input"] = tokens["input"]
        if Flag:
            out["output"] = tokens["output"][:stop_position]
            return input_list[:stop_position], out
        else:
            out["output"] = tokens["output"]
            return input_list, out



def DSC_window(input_list, greedy_list, tokens, greedy_tokens, allocate, initial_window_size, extend_window_size=5):
    stop_position = len(input_list)
    Flag = False
    if allocate == 1:
        out = {}
        out["input"] = greedy_tokens["input"]
        out["output"] = greedy_tokens["output"]
        return greedy_list, out
    else:
        easy_stop_judge = MyStoppingCriteria(0.95, 0)
        count = 0
        while True:
            if initial_window_size + extend_window_size*count >= len(input_list):
                Flag = False
                break
            judge_result = easy_stop_judge.should_stop(input_list[:initial_window_size + extend_window_size*count])
            if judge_result["easy_stop"]:
                Flag = True
                stop_position = initial_window_size + extend_window_size*count
                break
            count += 1

        out = {}
        out["input"] = tokens["input"]
        if Flag:
            out["output"] = tokens["output"][:stop_position]
            return input_list[:stop_position], out
        else:
            out["output"] = tokens["output"]
            return input_list, out

def DSC_pre(input_list, greedy_list, tokens, greedy_tokens, allocate, initial_window_size, extend_window_size=5):
    stop_position = len(input_list)
    Flag = False

    easy_stop_judge = MyStoppingCriteria(0.95, 0)
    count = 0
    while True:
        if initial_window_size + extend_window_size*count >= len(input_list):
            Flag = False
            break
        judge_result = easy_stop_judge.should_stop(input_list[:initial_window_size + extend_window_size*count])
        if judge_result["easy_stop"]:
            Flag = True
            stop_position = initial_window_size + extend_window_size*count
            break
        count += 1

    out = {}
    out["input"] = tokens["input"]
    if Flag:
        out["output"] = tokens["output"][:stop_position]
        return input_list[:stop_position], out
    else:
        out["output"] = tokens["output"]
        return input_list, out


def get_true_window(easy_stop_judge, input_list, init_window_size=4):
    out = init_window_size
    for i in range(init_window_size, len(input_list)):
        judge_result = easy_stop_judge.should_stop(input_list[:i])
        if judge_result["easy_stop"]:
            out = i
            break
    return out
def get_GSM8K_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    task_prompts = [
        {
            "Q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "A": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
        },
        {
            "Q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "A": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
        },
        {
            "Q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "A": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
        },
        {
            "Q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "A": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
        },
        {
            "Q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "A": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
        },
        {
            "Q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "A": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
        },
        {
            "Q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "A": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
        },
        {
            "Q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "A": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
        }
    ]
    prompt = instruction
    for tem in task_prompts:
        prompt += tem["Q"] + "\n"
        prompt += "Let's think step by step. " + tem["A"] + "\n"

    return prompt


def get_MATH_prompt():
    instruction = "Think the question step by step and give the answer.\n"
    task_prompts = [
        {
            "Q": "Problem: Solve the inequality\n\\[\\frac{(x - 2)(x - 3)(x - 4)}{(x - 1)(x - 5)(x - 6)} > 0.\\]",
            "A": "Solution: We can build a sign chart, but since all of the factors are linear, we can track what happens to the expression as $x$ increases.  At $x = 0,$ the expression is positive.  As $x$ increases past 1, the expression becomes negative.  As $x$ increases past 2, the expression becomes positive, and so on.  Thus, the solution is\n\\[x \\in \\boxed{(-\\infty,1) \\cup (2,3) \\cup (4,5) \\cup (6,\\infty)}.\\]"
        },
        {
            "Q": "Problem: Compute: $55\\times1212-15\\times1212$ .",
            "A": "Solution: We have $55 \\times 1212 - 15 \\times 1212 = 1212(55-15) = 1212(40) = 4848(10) = \\boxed{48480}$."
        },
        {
            "Q": "Problem: A right circular cone has a volume of $12\\pi$ cubic centimeters. The height of the cone is 4 cm. How many centimeters is the circumference of the base of the cone, in terms of $\\pi$?",
            "A": "Solution: The volume of a cone is $\\frac{1}{3}\\pi r^2 h$. We are given that the volume is $12\\pi$ and the height is $4$. Thus, $\\frac{1}{3}\\pi r^2 \\cdot 4 = 12\\pi$. Solving for $r$, we find $r = 3$. Therefore, the circumference of the base is $2\\pi r = \\boxed{6\\pi}$."
        },
        {
            "Q": "Problem: Compute $\\dbinom{16}{15}$.",
            "A": "Solution: $\\dbinom{16}{15}=\\dbinom{16}{1}=\\boxed{16}."
        }
    ]
    prompt = instruction
    for tem in task_prompts:
        prompt += tem["Q"] + "\n\n"
        prompt += tem["A"] + "\n\n\n"

    return prompt

def get_coin_flip_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    task_prompts = [
        {
            "Q": "A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?",
            "A": "The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes."
        },
        {
            "Q": "A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up?",
            "A": "The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes."
        },
        {
            "Q": "A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?",
            "A": "The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no."
        },
        {
            "Q": "A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up?",
            "A": "The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no."
        },
    ]
    prompt = instruction
    for tem in task_prompts:
        prompt += "Q: " + tem["Q"] + "\n"
        prompt += "A: " + tem["A"] + "\n"
    
    return prompt


def get_last_letter_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    task_prompts = [
        {
            "Q": "Take the last letters of the words in \"Elon Musk\" and concatenate them.",
            "A": "The last letter of \"Elon\" is \"n\". The last letter of \"Musk\" is \"k\". Concatenating them is \"nk\". The answer is nk."
        },
        {
            "Q": "Take the last letters of the words in \"Larry Page\" and concatenate them.\n",
            "A": "The last letter of \"Larry\" is \"y\". The last letter of \"Page\" is \"e\". Concatenating them is \"ye\". The answer is ye."
        },
        {
            "Q": "Take the last letters of the words in \"Sergey Brin\" and concatenate them.\n",
            "A": "The last letter of \"Sergey\" is \"y\". The last letter of \"Brin\" is \"n\". Concatenating them is \"yn\". The answer is yn."
        },
        {
            "Q": "Take the last letters of the words in \"Bill Gates\" and concatenate them.",
            "A": "The last letter of \"Bill\" is \"l\". The last letter of \"Gates\" is \"s\". Concatenating them is \"ls\". The answer is ls."
        }
    ]
    prompt = instruction
    for tem in task_prompts:
        prompt += "Q: " + tem["Q"] + "\n"
        prompt += "A: " + tem["A"] + "\n"
    return prompt

def get_common_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    task_prompts = [
        {
            "Q": "What do people use to absorb extra ink from a fountain pen? Answer Choices: (a) shirt pocket (b) calligrapher's hand (c) inkwell (d) desk drawer (e) blotter",
            "A": "The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e)."
        },
        {
            "Q": "What home entertainment equipment requires cable? Answer Choices: (a) radio shack (b) substation (c) cabinet (d) television (e) desk",
            "A": "The answer must be something that uses cable for entertainment. Of the above choices, only television requires cable for entertainment. So the answer is (d)."
        },
        {
            "Q": "The fox walked from the city into the forest, what was it looking for? Answer Choices: (a) pretty flowers (b) hen house (c) natural habitat (d) storybook",
            "A": "The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (c)."
        },
        {
            "Q": "Sammy wanted to go to where the people were. Where might he go? Answer Choices: (a) populated areas (b) race track (c) desert (d) apartment (e) roadblock",
            "A": "The answer must be a place where people gather. Of the above choices, populated areas are where people are. So the answer is (a)."
        },
        {
            "Q": "Where do you put your grapes just before checking out? Answer Choices: (a) mouth (b) grocery cart (c)super market (d) fruit basket (e) fruit market",
            "A": "The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b)."
        },
        {
            "Q": "Google Maps and other highway and street GPS services have replaced what? Answer Choices: (a) united states (b) mexico (c) countryside (d) atlas (e) oceans",
            "A": "The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d)."
        },
        {
            "Q": "Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: (a) harder (b) anguish (c) bitterness (d) tears (e) sadness",
            "A": "The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c)."
        },
        {
            "Q": "You can share files with someone if you have a connection to a what? Answer Choices: (a) freeway (b) radio (c) wires (d) computer network (e) electrical circuit",
            "A": "The answer must be something that allows for file sharing. Of the above choices, only a computer network allows for this. So the answer is (d)."
        }
    ]
    prompt = instruction
    for tem in task_prompts:
        prompt += "Q: " + tem["Q"] + "\n"
        prompt += "A: " + tem["A"] + "\n"
    return prompt

def get_strategy_prompt():
    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
    task_prompts = [
        {
            "Q": "Yes or no: Do hamsters provide food for any animals?",
            "A": "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes."
        },
        {
            "Q": "Yes or no: Could Brooke Shields succeed at University of Pennsylvania?",
            "A": "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes."
        },
        {
            "Q": "Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?",
            "A": "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. So the answer is no."
        },
        {
            "Q": "Yes or no: Is it common to see frost during some college commencements?",
            "A": "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes."
        },
        {
            "Q": "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
            "A": "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no."
        },
        {
            "Q": "Yes or no: Would a pear sink in water?",
            "A": "The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no."
        },
        {
            "Q": "Yes or no: Can a Goblin shark hypothetically ride a bike if it had limbs?",
            "A": "A Goblin shark weighs around 460 pounds. The weight capacity of the average bike is 300 pounds, which is less than the weight of Goblin shar. Thus, Goblin shark can not ride a bike if it had limbs. So the answer is no."
        },
        {
            "Q": "Yes or no: Would the chef at Carmine's restaurant panic if there was no basil?",
            "A": "Carmines is an Italian family-style restaurant. Basil is an essential in Italian cooking. Thus, the chef at Carmine's restaurant would panic if there was no basil. So the answer is yes."
        }
    ]
    prompt = instruction
    for tem in task_prompts:
        prompt += "Q: " + tem["Q"] + "\n"
        prompt += "A: " + tem["A"] + "\n"
    return prompt
def cal_tokens(input, tokenizer):
    tokens = tokenizer.tokenize(input)
    return len(tokens)

def cal_sc_tokens(input, output_list, tokenizer):
    input_token_num = cal_tokens(input, tokenizer)
    output_token_num = sum([cal_tokens(output_list[i], tokenizer) for i in range(len(output_list))])

    return input_token_num, output_token_num


def cal_esc_tokens(input, output_list, tokenizer, window_size=5):
    input_token_num = cal_tokens(input, tokenizer)*len(output_list)//window_size
    output_token_num = sum([cal_tokens(output_list[i], tokenizer) for i in range(len(output_list))])
    return input_token_num, output_token_num


def cal_asc_tokens(input, output_list, tokenizer):
    input_token_num = cal_tokens(input, tokenizer)*len(output_list)
    output_token_num = sum([cal_tokens(output_list[i], tokenizer) for i in range(len(output_list))])
    return input_token_num, output_token_num

def random_elements(A, input_tokens, num):
    indices = random.sample(range(len(A)), num)
    selected_A = [A[i] for i in indices]
    out = {}
    out["input"] = input_tokens["input"]
    out["output"] = [input_tokens["output"][i] for i in indices]
    return selected_A, out

def get_dollars(input_token, output_token, gpt_version):
    if gpt_version=="3.5":
        price = 0.5*input_token/1000000 + 1.5*output_token/1000000
    else:
        price = 30*input_token/1000000 + 60*output_token/1000000

    return price

def dsc_allocate(pre_list, judge_window_size, sample_window_size, max_sample_size): # input difficulty ordered list
    length = len(pre_list)
    allocate = [0 for i in range(length)]
    easy_sample_flag = [False for i in range(length)]
    hard_sample_flag = [False for i in range(length)]
    left = 0
    right = length
    while True:
        if right <= left:
            break
        idx = (right+left)//2
        judge_left = max(0, idx - judge_window_size//2)
        judge_right = min(idx + judge_window_size//2, length)

        for j in range(judge_left, judge_right):
            random_choice = random.sample(pre_list[j], max_sample_size)
            sub_allocate = DSC_judge(random_choice, max_sample_size=max_sample_size, easy_threshold=0.95, hard_threshold=0, window_size=sample_window_size)  # we do not set hard here in case of mixing easy and hard
            if sub_allocate == sample_window_size:
                easy_sample_flag[j] = True

        if all(tem == True for tem in easy_sample_flag[judge_left:judge_right]):
            for j in range(left, idx+1):
                allocate[j] = 1
            left = idx+1
        else:
            right = idx

    left = 0
    right = length
    while True:
        if right <= left:
            break
        idx = (right + left) // 2
        judge_left = max(0, idx - judge_window_size // 2)
        judge_right = min(idx + judge_window_size // 2, length)

        for j in range(judge_left, judge_right):
            sub_allocate = DSC_judge(pre_list[j][:max_sample_size], max_sample_size=max_sample_size, easy_threshold=0.95, hard_threshold=0)  # we do not set hard here in case of mixing easy and hard
            if sub_allocate >= (max_sample_size//sample_window_size)*sample_window_size/2:
                hard_sample_flag[j] = True

        if sum(hard_sample_flag[judge_left:judge_right]) >= judge_window_size:
            for j in range(idx, right):
                allocate[j] = max_sample_size
            right = idx
        else:
            left = idx+1


    for i in range(len(allocate)):
        if allocate[i] == 0:
            allocate[i] = -1

    return allocate


def dsc_allocate_greedy(pre_list, judge_window_size, sample_window_size, max_sample_size): # input difficulty ordered list
    length = len(pre_list)
    allocate = [0 for i in range(length)]
    easy_sample_flag = [False for i in range(length)]
    hard_sample_flag = [False for i in range(length)]
    left = 0
    right = length
    Flag = False
    for j in range(length):
        random_choice = random.sample(pre_list[length-j-1], max_sample_size)
        sub_allocate = DSC_judge(random_choice, max_sample_size=max_sample_size, easy_threshold=0.95, hard_threshold=0,
                             window_size=sample_window_size)
        if sub_allocate == sample_window_size:
            easy_sample_flag[length-j-1] = True

        if j >= judge_window_size-1:
            if all(tem == True for tem in easy_sample_flag[length-j-1:length-j-1+judge_window_size]):
                Flag = True
                for i in range(length-j-1+judge_window_size):
                    allocate[i] = 1
                break


    for i in range(len(allocate)):
        if allocate[i] == 0:
            allocate[i] = -1

    return allocate

def DSC_judge(input_list, max_sample_size, easy_threshold=0.95, hard_threshold=0.50, window_size=5):
    easy_stop_judge = MyStoppingCriteria(easy_threshold, hard_threshold)
    Flag = False
    for i in range(max_sample_size // window_size):
        judge_result = easy_stop_judge.should_stop(input_list[:(i+1)*window_size])
        if (i+1)*window_size >= max_sample_size:
            break
        if judge_result["easy_stop"]:
            Flag = True
            allocate = (i+1)*window_size
            break
        if judge_result["hard_stop"]:
            Flag = True
            allocate = (i+1)*window_size
            break
    if Flag:
        return allocate
    else:
        return max_sample_size

def split_easy_hard(eval_result, pre_batch, judge_window_size, window_size, max_sample_size):
    question = eval_result["questions"]
    eval = eval_result["eval"]
    index = range(len(question))
    zipped = sorted(zip(eval, question, pre_batch, index), reverse=False)
    eval, question, pre_batch, index = zip(*zipped)
    allocate_list = dsc_allocate(pre_batch, judge_window_size, window_size, max_sample_size)
    zipped = sorted(zip(index, eval, question, pre_batch, allocate_list), reverse=False)
    index, eval, question, pre_batch, allocate_list = zip(*zipped)
    print("easy count={}".format(sum([sub == 1 for sub in allocate_list])))
    print("hard count={}".format(sum([sub == n for sub in allocate_list])))
    return index, eval, question, pre_batch, allocate_list

def split_easy_hard_greedy(eval_result, pre_batch, judge_window_size, window_size, max_sample_size):
    question = eval_result["questions"]
    eval = eval_result["eval"]
    index = range(len(question))
    zipped = sorted(zip(eval, question, pre_batch, index), reverse=False)
    eval, question, pre_batch, index = zip(*zipped)
    allocate_list = dsc_allocate_greedy(pre_batch, judge_window_size, window_size, max_sample_size)
    zipped = sorted(zip(index, eval, question, pre_batch, allocate_list), reverse=False)
    index, eval, question, pre_batch, allocate_list = zip(*zipped)
    print("greedy easy count={}".format(sum([sub == 1 for sub in allocate_list])))
    print("greedy hard count={}".format(sum([sub == n for sub in allocate_list])))
    return index, eval, question, pre_batch, allocate_list


def get_min_allocate_size(easy_stop_judge, input, max_sample_size):
    out = max_sample_size
    for i in range(max_sample_size):
        judge_result = easy_stop_judge.should_stop(input[:i+1])
        if judge_result["easy_stop"]:
            Flag = True
            out = i+1
            break
    return out


def allocate_run(pre_list, max_sample_size, interplot_size, window_size=4): # input difficulty ordered list
    length = len(pre_list)
    actual_window = [0 for i in range(length)]
    true_window = [0 for i in range(length)]

    easy_stop_judge = MyStoppingCriteria(0.95, 0)
    for j in range(length):
        if j <= interplot_size-1:
            current_window_size = window_size
        else:
            #current_window_size = min(true_window[max(0, j - interplot_size): j]) // 4 * 4
            current_window_size = round(sum(true_window[max(0, j-interplot_size): j]) / len(true_window[max(0, j-interplot_size): j])/4) * 4
        random_choice = random.sample(pre_list[j], max_sample_size)

        actual_window[j] = current_window_size
        true_window[j] = get_min_allocate_size(easy_stop_judge, random_choice, max_sample_size)
        if true_window[j] == 4:
            actual_window[j] = 4
    # windows = {'actual_window': actual_window, 'true_window': true_window}
    # with open("window.json", "w") as f:
    #     json.dump(windows, f)
    #     f.close()

    return actual_window


def allocate_window(eval_result, pre_batch, max_sample_size, interplot_size=5, window_size=4): # using greedy judge
    question = eval_result["questions"]
    eval = eval_result["eval"]
    index = range(len(question))
    zipped = sorted(zip(eval, question, pre_batch, index), reverse=False) # easy to hard
    eval, question, pre_batch, index = zip(*zipped)

    actual_window= allocate_run(pre_batch, max_sample_size, interplot_size, window_size)
    zipped = sorted(zip(index, eval, question, pre_batch, actual_window), reverse=False)
    index, eval, question, pre_batch, actual_window = zip(*zipped)

    return actual_window

def clean_data(pre_batch, eval_result, pre, pre_answer, pre_tokens, greedy_pre_tokens, greedy_pre_batch, sc_num):
    cleaned_data = []
    cleaned_eval_result = {}
    cleaned_pre, cleaned_pre_answer, cleaned_pre_tokens, cleaned_greedy_pre_tokens, cleaned_greedy_pre_batch = [], [], [], [], []
    for key in eval_result.keys():
        cleaned_eval_result[key] = []

    for i in range(len(pre_batch)):
        if len(pre_batch[i]) < sc_num:
            continue
        else:
            cleaned_data.append(deepcopy(pre_batch[i]))
            for key in eval_result.keys():
                cleaned_eval_result[key].append(deepcopy(eval_result[key][i]))

            cleaned_pre.append(deepcopy(pre[i]))
            cleaned_pre_answer.append(deepcopy(pre_answer[i]))
            cleaned_pre_tokens.append(deepcopy(pre_tokens[i]))
            cleaned_greedy_pre_batch.append(deepcopy(greedy_pre_batch[i]))
            cleaned_greedy_pre_tokens.append(deepcopy(greedy_pre_tokens[i]))

    return cleaned_data, cleaned_eval_result, cleaned_pre, cleaned_pre_answer, cleaned_pre_tokens, cleaned_greedy_pre_tokens, cleaned_greedy_pre_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="result/coin_flip_gpt_4_samples_n_50_temp_0.7.jsonl")
    parser.add_argument("--dsc_input_path", type=str, default="result/coin_flip_gpt_4_samples_n_1_temp_0.jsonl")
    parser.add_argument("--eval_path", type=str, default="result/coin_flip_gpt_4_eval_strategy_random_prompt_short_batchsize_8_iter_5_n_1_temp_0.jsonl")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    input_path = args.input_path
    dsc_input_path = args.dsc_input_path
    eval_path = args.eval_path

    n = args.n
    repeat = args.repeat
    dataset = input_path.split("_gpt_")[0].split("/")[-1]
    # dataset = ["MATH", "GSM8K"]
    # dataset = dataset[1]
    interplot_size = 16
    window_size = 4
    judge_window_size = 32
    gpt_version = input_path.split("gpt_")[-1].split("_")[0]

    output_path = "cmp/{}_n_{}_judge_window_{}_interplot_{}.json".format(input_path.split("/")[-1].split(".jsonl")[0], n, judge_window_size, interplot_size)
    print(output_path)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            result = json.load(f)
            f.close()
        print(result)
    else:
        model, tokenizer = load_model(
            "Mistral-7B-v0.1",
            device="cuda",
            num_gpus=1,
            cuda_load=False
        )

        sc_accuracy_list = []
        esc_accuracy_list = []
        asc_accuracy_list = []
        easc_accuracy_list = []
        dsc_hard_accuracy_list = []
        dsc_accuracy_list = []
        dsc_greedy_accuracy_list = []
        dsc_window_accuracy_list = []
        dsc_pre_accuracy_list = []

        tokens_count = {"sc": {"input": [], "output": []}, "esc": {"input": [], "output": []}, "asc": {"input": [], "output": []},
                        "dsc_hard": {"input": [], "output": []}, "dsc": {"input": [], "output": []},
			            "dsc_greedy": {"input": [], "output": []}, "dsc_window": {"input": [], "output": []},  "dsc_pre": {"input": [], "output": []}}

        with open(input_path, "r") as f:
            data = f.readlines()
            f.close()

        with open(dsc_input_path, "r") as f:
            greedy_data = f.readlines()
            f.close()


        all_dict = {}

        #type_all = []
        eval_result = []
        pre = []
        pre_answer = []
        pre_batch = []
        pre_tokens = []

        greedy_pre = []
        greedy_pre_answer = []
        greedy_pre_batch = []
        greedy_pre_tokens = []

        count = 0
        unmatch_count = 0
        greedy_unmatch_count = 0

        if dataset == "MATH":
            prompt = get_MATH_prompt()
        elif dataset == "GSM8K":
            prompt = get_GSM8K_prompt()
        elif dataset == "coin_flip":
            prompt = get_coin_flip_prompt()
        elif dataset == "last_letter":
            prompt = get_last_letter_prompt()
        elif dataset == "common":
            prompt = get_common_prompt()
        else:
            prompt = get_strategy_prompt()

        for line in data:
            tem = json.loads(line)
            pre.append(tem)
            if dataset == "MATH":
                pre_answer.append(tem['answer'])
                #type_all.append(tem["type"])
            elif dataset == "GSM8K":
                pre_answer.append(extract_answer(tem['answer']))
            else:
                pre_answer.append(tem['answer'])

            shu_batch = tem['completion']
            cur_input_token = cal_tokens(prompt + tem["input"], tokenizer)
            cur_output_token = [cal_tokens(sub, tokenizer) for sub in shu_batch]
            pre_tokens.append({"input": cur_input_token, "output": deepcopy(cur_output_token)})

            if dataset == "MATH":
                predict_batch = [extract_math_answer(item) for item in shu_batch]
            elif dataset == "GSM8K":
                predict_batch = [extract_gsm8k_answer(item) for item in shu_batch]
            elif dataset == "coin_flip":
                predict_batch = [extract_coin_flip_answer(item) for item in shu_batch]
            elif dataset == "last_letter":
                predict_batch = [extract_last_letter_answer(item) for item in shu_batch]
            elif dataset == "common":
                predict_batch = [extract_common_answer(item) for item in shu_batch]
            else:
                predict_batch = [extract_strategy_answer(item) for item in shu_batch]
            for item in predict_batch:
                if item == '':
                    unmatch_count += 1
            pre_batch.append(deepcopy(predict_batch))

        print("unmatch count={}".format(unmatch_count))

        for line in greedy_data:
            tem = json.loads(line)
            greedy_pre.append(tem)
            if dataset == "MATH":
                greedy_pre_answer.append(tem['answer'])
            elif dataset == "GSM8K":
                greedy_pre_answer.append(extract_answer(tem['answer']))
            else:
                greedy_pre_answer.append(tem['answer'])

            shu_batch = tem['completion']
            cur_input_token = cal_tokens(prompt + tem["input"], tokenizer)
            cur_output_token = [cal_tokens(sub, tokenizer) for sub in shu_batch]
            greedy_pre_tokens.append({"input": cur_input_token, "output": deepcopy(cur_output_token)})

            if dataset == "MATH":
                predict_batch = [extract_math_answer(item) for item in shu_batch]
            elif dataset == "GSM8K":
                predict_batch = [extract_gsm8k_answer(item) for item in shu_batch]
            elif dataset == "coin_flip":
                predict_batch = [extract_coin_flip_answer(item) for item in shu_batch]
            elif dataset == "last_letter":
                predict_batch = [extract_last_letter_answer(item) for item in shu_batch]
            elif dataset == "common":
                predict_batch = [extract_common_answer(item) for item in shu_batch]
            else:
                predict_batch = [extract_strategy_answer(item) for item in shu_batch]

            for item in predict_batch:
                if item == '':
                    greedy_unmatch_count += 1
            greedy_pre_batch.append(deepcopy(predict_batch))

        print("greedy unmatch count={}".format(greedy_unmatch_count))

        with open(eval_path, "r") as f:
            for line in f:
                eval_result.append(json.loads(line))
            f.close()
        # eval allocate

        if dataset != "MATH":
            eval_result = eval_result[0]
        else:
            total_length = len(eval_result)
            eval_result_math = {} #{"questions":[], "eval":[], "subject":[]}
            for i in range(len(eval_result)):
                item = eval_result[i]
                if item["subject"] not in eval_result_math.keys():
                    eval_result_math[item["subject"]] = {"questions":[], "eval":[], "pre_batch":[], "index":[]}
                eval_result_math[item["subject"]]["questions"].append(item["problem"])
                eval_result_math[item["subject"]]["eval"].append(item["eval"])
                eval_result_math[item["subject"]]["pre_batch"].append(deepcopy(pre_batch[i]))
                eval_result_math[item["subject"]]["index"].append(i)
                # eval_result_math[item["subject"]]["subject"].append(item["subject"])
            eval_result = eval_result_math

        # question = eval_result["questions"]
        # eval = eval_result["eval"]
        # index = range(len(question))
        # zipped = sorted(zip(eval, question, pre_batch, index), reverse=False)
        # eval, question, pre_batch, index = zip(*zipped)
        # allocate_list = dsc_allocate(pre_batch, judge_window_size, window_size, max_sample_size=n)
        # zipped = sorted(zip(index, eval, question, pre_batch, allocate_list), reverse=False)
        # index, eval, question, pre_batch, allocate_list = zip(*zipped)
        # print("easy count={}".format(sum([sub == 1 for sub in allocate_list])))
        # print("hard count={}".format(sum([sub == n for sub in allocate_list])))
        if dataset != "MATH":
            pre_batch, eval_result, pre, pre_answer, pre_tokens, greedy_pre_tokens, greedy_pre_batch = clean_data(pre_batch,
                                                                                                              eval_result,
                                                                                                              pre,
                                                                                                              pre_answer,
                                                                                                              pre_tokens,
                                                                                                              greedy_pre_tokens,
                                                                                                              greedy_pre_batch,
                                                                                                              n)
        for repeat_num in tqdm(range(repeat)):

            if dataset != "MATH":
                index, eval, question, pre_batch, allocate_list = split_easy_hard(eval_result, pre_batch, judge_window_size, window_size, n)
                index_greedy, eval_greedy, question_greedy, pre_batch_greedy, allocate_list_greedy = split_easy_hard_greedy(eval_result, pre_batch, judge_window_size, window_size, n)
                window_list_actual = allocate_window(eval_result, pre_batch, n, interplot_size=interplot_size, window_size=window_size)
            else:
                allocate_list = [0 for i in range(total_length)]
                allocate_list_greedy = [0 for i in range(total_length)]
                window_list_actual = [0 for i in range(total_length)]
                for key in eval_result.keys():
                    print(key)
                    sub_eval_result = eval_result[key]
                    sub_index = sub_eval_result["index"]
                    index, sub_eval, sub_question, sub_pre_batch, sub_allocate_list = split_easy_hard(sub_eval_result,
                                                                                                      sub_eval_result[
                                                                                                          "pre_batch"],
                                                                                                      judge_window_size,
                                                                                                      window_size, n)

                    index_greedy, sub_eval_greedy, sub_question_greedy, sub_pre_batch_greedy, sub_allocate_list_greedy = split_easy_hard_greedy(sub_eval_result,
                                                                                                      sub_eval_result[
                                                                                                          "pre_batch"],
                                                                                                      judge_window_size,
                                                                                                      window_size, n)
                    actual_window = allocate_window(sub_eval_result, sub_eval_result["pre_batch"], n,
                                                                                  interplot_size=interplot_size)
                    for i in range(len(sub_allocate_list)):
                        allocate_list[sub_index[i]] = sub_allocate_list[i]
                        allocate_list_greedy[sub_index[i]] = sub_allocate_list_greedy[i]
                        window_list_actual[sub_index[i]] = actual_window[i]


            for sc_num in [n]:
                sc_result_list = []
                esc_result_list = []
                asc_result_list = []
                dsc_hard_result_list = []
                dsc_result_list = []
                dsc_greedy_result_list = []
                dsc_window_result_list = []
                dsc_pre_result_list = []

                sub_tokens_count = {"sc": {"input": [], "output": []}, "esc": {"input": [], "output": []}, "asc": {"input": [], "output": []},
                                    "dsc_hard": {"input": [], "output": []}, "dsc": {"input": [], "output": []},
                                    "dsc_greedy": {"input": [], "output": []}, "dsc_window": {"input": [], "output": []}, "dsc_pre": {"input": [], "output": []}}

                num_list = []
                true_window_list = []
                right_nums = 0
                n_nums = 0
                count = 0

                for i in range(len(pre)):
                    flag = False
                    pre_dict = {}
                    esc_pre_dict = {}
                    asc_pre_dict = {}
                    dsc_hard_pre_dict = {}
                    dsc_pre_dict = {}
                    dsc_greedy_pre_dict = {}
                    dsc_window_pre_dict = {}
                    dsc_pre_pre_dict = {}

                    tem = pre[i]
                    question = tem["input"]
                    answer = pre_answer[i]
                    tokens_batch = pre_tokens[i]
                    greedy_tokens_batch = greedy_pre_tokens[i]
                    sub_allocate = allocate_list[i]
                    sub_allocate_greedy = allocate_list_greedy[i]

                    sub_window_actual = window_list_actual[i]

                    # print("sub_window_actual:{}".format(sub_window_actual))

                    shu_batch = tem['completion']

                    predict_batch = pre_batch[i]
                    greedy_predict_batch = greedy_pre_batch[i]
                    # print(len(predict_batch))
                    if len(predict_batch) < sc_num:
                        count += 1
                        continue


                    predict_batch, tokens_batch = random_elements(predict_batch, tokens_batch, sc_num)
                    # predict_batch = random.sample(predict_batch, sc_num)
                    esc_batch, esc_token_batch = ESC(predict_batch, tokens_batch, window_size=5)
                    asc_batch, asc_token_batch = ASC(predict_batch, tokens_batch)
                    dsc_hard_batch, dsc_hard_token_batch = DSC_hard(predict_batch, tokens_batch, window_size=window_size)
                    dsc_batch, dsc_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate, window_size=window_size)
                    dsc_greedy_batch, dsc_greedy_token_batch = DSC(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, window_size=window_size)
                    dsc_window_batch, dsc_window_token_batch = DSC_window(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)
                    dsc_pre_batch, dsc_pre_token_batch = DSC_pre(predict_batch, greedy_predict_batch, tokens_batch, greedy_tokens_batch, sub_allocate_greedy, sub_window_actual, extend_window_size=window_size)

                    sc_input_token, sc_output_token = tokens_batch["input"], sum(tokens_batch["output"])
                    esc_input_token, esc_output_token = tokens_batch["input"]*len(esc_token_batch["output"]) // window_size, sum(esc_token_batch["output"])
                    asc_input_token, asc_output_token = tokens_batch["input"]*len(asc_token_batch["output"]), sum(asc_token_batch["output"])
                    dsc_hard_input_token, dsc_hard_output_token = tokens_batch["input"] * len(dsc_hard_token_batch["output"]) // window_size, sum(dsc_hard_token_batch["output"])
                    dsc_input_token, dsc_output_token = tokens_batch["input"] * max(1, len(dsc_token_batch["output"]) // window_size), sum(dsc_token_batch["output"])
                    dsc_greedy_input_token, dsc_greedy_output_token = tokens_batch["input"] * max(1, len(dsc_greedy_token_batch["output"]) // window_size), sum(dsc_greedy_token_batch["output"])
                    dsc_window_input_token, dsc_window_output_token = tokens_batch["input"] * max(1, len(dsc_window_token_batch["output"]) // sub_window_actual), sum(dsc_window_token_batch["output"])
                    dsc_pre_input_token, dsc_pre_output_token = tokens_batch["input"] * max(1,len(dsc_pre_token_batch["output"]) // sub_window_actual), sum(dsc_pre_token_batch["output"])

                    # sc_input_token, sc_output_token = cal_sc_tokens(prompt + question, completion_batch, tokenizer)
                    # esc_input_token, esc_output_token = cal_esc_tokens(prompt + question, esc_completion_batch, tokenizer)
                    # asc_input_token, asc_output_token = cal_asc_tokens(prompt + question, asc_completion_batch, tokenizer)

                    sub_tokens_count["sc"]["input"].append(sc_input_token)
                    sub_tokens_count["sc"]["output"].append(sc_output_token)
                    sub_tokens_count["esc"]["input"].append(esc_input_token)
                    sub_tokens_count["esc"]["output"].append(esc_output_token)
                    sub_tokens_count["asc"]["input"].append(asc_input_token)
                    sub_tokens_count["asc"]["output"].append(asc_output_token)
                    sub_tokens_count["dsc_hard"]["input"].append(dsc_hard_input_token)
                    sub_tokens_count["dsc_hard"]["output"].append(dsc_hard_output_token)
                    sub_tokens_count["dsc"]["input"].append(dsc_input_token)
                    sub_tokens_count["dsc"]["output"].append(dsc_output_token)
                    sub_tokens_count["dsc_greedy"]["input"].append(dsc_greedy_input_token)
                    sub_tokens_count["dsc_greedy"]["output"].append(dsc_greedy_output_token)
                    sub_tokens_count["dsc_window"]["input"].append(dsc_window_input_token)
                    sub_tokens_count["dsc_window"]["output"].append(dsc_window_output_token)
                    sub_tokens_count["dsc_pre"]["input"].append(dsc_pre_input_token)
                    sub_tokens_count["dsc_pre"]["output"].append(dsc_pre_output_token)
                    for item in predict_batch:
                        if item in pre_dict.keys():
                            pre_dict[item] += 1
                        else:
                            pre_dict[item] = 1

                    for item in esc_batch:
                        if item in esc_pre_dict.keys():
                            esc_pre_dict[item] += 1
                        else:
                            esc_pre_dict[item] = 1

                    for item in asc_batch:
                        if item in asc_pre_dict.keys():
                            asc_pre_dict[item] += 1
                        else:
                            asc_pre_dict[item] = 1


                    for item in dsc_hard_batch:
                        if item in dsc_hard_pre_dict.keys():
                            dsc_hard_pre_dict[item] += 1
                        else:
                            dsc_hard_pre_dict[item] = 1

                    for item in dsc_batch:
                        if item in dsc_pre_dict.keys():
                            dsc_pre_dict[item] += 1
                        else:
                            dsc_pre_dict[item] = 1

                    for item in dsc_greedy_batch:
                        if item in dsc_greedy_pre_dict.keys():
                            dsc_greedy_pre_dict[item] += 1
                        else:
                            dsc_greedy_pre_dict[item] = 1

                    for item in dsc_window_batch:
                        if item in dsc_window_pre_dict.keys():
                            dsc_window_pre_dict[item] += 1
                        else:
                            dsc_window_pre_dict[item] = 1

                    for item in dsc_pre_batch:
                        if item in dsc_pre_pre_dict.keys():
                            dsc_pre_pre_dict[item] += 1
                        else:
                            dsc_pre_pre_dict[item] = 1

                    sc_pre = max(pre_dict, key=pre_dict.get)
                    sc_result_list.append(sc_pre == answer)

                    esc_pre = max(esc_pre_dict, key=esc_pre_dict.get)
                    esc_result_list.append(esc_pre == answer)

                    asc_pre = max(asc_pre_dict, key=asc_pre_dict.get)
                    asc_result_list.append(asc_pre == answer)


                    dsc_hard_pre = max(dsc_hard_pre_dict, key=dsc_hard_pre_dict.get)
                    dsc_hard_result_list.append(dsc_hard_pre == answer)

                    dsc_pre = max(dsc_pre_dict, key=dsc_pre_dict.get)
                    dsc_result_list.append(dsc_pre == answer)

                    dsc_greedy_pre = max(dsc_greedy_pre_dict, key=dsc_greedy_pre_dict.get)
                    dsc_greedy_result_list.append(dsc_greedy_pre == answer)

                    dsc_window_pre = max(dsc_window_pre_dict, key=dsc_window_pre_dict.get)
                    dsc_window_result_list.append(dsc_window_pre == answer)

                    dsc_pre_pre = max(dsc_pre_pre_dict, key=dsc_pre_pre_dict.get)
                    dsc_pre_result_list.append(dsc_pre_pre == answer)
                    # result_list_level[hard_level - 1].append(sc_pre == answer)
                    # result_list_subject[subject].append(sc_pre == answer)
                    # result_list_detail[subject][hard_level - 1].append(sc_pre == answer)

                sc_accuracy_list.append(np.array(sc_result_list).mean())
                esc_accuracy_list.append(np.array(esc_result_list).mean())
                asc_accuracy_list.append(np.array(asc_result_list).mean())
                dsc_hard_accuracy_list.append(np.array(dsc_hard_result_list).mean())
                dsc_accuracy_list.append(np.array(dsc_result_list).mean())
                dsc_greedy_accuracy_list.append(np.array(dsc_greedy_result_list).mean())
                dsc_window_accuracy_list.append(np.array(dsc_window_result_list).mean())
                dsc_pre_accuracy_list.append(np.array(dsc_pre_result_list).mean())

                for key in tokens_count.keys():
                    tokens_count[key]["input"].append(np.array(sub_tokens_count[key]["input"]).mean())
                    tokens_count[key]["output"].append(np.array(sub_tokens_count[key]["output"]).mean())


        # print(accuracy_list)
        sc_input_token, sc_output_token = np.array(tokens_count["sc"]["input"]).mean(), np.array(tokens_count["sc"]["output"]).mean()
        esc_input_token, esc_output_token = np.array(tokens_count["esc"]["input"]).mean(), np.array(tokens_count["esc"]["output"]).mean()
        asc_input_token, asc_output_token = np.array(tokens_count["asc"]["input"]).mean(), np.array(tokens_count["asc"]["output"]).mean()
        dsc_hard_input_token, dsc_hard_output_token = np.array(tokens_count["dsc_hard"]["input"]).mean(), np.array(tokens_count["dsc_hard"]["output"]).mean()
        dsc_input_token, dsc_output_token = np.array(tokens_count["dsc"]["input"]).mean(), np.array(tokens_count["dsc"]["output"]).mean()
        dsc_greedy_input_token, dsc_greedy_output_token = np.array(tokens_count["dsc_greedy"]["input"]).mean(), np.array(tokens_count["dsc_greedy"]["output"]).mean()
        dsc_window_input_token, dsc_window_output_token = np.array(tokens_count["dsc_window"]["input"]).mean(), np.array(tokens_count["dsc_window"]["output"]).mean()
        dsc_pre_input_token, dsc_pre_output_token = np.array(tokens_count["dsc_pre"]["input"]).mean(), np.array(tokens_count["dsc_pre"]["output"]).mean()


        sc_price = get_dollars(sc_input_token, sc_output_token, gpt_version)
        esc_price = get_dollars(esc_input_token, esc_output_token, gpt_version)
        asc_price = get_dollars(asc_input_token, asc_output_token, gpt_version)
        dsc_hard_price = get_dollars(dsc_hard_input_token, dsc_hard_output_token, gpt_version)
        dsc_price = get_dollars(dsc_input_token, dsc_output_token, gpt_version)
        dsc_greedy_price = get_dollars(dsc_greedy_input_token, dsc_greedy_output_token, gpt_version)
        dsc_window_price = get_dollars(dsc_window_input_token, dsc_window_output_token, gpt_version)
        dsc_pre_price = get_dollars(dsc_pre_input_token, dsc_pre_output_token, gpt_version)

        print("SC mean_acc = {} input_token={} output_token={} price={}$".format(np.array(sc_accuracy_list).mean(), sc_input_token, sc_output_token, sc_price))
        print("ESC mean_acc = {} input_token={} output_token={} price={}$".format(np.array(esc_accuracy_list).mean(), esc_input_token, esc_output_token, esc_price))
        print("ASC mean_acc = {} input_token={} output_token={} price={}$".format(np.array(asc_accuracy_list).mean(), asc_input_token, asc_output_token, asc_price))
        print("DSC_hard mean_acc = {} input_token={} output_token={} price={}$".format(np.array(dsc_hard_accuracy_list).mean(), dsc_hard_input_token, dsc_hard_output_token, dsc_hard_price))
        print("DSC mean_acc = {} input_token={} output_token={} price={}$".format(np.array(dsc_accuracy_list).mean(), dsc_input_token, dsc_output_token, dsc_price))
        print("DSC_greedy mean_acc = {} input_token={} output_token={} price={}$".format(np.array(dsc_greedy_accuracy_list).mean(), dsc_greedy_input_token, dsc_greedy_output_token, dsc_greedy_price))
        print("DSC_window mean_acc = {} input_token={} output_token={} price={}$".format(np.array(dsc_window_accuracy_list).mean(), dsc_window_input_token, dsc_window_output_token, dsc_window_price))
        print("DSC_pre mean_acc = {} input_token={} output_token={} price={}$".format(np.array(dsc_pre_accuracy_list).mean(), dsc_pre_input_token, dsc_pre_output_token, dsc_pre_price))

        result_dict = {
                       "sc": {"acc": np.array(sc_accuracy_list).mean(), "input": sc_input_token, "output": sc_output_token, "price": sc_price},
                       "esc": {"acc": np.array(esc_accuracy_list).mean(), "input": esc_input_token, "output": esc_output_token, "price": esc_price},
                       "asc": {"acc": np.array(asc_accuracy_list).mean(), "input": asc_input_token, "output": asc_output_token, "price": asc_price},
                       "dsc_hard": {"acc": np.array(dsc_hard_accuracy_list).mean(), "input": dsc_hard_input_token, "output": dsc_hard_output_token, "price": dsc_hard_price},
                       "dsc": {"acc": np.array(dsc_accuracy_list).mean(), "input": dsc_input_token, "output": dsc_output_token, "price": dsc_price},
                       "dsc_greedy": {"acc": np.array(dsc_greedy_accuracy_list).mean(), "input": dsc_greedy_input_token, "output": dsc_greedy_output_token, "price": dsc_greedy_price},
                       "dsc_window": {"acc": np.array(dsc_window_accuracy_list).mean(), "input": dsc_window_input_token, "output": dsc_window_output_token, "price": dsc_window_price},
                       "dsc_pre": {"acc": np.array(dsc_pre_accuracy_list).mean(), "input": dsc_pre_input_token, "output": dsc_pre_output_token, "price": dsc_pre_price}
                    }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent="\t")
            f.close()
        print("done !")
