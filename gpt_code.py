import json
import time

import openai
from typing import Dict, Iterable
from tqdm import tqdm
import os
from os import PathLike
import gzip
from copy import deepcopy

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
    def calc(self,query,instruction="", temp=0,n=1,model=3.5,task="code", few_shot=True, onestep=False, one_shot=False):
            openai.api_type = ""
            openai.api_base = ""
            openai.api_version =""

            if task == "human_eval" or task == "mbpp":
                instruction = "Please complete the given function with Python code according to the description."
                cur = []
            elif task == "summ_screen":
                instruction = "Please write a short, one-sentence summary of the following TV show episode:"
                cur = []
            elif task == "MATH" or task == "MATH500":
                instruction = "Think the question step by step and give the answer."
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
                cur = []
                for tem in task_prompts:
                    cur.append({"role": "user", "content": tem["Q"] + "\n\n"})
                    cur.append({"role": "assistant", "content":  tem["A"] + "\n\n\n"})
                
                query = "Problem: " + query + "\n\n" + "Solution: "

            elif task == "coin_flip":
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
                cur = []
                for tem in task_prompts:
                    cur.append({"role": "user", "content": "Q: " + tem["Q"]})
                    cur.append({"role": "assistant", "content": "A: " + tem["A"]})
                query = "Q: " + query

            elif task == "last_letter":
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
                cur = []
                for tem in task_prompts:
                    cur.append({"role": "user", "content": "Q: " + tem["Q"]})
                    cur.append({"role": "assistant", "content": "A: " + tem["A"]})
                query = "Q: " + query

            elif task == "common":
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
                cur = []
                for tem in task_prompts:
                    cur.append({"role": "user", "content": "Q: " + tem["Q"]})
                    cur.append({"role": "assistant", "content": "A: " + tem["A"]})
                query = "Q: " + query

            elif task == "strategy":
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
                cur = []
                for tem in task_prompts:
                    cur.append({"role": "user", "content": "Q: " + tem["Q"]})
                    cur.append({"role": "assistant", "content": "A: " + tem["A"]})
                query = "Q: " + query

            elif task == "GSM8K":
                if few_shot == False:
                    instruction = "You are a helpful assistant. Think the question step by step and give the answer (conclude the answer with \"so the answer is\" at the end of your reply.):\n"
                elif one_shot == True:
                    instruction = "You are a helpful assistant. Think the question step by step and give the answer:\n"
                elif onestep == True:
                    instruction = "You are a helpful assistant. Think the question in one step and give the answer:\n"
                else:
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
                cur = []
                if one_shot:
                    task_prompts = task_prompts[:1]
                for tem in task_prompts:
                    cur.append({"role": "user", "content": tem["Q"]})
                    if onestep:
                        cur.append({"role": "assistant", "content": tem["A"]})
                    else:
                        cur.append({"role": "assistant", "content": "Let's think step by step. " + tem["A"]}) # "Let's think step by step. " +
                query = query
            else:
                cur = []
                instruction = instruction

            count = 0
            while True:
                if not few_shot:
                    messages = [{"role": "system", "content": instruction}] + [{"role": "user", "content": query}]
                elif onestep:
                    messages = [{"role": "system", "content": instruction}] + cur + [{"role": "user", "content": query + " Directly give the answer without think step by step (start your answer with \"the answer is\"):"}]
                else:
                    messages = [{"role": "system", "content": instruction}] + cur + [{"role": "user", "content": query}]


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
                    if count > 0:
                        return [" "]
                    print("An error occurred:", e)
                    count += 1
                    time.sleep(18)




if __name__ == "__main__":
    Gen = Get()
    model = 4
    temp = 0
    n = 1
    sub_n = 1
    samples = []
    i = 0
    all_temp = [0]
    few_shot = True
    onestep = False
    oneshot = True
    dataset = ["GSM8K", "MATH", "MATH500", "coin_flip", "strategy", "last_letter", "common"]

    dataset = dataset[0]
    
    if dataset == "GSM8K":
        input_path = "gsm8k_test.json"
        with open(input_path, "r") as f:
            data = json.load(f)
            f.close()

    if dataset == "MATH500":
        input_path = "MATH500.jsonl"
        data = []
        with open(input_path, "r") as f:
            for line in f:
                sub_data = json.loads(line)
                data.append(deepcopy(sub_data))
            f.close()

    if dataset == "MATH":
        input_path = "MATH.jsonl"
        data = []
        with open(input_path, "r") as f:
            for line in f:
                sub_data = json.loads(line)
                data.append(deepcopy(sub_data))
            f.close()

    if dataset == "coin_flip":
        input_path = "coin_flip.jsonl"
        data = []
        with open(input_path, "r") as f:
            for line in f:
                sub_data = json.loads(line)
                data.append(deepcopy(sub_data))
            f.close()

    if dataset == "strategy":
        input_path = "strategy.jsonl"
        data = []
        with open(input_path, "r") as f:
            for line in f:
                sub_data = json.loads(line)
                data.append(deepcopy(sub_data))
            f.close()

    if dataset == "last_letter":
        input_path = "last_letter.jsonl"
        data = []
        with open(input_path, "r") as f:
            for line in f:
                sub_data = json.loads(line)
                data.append(deepcopy(sub_data))
            f.close()

    if dataset == "common":
        input_path = "common.jsonl"
        data = []
        with open(input_path, "r") as f:
            for line in f:
                sub_data = json.loads(line)
                data.append(deepcopy(sub_data))
            f.close()


    flag = False
    count = 0
    while True:
        count += 1
        print("loop time={}".format(count))
        if flag:
            break
        flag = True
        for temp in all_temp:
            samples = []
            print("temperature={}".format(temp))
            if not few_shot:
                out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}_zeroshot.jsonl".format(dataset, model, n, temp)
            elif oneshot:
                out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}_oneshot.jsonl".format(dataset, model, n, temp)
            elif onestep:
                out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}_onestep.jsonl".format(dataset, model, n, temp)
            else:
                out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}.jsonl".format(dataset, model, n, temp)
            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    for line in f:
                        pre_data = json.loads(line)
                        samples.append(pre_data)
                    f.close()
                for i in tqdm(range(len(samples))):
                    item = samples[i]
                    if len(item["completion"]) < sub_n or (len(item["completion"])>0 and item["completion"][0]==" "):
                        if len(item["completion"])>0 and item["completion"][0]==" ":
                            samples[i]["completion"] = []
                        flag = False
                        completion = Gen.calc(query=item["input"], temp=temp, n=sub_n, model=model, task=dataset, few_shot=few_shot, onestep=onestep)[:sub_n]
                        if len(completion) == sub_n and completion[0]!=" ":
                            samples[i]["completion"].extend(completion)

                    #write_jsonl("result/{}_gpt_{}_samples_n_{}_temp_{}_step_by_step.jsonl".format(dataset, model, n, temp), samples)
                write_jsonl(out_path, samples)


            else:
                if not few_shot:
                    out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}_zeroshot.jsonl".format(dataset, model, n, temp)
                elif oneshot:
                    out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}_oneshot.jsonl".format(dataset, model, n, temp)
                elif onestep:
                    out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}_onestep.jsonl".format(dataset, model, n, temp)
                else:
                    out_path = "result/{}_gpt_{}_samples_n_{}_temp_{}.jsonl".format(dataset, model, n, temp)

                flag = False
                if dataset != "MATH" and dataset != "MATH500":
                    for item in tqdm(data):
                        completion = Gen.calc(query=item["input"], temp=temp, n=sub_n, model=model, task=dataset, few_shot=few_shot, onestep=onestep, one_shot=oneshot)[:sub_n]
                        if len(completion) == sub_n:
                            samples.append(dict(input=item["input"], answer=item["output"],
                                                completion=completion))
                        else:
                            samples.append(dict(input=item["input"], answer=item["output"],
                                                completion=[]))
                        write_jsonl(out_path, samples)
                    write_jsonl(out_path, samples)
                else:
                    for item in tqdm(data):
                        completion = Gen.calc(query=item["problem"], temp=temp, n=sub_n, model=model, task=dataset, few_shot=few_shot, onestep=onestep, one_shot=oneshot)[:sub_n]
                        if len(completion) == sub_n:
                            samples.append(dict(input=item["problem"], answer=item["answer"], hard_level=item["level"], subject=item["subject"],
                                                completion=completion))
                        else:
                            samples.append(dict(input=item["problem"], answer=item["answer"], hard_level=item["level"],
                                                subject=item["subject"],
                                                completion=[]))
                        write_jsonl(out_path, samples)
                    write_jsonl(out_path, samples)

