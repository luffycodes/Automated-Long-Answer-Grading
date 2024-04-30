import pandas as pd
import argparse
import json
import csv
import evaluate
import random
import sys
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import ast
import openai
from openai import OpenAI
from tqdm import tqdm


def add_params():
    parser = argparse.ArgumentParser()

    # Read arguments
    parser.add_argument("--dataset_name", type=str, default="CHEM121", help="the name of the dataset")
    parser.add_argument("--raw_data_dir", type=str, default="data", help="the directory to the data")
    parser.add_argument("--processed_data_dir", type=str, default="data_processed", help="the directory to the data")

    parser.add_argument("--test_size", type=float, default=0.1, help="the size of the testing dataset")

    parser.add_argument("--use_rubric", action='store_true', default=True, help="whether to prompt with rubrics")
    parser.add_argument("--prompt_choice", type=str, default="zero-shot", help="choice of prompting method")

    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="the model name")
    parser.add_argument("--max_tokens", type=int, default=512, help="maximum number of token generated")
    parser.add_argument("--seed", type=int, default=42, help="the seed for the random module")

    parser.add_argument("--output_dir", type=str, default="llm_output", help="the directory to the output")
    parser.add_argument("--cache_dir", type=str, default="cache", help="the directory to the cache")

    params = parser.parse_args()

    params.valid_size = 0.1 if params.test_size <= 0.1 else 0.2
    params.train_size = round(1.0 - params.valid_size - params.test_size, 1)

    return params

def concat_path(data_dir, files):
    prefix = data_dir + '/'
    for i in range(len(files)):
        files[i] = prefix + files[i]

def prepare_prompt(params, question, rubric_item, student_response):
    content = ""
    
    if params.use_rubric:
   
        content = \
f'''You are an automated grader for a college-level chemistry class. Your task is to evaluate students' responses to chemistry exam questions based on the rubric items provided. Your objective is to determine whether the rubric item has been correctly addressed by the student's response.

If a rubric item is considered correctly answered, it means that the student's response meets the expectations outlined in that specific criterion.

If a rubric item is considered incorrectly answered, it indicates that the student's response does not meet the expectations outlined in that specific criterion. The response may lack essential elements, contain errors, or deviate from the required standards. The incorrect designation signifies that the student's response does not align with the predefined rubric criteria and does not demonstrate proficiency in that particular aspect of the question.

This is the question: {question}

This is the rubric item: {rubric_item}\n'''

        content += \
f'''This is the response that you need to grade: {student_response}

If the rubric item is correcly answered, return the correctness is T, representing true. Otherwise, if the rubric is not correctly answered, return the correctness as F, representing false. Corrrectness:'''

    return content

def prepare_gpt_messages(params, question, rubric_item, student_response):
    content = ""
    
    if params.use_rubric:
        json_str = '''{"Reason": "...", "Correctness": "..."}'''

        content = \
f'''You are an automated grader for a college-level chemistry class. Your task is to evaluate students' responses to chemistry exam questions based on the rubric items provided. Your objective is to determine whether the rubric item has been correctly addressed by the student's response.

If a rubric item is considered correctly answered, it means that the student's response meets the expectations outlined in that specific criterion.

If a rubric item is considered incorrectly answered, it indicates that the student's response does not meet the expectations outlined in that specific criterion. The response may lack essential elements, contain errors, or deviate from the required standards. The incorrect designation signifies that the student's response does not align with the predefined rubric criteria and does not demonstrate proficiency in that particular aspect of the question.

Format your entire output in the following JSON structure:
{json_str} where "Reason" contains all of your reasoning to reach the conclusion, and "Correctness" represents whether the rubric item is correctly answered. If a rubric item is correcly answered, "Correctness" will be "True". Otherwise, "Correctness" will be "False". All other values are invalid. Make the json structure the only thing in your output.

This is the question: {question}

This is the rubric item: {rubric_item}'''

        content += f'''This is the response that you need to grade: {student_response}'''
    return [{"role": "system", "content": ""},
        {"role": "user", "content": content}]
    
def prompt_model(model, tokenizer, prompt):    
    # Get logits
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits[:, -1, :]

    # Compute the probs
    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits[:, tokenizer(" T").input_ids[-1]],
                    logits[:, tokenizer(" F").input_ids[-1]],
                ]
            ).float(),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )

    pred = np.argmax(probs)
    return {0: "T", 1: "F"}[pred]

def prompt_gpt(params, model, prompt):
    try:
        response = model.chat.completions.create(model=params.model, messages=prompt, max_tokens=params.max_tokens, temperature=1.0)
    except (openai.APIConnectionError, openai.APIError) as e:
        time.sleep(60)
        return prompt_gpt(params, prompt)
    return response
    
def predict_with_rubric(model, tokenizer, params, question, rubric_items, answer):
    pred_values = []

    for rubric_item in rubric_items:
        # Predict with GPT
        if tokenizer is not None:
            prompt = prepare_prompt(params, question, rubric_item, answer)

            pred = prompt_model(model, tokenizer, prompt)
            pred_values.append(1 if pred=="T" else 0)
        # Predict with an open-sourced LLM
        else:
            messages = prepare_gpt_messages(params, question, rubric_item, answer)

            response = prompt_gpt(params, model, messages)

            try:
                correctness = ast.literal_eval(response.choices[0].message.content)["Correctness"]
            except:
                correctness = ""
            
            if correctness == "":
                pred_values.append(-1)
            else:
                pred_values.append(1 if correctness=="True" else 0)
                
    return pred_values

if __name__ == "__main__":
    # Get user params
    params = add_params()

    os.makedirs(params.output_dir, exist_ok=True)
    
    # Set the seed
    random.seed(params.seed)

    # List answer csvs
    answer_csvs = ["Sample CHEM121 Student Answers.xlsx - F20 Exam 2 Q4.csv",
                   "Sample CHEM121 Student Answers.xlsx - F20 Exam 2 Q7.csv",
                   "Sample CHEM121 Student Answers.xlsx - F20 Exam 3 Q7.csv",
                   "Sample CHEM121 Student Answers.xlsx - F20 Final Exam Q3.csv"]

    # Get the test dataset split csv
    sizes = str(params.train_size) + '_' + str(params.valid_size) + '_' + str(params.test_size)
    test_dataset_split = params.processed_data_dir + '/' + params.dataset_name + "_" + sizes + "/test.csv"

    # Concat the path
    concat_path(params.raw_data_dir, answer_csvs)
    
    # Map ids to questions along with their rubrics
    with open(params.processed_data_dir + "/question_rubrics.json", 'r') as file:
        id2ques = json.load(file)

    # Read the test dataset split file
    test_file = open(test_dataset_split, 'r')
    reader = csv.reader(test_file)

    # Create the output file
    prefix = params.dataset_name + "_" + sizes
    output_file_name = params.output_dir + '/' +  prefix +  '_' + params.model.split("/")[-1] + ('_rubric_' if params.use_rubric else '_no-rubric_') + params.prompt_choice + ".csv"

    output_csv = open(output_file_name, 'w')
    writer = csv.writer(output_csv)

    # Create the model
    if "gpt" in params.model:
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        model = OpenAI(api_key=api_key)
        tokenizer = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            params.model,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir=params.cache_dir,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(params.model, cache_dir=params.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token

    # Load the metrics
    acc_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    rubric_values = []
    pred_rubric_values = []
    
    for i, row in enumerate(tqdm(reader, desc="Processing rows")):
        # Get the data
        question = id2ques[row[0]]["question"] # the question
        rubric_items = id2ques[row[0]]["items"] # the rubric items
        answer = row[1] # the answer

        values = [1 if value.lower() == "true" else 0 for value in row[2:-1]] # the rubric values
        score = int(values.count(1) / int(row[-1]) * 8) # the score

        # Prompt the GPT
        pred_row = [i, answer]
        
        if params.use_rubric:
            pred_values = predict_with_rubric(model, tokenizer, params, question, rubric_items, answer)

            if len(pred_values) == 0:
                continue

            pred_score = int(pred_values.count(1) / int(row[-1]) * 8)
            
            pred_row.extend([values, pred_values, score, pred_score])

            rubric_values.extend(values)
            pred_rubric_values.extend(pred_values)
        
        writer.writerow(pred_row)
        
    test_file.close()
    output_csv.close()

    # Evaluate the binary classification
    print("Logging the binary evaluation metrics...")

    acc = acc_metric.compute(predictions=pred_rubric_values, references=rubric_values)["accuracy"]
    precision = precision_metric.compute(predictions=pred_rubric_values, references=rubric_values, average="macro")["precision"]
    recall = recall_metric.compute(predictions=pred_rubric_values, references=rubric_values, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=pred_rubric_values, references=rubric_values, average="macro")["f1"]

    print(f"  accuracy  = {acc:.4f}")
    print(f"  precision = {precision:.4f}")
    print(f"  recall    = {recall:.4f}")
    print(f"  f1        = {f1:.4f}")

    output_eval_file_name = params.output_dir + '/' +  prefix +  '_' + params.model.split("/")[-1] + ('_rubric_' if params.use_rubric else '_no-rubric_') + params.prompt_choice + ".json"
    with open(output_eval_file_name, 'w') as file:
        eval_results = {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        json.dump(eval_results, file, indent=4)