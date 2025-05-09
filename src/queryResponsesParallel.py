# pip install --upgrade openai
import os
import argparse
import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from tqdm.contrib.concurrent import process_map
import json


def readPrompt(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print("Instructions file not found.")
        return ""
    
def read_txt_files(directory):
    """Crawls through a directory and reads the content of each .txt file.

    Args:
        directory: The path to the directory to crawl.
    """
    instructionSet = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # print(f"Content of {file_path}:\n{content}\n{'='*20}")
                        instructionSet[file[:-4]] = content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return instructionSet

'''
Comment out models to skip evaluation on them.
'''
modelInfo = {
    "GPT-o3" : {"key": "OPENAI_API_KEY", "modelName": "o3-2025-04-16", "url": "https://api.openai.com/v1"},
    "GeminiFlash" : {"key": "GEMINI_API_KEY", "modelName": "gemini-2.5-flash-preview-04-17", "url":"https://generativelanguage.googleapis.com/v1beta/openai"},
    "GeminiPro" : {"key": "GEMINI_API_KEY", "modelName": "gemini-2.5-pro-exp-03-25", "url":"https://generativelanguage.googleapis.com/v1beta/openai"},
    "DSChat" : {"key": "DEEPSEEK_API_KEY", "modelName": "deepseek-chat", "url": "https://api.deepseek.com"},
    "DSReason" : {"key": "DEEPSEEK_API_KEY", "modelName": "deepseek-reasoner", "url": "https://api.deepseek.com"},
    "Qwen1" : {"key": "TOGETHER_API_KEY", "modelName": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "url": "https://api.together.xyz/v1"},
    "Qwen14" : {"key": "TOGETHER_API_KEY", "modelName": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "url": "https://api.together.xyz/v1"},
    "Qwen70" : {"key": "TOGETHER_API_KEY", "modelName": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "url": "https://api.together.xyz/v1"},
}

def query(question, instructions, model):
    # print("TESTING", model)

    # print("QUERYING", model)

    client = OpenAI(
        api_key= os.getenv(modelInfo[model]["key"]),
        base_url = modelInfo[model]["url"],
    )

    try:
        response = client.chat.completions.create(
            model=modelInfo[model]["modelName"],
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
            stream=False,
            timeout=360,
            max_completion_tokens=10000,
        )
    except Exception as e:
        print(f"Error querying {model}: {e}")
        response = None
    # print("RESPONSE", response)
    return response.choices[0].message.content

def process_task(t):
    return process_pair(*t)

def process_pair(index, prompt, question, hint, solution, instructions, model, dataset, name):
    status = True
    results = pd.read_csv(f"../responses/{dataset}/{name}/resultsTemp.csv")
    # print("LOOKING AT PROBLEM", index, "WITH PROMPT", prompt)
    if "hint" in instructions.lower():
        question = f"Question: {question}\n Hint: {hint}"

    if "SolutionSummary" in name:
        question = f"Question: {question}\n Solution: {solution}"

    try:
        response = query(question, instructions, model)
    except Exception as e:
        # print(f"Error querying {model} for question {index} on prompt {prompt}: {e}")
        response = None
        status = False

    entry = {
        'ID'        : index,
        'Question'  : question,
        'Hint'      : hint,
        'Human Solution'  : solution,
        'Model'     : model,
        'PromptType': prompt,
        'Response'  : response,
        'Status'    : status
    }
    
    results = pd.concat([results, pd.DataFrame([entry])], ignore_index=True)
    results.to_csv(f"../responses/{dataset}/{name}/resultsTemp.csv", index=False)


    print("FINISHED!")

    return entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)
    parser.add_argument("--rows", help="Number of rows to sample", type=int, default=1, required=False)
    parser.add_argument("--samples", help="Number of samples to run", type=int, default=1, required=False)
    parser.add_argument("--model", help="Model to run on", choices=modelInfo.keys(), required=False, default="GPT-o3")
    args = parser.parse_args()

    os.makedirs(f"../responses/{args.dataset}/{args.name}", exist_ok=True)
    pd.DataFrame(columns=['ID', 'Question', 'Hint', 'Human Solution', 'Model', 'PromptType', 'Response', 'Status']).to_csv(f"../responses/{args.dataset}/{args.name}/resultsTemp.csv", index=False)

    data = pd.read_csv(f'../data/braingle/braingle_{args.dataset}.csv')

    instructionSet = read_txt_files("../prompting/brainteaserPrompts")
    # instructionSet = {'basicprompt': instructionSet['basicprompt'], 'mathPrompt': instructionSet['mathPrompt']}
    # instructionSet = {'solutionSummary': instructionSet['solutionSummary']}
    instructionSet = {'hintPrompt': instructionSet['hint_prompt']}

    results = []
    for prompt in instructionSet:
        
        for _ in range(args.samples):
            for index, row in itertools.islice(data.iterrows(), min(args.rows, len(data))):
                # if (index < 223):
                #     continue
                task = (
                    index,
                    prompt,
                    row['Question'],
                    row['Hint'],
                    row['Answer'],
                    instructionSet[prompt],
                    args.model,
                    args.dataset,
                    args.name,
                )

                # print("TASK:", task)
                entry = process_task(task)
                results.append(entry)
                with open(f'../responses/{args.dataset}/{args.name}/results.jsonl', 'a') as jsonfile:
                    jsonfile.write(json.dumps(entry) + "\n")

        pd.DataFrame(results).to_csv(f"../responses/{args.dataset}/{args.name}/resultsAll.csv", index=False)

if __name__ == "__main__":
    main()