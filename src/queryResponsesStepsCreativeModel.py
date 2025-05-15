#pip install --upgrade openai
#pip install pandas
#pip install numpy
import os
import argparse
import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
import re
# from google import genai
# from google.genai import types

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
    #"GeminiFlash" : {"key": "GEMINI_API_KEY", "modelName": "gemini-2.5-flash-preview-04-17", "url":"https://generativelanguage.googleapis.com/v1beta/openai"},
    #"GeminiPro" : {"key": "GEMINI_API_KEY", "modelName": "gemini-2.5-pro-exp-03-25", "url":"https://generativelanguage.googleapis.com/v1beta/openai"},
    #"DSChat" : {"key": "DEEPSEEK_API_KEY", "modelName": "deepseek-chat", "url": "https://api.deepseek.com"},
    #"DSReason" : {"key": "DEEPSEEK_API_KEY", "modelName": "deepseek-reasoner", "url": "https://api.deepseek.com"},
    #"Qwen1" : {"key": "TOGETHER_API_KEY", "modelName": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "url": "https://api.together.xyz/v1"},
    #"Qwen14" : {"key": "TOGETHER_API_KEY", "modelName": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "url": "https://api.together.xyz/v1"},
    #"Qwen70" : {"key": "TOGETHER_API_KEY", "modelName": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "url": "https://api.together.xyz/v1"},
}

def query(question, instructions, hint, solution, model):
    # print("TESTING", model)

    client = OpenAI(
        api_key= os.getenv(modelInfo[model]["key"]),
        base_url = modelInfo[model]["url"],
    )

    if "hint" in instructions.lower():
        question = f"Question: {question}\n Hint: {hint}"

    response = client.chat.completions.create(
        model=modelInfo[model]["modelName"],
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
            {"role": "user", "content": solution}
        ],
        stream=False
    )
    return response.choices[0].message.content

    # response = client.responses.create(
    #                 model=modelInfo[model]["modelName"],
    #                 instructions=instructions,
    #                 input=question,
    #                 max_output_tokens=10000,
    #             )
    # return response.output_text, response.status

def main():
   #api_key = os.getenv(modelInfo[model]["key"])
    #print("→ Using API key:", api_key is not None)
    #client = OpenAI(api_key=api_key)
    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #print(client.models.list())   # should include “o3-2025-04-16”
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    #parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic", "Logic1"], required=True)
    parser.add_argument("--rows", help="Number of rows to sample", type=int, default=1, required=False)
    parser.add_argument("--samples", help="Number of samples to run", type=int, default=1, required=False)
    
    args = parser.parse_args()
    os.makedirs(f"../responses/Math/{args.name}", exist_ok=True)

    data = pd.read_csv(f'../responses/Math/FinalMath-DSChat/resultsAll.csv')
    outputs = pd.DataFrame(columns=['ID', 'Question', 'Hint', 'Human Solution', 'Model', 'PromptType', 'Response', 'Status', 'StepCount','Steps', 'Creative', 'Rudimentary'])

    instructionSet = {
    "stepcounter": readPrompt("../prompting/brainteaserPrompts/creativity_experiments/stepcountercreative.txt")
    }
    #print("Instructions:", instructionSet)
    #print("PWD:", os.getcwd())
    #print("Loaded prompts:", instructionSet.keys(), "— lengths:",
      #{k: len(v) for k,v in instructionSet.items()})
    for index, row in data.iterrows():
        if index >= args.rows:
            break

        for _ in range(args.samples):

            for prompt in instructionSet:
                for model in modelInfo.keys():
                    instructions = instructionSet[prompt]
                    question = row['Question']
                    solution = row['Response']
                    hint = row['Hint']

                    try:
                        response = query(question, instructions, hint, solution, model)
                        num_steps = None
                        m = re.search(r'Total Step Count:\s*(\d+)', response)
                        if m:
                            num_steps = int(m.group(1))

                        creative_steps = None
                        m_creative = re.search(r'Creative Steps:\s*(\d+)', response)
                        if m_creative:
                                creative_steps = int(m_creative.group(1))

                        rudimentary_steps = None
                        m_rudimentary = re.search(r'Rudimentary Steps:\s*(\d+)', response)
                        if m_rudimentary:
                            rudimentary_steps = int(m_rudimentary.group(1))

                        parts = response.split("Steps:", 1)
                        steps_block = parts[1].strip() if len(parts) == 2 else None

                    except Exception as e:
                        print(f"Error querying {model} for question {index} with prompt {prompt}: {e}")

                    outputs = pd.concat([outputs, pd.DataFrame({
                        'ID': [index],
                        'Question': [question],
                        'Hint': [hint],
                        'Human Solution': [solution],
                        'Model': [model],
                        'PromptType': [prompt],
                        'Response': [response],
                        'Status': [None],
                        'StepCount': [num_steps],
                        'Creative': [creative_steps],
                        'Rudimentary': [rudimentary_steps],
                        'Steps': [parts],
                    })], ignore_index=True)

                    #print(response)

    outputs.to_csv(f"../responses/Math/{args.name}/dscstepcountresults.csv", index=False)

if __name__ == "__main__":
    main()