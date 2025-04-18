# pip install --upgrade openai
import os
import argparse
import datetime
import pandas as pd
import numpy as np
from openai import OpenAI

def readPrompt(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print("Instructions file not found.")
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", default=f'TestExperiment-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}', required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], default="Math", required=True)

    args = parser.parse_args()
    os.makedirs(f"../../results/{args.name}", exist_ok=True)

    data = pd.read_csv(f'../data/braingle/braingle_{args.dataset}.csv')
    outputs = pd.DataFrame(columns=['Question', 'Response', 'Correct', 'BruteForce'])

    instructions = readPrompt(f'../../prompting/{args.name}.txt')
    
    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY")
    )
    
    for _, row in data.iterrows():
        question = row['Question']
        solution = row['Answer']
        hint = row['Hint']

        response = client.completions.create(
            model="gpt-4o",
            prompt=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": question}
            ]
        )

        modelResponse = response.choices[0].message.content

        outputs = outputs.append({
            'Question': question,
            'Response': modelResponse,
            'Correct': False,
            'BruteForce': False
        }, ignore_index=True)

if __name__ == "__main__":
    main()