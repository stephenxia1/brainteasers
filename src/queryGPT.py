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
    parser.add_argument("--prompt", help="Prompting File to use", required=True)
    parser.add_argument("--rows", help="Number of rows to sample", type=int, default=1, required=False)
    parser.add_argument("--samples", help="Number of samples to run", type=int, default=1, required=False)
    
    args = parser.parse_args()
    os.makedirs(f"../responses/{args.dataset}/{args.name}", exist_ok=True)

    data = pd.read_csv(f'../data/braingle/braingle_{args.dataset}.csv')
    outputs = pd.DataFrame(columns=['Question', 'Response', 'Correct', 'BruteForce'])

    instructions = readPrompt(f'../prompting/{args.prompt}.txt')
    print("Instructions:", instructions)

    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY")
    )
    
    for index, row in data.iterrows():
        if index >= args.rows:
            break

        for _ in range(args.samples):
            question = row['Question']
            solution = row['Answer']
            hint = row['Hint']

            response = client.responses.create(
                # model="gpt-o1-2024-12-17",
                model="o3-2025-04-16",
                instructions=instructions,
                input=question,
                # text={
                #     "format": {
                #         "type": "json_schema",
                #         "name": "math_reasoning",
                #         "schema": {
                #             "type": "object",
                #             "properties": {
                #                 "steps": {
                #                     "type": "array",
                #                     "items": {
                #                         "type": "object",
                #                         "properties": {
                #                             "explanation": { "type": "string" },
                #                             "output": { "type": "string" }
                #                         },
                #                         "required": ["explanation", "output"],
                #                         "additionalProperties": False
                #                     }
                #                 },
                #                 "final_answer": { "type": "string" }
                #             },
                #             "required": ["steps", "final_answer"],
                #             "additionalProperties": False
                #         },
                #         "strict": True
                #     }
                # }
                max_output_tokens=25000,
            )
            print(response.output_text)

            print("TOKENS USED:", response.usage.total_tokens)

            # modelResponse = response.output[0].content[0]
            modelResponse = response.output_text

            outputs = pd.concat([outputs, pd.DataFrame({
                'Question': [question],
                'Response': [modelResponse],
                'Correct': [None],
                'BruteForce': [None],
                'Status': [response.status],
            })], ignore_index=True)

    outputs.to_csv(f"../responses/{args.dataset}/{args.name}/results.csv", index=False)

if __name__ == "__main__":
    main()