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
    parser.add_argument("--rows", help="Number of rows to sample", type=int, default=1, required=False)
    parser.add_argument("--samples", help="Number of samples to run", type=int, default=1, required=False)
    
    args = parser.parse_args()
    os.makedirs(f"../../results/{args.name}", exist_ok=True)

    data = pd.read_csv(f'../data/braingle/braingle_{args.dataset}.csv')
    outputs = pd.DataFrame(columns=['Question', 'Response', 'Correct', 'BruteForce'])

    instructions = readPrompt(f'../../prompting/{args.name}.txt')
    
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
                model="gpt-4o",
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": question}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "math_reasoning",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "explanation": { "type": "string" },
                                            "output": { "type": "string" }
                                        },
                                        "required": ["explanation", "output"],
                                        "additionalProperties": False
                                    }
                                },
                                "final_answer": { "type": "string" }
                            },
                            "required": ["steps", "final_answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            print(response.output_text)

            # modelResponse = response.output[0].content[0]

            outputs = pd.concat([outputs, pd.DataFrame({
                'Question': [question],
                'Response': [response.output_text],
                'Correct': [solution],
                'BruteForce': [hint]
            })], ignore_index=True)

    outputs.to_csv(f"../results/{args.name}-results.csv", index=False)

if __name__ == "__main__":
    main()