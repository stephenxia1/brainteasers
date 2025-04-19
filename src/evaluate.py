import pandas as pd
import argparse
import os
import numpy as np
from openai import OpenAI

def read_txt_files(directory):
    instructionSet = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        instructionSet[file[:-4]] = content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return instructionSet

def evaluateResponse(client, instructions, modelResponse, solution):
    response = client.responses.create(
        model="o3-2025-04-16",
        instructions=instructions,
        input="STUDENT RESPONSE:\n" + modelResponse + "\n\nSOLUTION:\n" + solution,
        max_output_tokens=10000,
    )
    return response.output_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)

    args = parser.parse_args()

    data = pd.read_csv(f'../data/braingle/braingle_{args.dataset}.csv')
    responses = pd.read_csv(f'../responses/{args.dataset}/{args.name}/results.csv')

    evaluationPrompts = read_txt_files("../prompting/evaluationPrompts")

    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY")
    )

    responses['Correct'] = np.nan

    for index, row in responses.iterrows():
        question = row['Question']
        dataEntry = data[data['Question'] == question].iloc[0]
        solution = dataEntry['Answer']
        modelResponse = row['Response']

        correctness = evaluateResponse(client, evaluationPrompts['correctness'], modelResponse, solution)

        responses.at[index, 'Correct'] = correctness
        
    responses.to_csv(f"../response_evaluation/{args.dataset}/{args.name}-evaluation.csv", index=False)

if __name__ == "__main__":
    main()