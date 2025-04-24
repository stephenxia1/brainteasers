import pandas as pd
import argparse
import os, json, time
import numpy as np
from openai import OpenAI
import openai

MAX_RETRIES = 5
RETRY_DELAY = 5 

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
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.responses.create(
                model="o3-2025-04-16", # 
                instructions=instructions,
                input="STUDENT RESPONSE:\n" + modelResponse + "\n\nSOLUTION:\n" + solution,
                max_output_tokens=10000,
            )
            # print(response.choices[0].message.content)
            return response.output_text

        except openai.AuthenticationError:
            print("Authentication failed: Invalid API key.")
            break  # Don't retry on bad key

        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
            print(f"Retryable error occurred: {e}. Retrying in {RETRY_DELAY} seconds...")
            retries += 1
            time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)

    args = parser.parse_args()

    # data = pd.read_csv(f'../data/braingle/braingle_{args.dataset}.csv')
    responses = pd.read_csv(f'../responses/{args.dataset}/{args.name}/resultsAll.csv')

    evaluationPrompts = read_txt_files("../prompting/evaluationPrompts")

    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY")
    )

    responses['Correct'] = np.nan

    # responses_iloc = responses.iloc[114:]

    for index, row in responses.iterrows():
        # question = row['Question']
        # dataEntry = data[data['Question'] == question].iloc[0]
        # solution = dataEntry['Answer']
        solution = row['Human Solution']
        modelResponse = row['Response']

        print(row.to_dict().keys())
        print(row)

        # if row['PromptType'] == "nl_to_symbol_prompt":
        #     continue

        if type(modelResponse) == type("string"):
            
            correctness = evaluateResponse(client, evaluationPrompts['correctness'], modelResponse, solution)

            responses.at[index, 'Correct'] = correctness

            entry = row.to_dict()

            entry["correctness"] = correctness
            print(correctness)

            with open(f'../response_evaluation/{args.dataset}/{args.name}/resultsEvaluations.jsonl', 'a') as jsonfile:
                jsonfile.write(json.dumps(entry) + "\n")
        
    responses.to_csv(f"../response_evaluation/{args.dataset}/{args.name}-evaluation.csv", index=False)

if __name__ == "__main__":
    main()