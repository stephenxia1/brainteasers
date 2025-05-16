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

def evaluateResponse(client, instructions, problem, modelResponse, solution, evaluationModel):
    if len(modelResponse) == 0:
        return None
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model = evaluationModel, # 
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": "PROBLEM:\n" + problem + "\n\nSTUDENT RESPONSE:\n" + modelResponse + "\n\nSOLUTION:\n" + solution},
                ],
                stream = False,
                max_completion_tokens=10000,
            )
            # print(response.choices[0].message.content)
            return response.choices[0].message.content

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
    
    print("Max retries reached. Returning None.")

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--model", help="Evaluation Model", required=False, default="o3-2025-04-16")
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)
    parser.add_argument("--from_row", help="Continue evaluation from which row", type=int, default=0, required=False)

    args = parser.parse_args()

    # data = pd.read_csv(f'../data/braingle/braingle_{args.dataset}.csv')
    responses = pd.read_csv(f'../responses/{args.dataset}/{args.name}/resultsAll.csv')

    print("Responses length:", len(responses))
    evaluationPrompts = read_txt_files("../prompting/evaluationPrompts")

    os.makedirs(f"../response_evaluation/{args.dataset}/{args.name}", exist_ok=True)

    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY")
    )

    responses['Correct'] = np.nan

    responses_iloc = responses.iloc[args.from_row:]

    for index, row in responses_iloc.iterrows():
        question = row['Question']
        # dataEntry = data[data['Question'] == question].iloc[0]
        # solution = dataEntry['Answer']
        solution = row['Human Solution']
        modelResponse = str(row['Response'])

        # print(row.to_dict().keys())
        # print(row)

        # if row['PromptType'] == "nl_to_symbol_prompt":
        #     continue

        # if type(modelResponse) == type("string"):
            
        correctness1 = evaluateResponse(client, evaluationPrompts['correctness'], question, modelResponse, solution, args.model)
        correctness2 = evaluateResponse(client, evaluationPrompts['correctness'], question, modelResponse, solution, args.model)
        correctness3 = evaluateResponse(client, evaluationPrompts['correctness'], question, modelResponse, solution, args.model)
        correctness4 = evaluateResponse(client, evaluationPrompts['correctness'], question, modelResponse, solution, args.model)
        correctness5 = evaluateResponse(client, evaluationPrompts['correctness'], question, modelResponse, solution, args.model)
        modelbruteforced1 = evaluateResponse(client, evaluationPrompts['brute-force'], question, modelResponse, solution, args.model)
        modelbruteforced2 = evaluateResponse(client, evaluationPrompts['brute-force'], question, modelResponse, solution, args.model)
        modelbruteforced3 = evaluateResponse(client, evaluationPrompts['brute-force'], question, modelResponse, solution, args.model)
        modelbruteforced4 = evaluateResponse(client, evaluationPrompts['brute-force'], question, modelResponse, solution, args.model)
        modelbruteforced5 = evaluateResponse(client, evaluationPrompts['brute-force'], question, modelResponse, solution, args.model)
        humanbruteforced1 = evaluateResponse(client, evaluationPrompts['brute-force'], question, solution, solution, args.model)
        humanbruteforced2 = evaluateResponse(client, evaluationPrompts['brute-force'], question, solution, solution, args.model)
        humanbruteforced3 = evaluateResponse(client, evaluationPrompts['brute-force'], question, solution, solution, args.model)
        humanbruteforced4 = evaluateResponse(client, evaluationPrompts['brute-force'], question, solution, solution, args.model)
        humanbruteforced5 = evaluateResponse(client, evaluationPrompts['brute-force'], question, solution, solution, args.model)
        # responses.at[index, 'Correct1'] = correctness1
        # responses.at[index, 'Correct2'] = correctness2
        # responses.at[index, 'Correct3'] = correctness3
        # responses.at[index, 'Correct4'] = correctness4
        # responses.at[index, 'Correct5'] = correctness5
        # responses.at[index, 'ModelBruteForce1'] = modelbruteforced1
        # responses.at[index, 'ModelBruteForce2'] = modelbruteforced2
        # responses.at[index, 'ModelBruteForce3'] = modelbruteforced3
        # responses.at[index, 'ModelBruteForce4'] = modelbruteforced4
        # responses.at[index, 'ModelBruteForce5'] = modelbruteforced5
        # responses.at[index, 'HumanBruteForce1'] = humanbruteforced1
        # responses.at[index, 'HumanBruteForce2'] = humanbruteforced2
        # responses.at[index, 'HumanBruteForce3'] = humanbruteforced3
        # responses.at[index, 'HumanBruteForce4'] = humanbruteforced4
        # responses.at[index, 'HumanBruteForce5'] = humanbruteforced5         
        entry = row.to_dict()

        # entry["correctness"] = correctness
        # entry["model_bruteforce"] = modelbruteforced
        # entry["human_bruteforce"] = humanbruteforced
        entry["correctness"] = [correctness1, correctness2, correctness3, correctness4, correctness5]
        entry["model_bruteforce"] = [modelbruteforced1, modelbruteforced2, modelbruteforced3, modelbruteforced4, modelbruteforced5]
        entry["human_bruteforce"] = [humanbruteforced1, humanbruteforced2, humanbruteforced3, humanbruteforced4, humanbruteforced5]

        with open(f'../response_evaluation/{args.dataset}/{args.name}/resultsEvaluations_evaluatedby{args.model}.jsonl', 'a') as jsonfile:
            jsonfile.write(json.dumps(entry) + "\n")
        
    responses.to_csv(f"../response_evaluation/{args.dataset}/{args.name}-evaluation_from_row{args.from_row}.csv", index=False)

if __name__ == "__main__":
    main()