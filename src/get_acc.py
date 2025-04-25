import pandas as pd
import argparse, json
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def get_token_num(text):
    tokens = word_tokenize(text)
    return len(tokens)

def get_acc(args):

    total_len_dict = {
        "basicprompt": 0,
        "mathPrompt": 0,
        "implicit_symbol_reasoning_prompt": 0,
        "explicit_symbol_reasoning_prompt": 0,
        "hint_prompt": 0,
    }
        
    correct_count_dict = {
        "basicprompt": [0, 0, 0, 0, 0],
        "mathPrompt": [0, 0, 0, 0, 0],
        "implicit_symbol_reasoning_prompt": [0, 0, 0, 0, 0],
        "explicit_symbol_reasoning_prompt": [0, 0, 0, 0, 0],
        "hint_prompt": [0, 0, 0, 0, 0],
    }

    token_num_dict = {
        "basicprompt": [0, 0, 0, 0, 0],
        "mathPrompt": [0, 0, 0, 0, 0],
        "implicit_symbol_reasoning_prompt": [0, 0, 0, 0, 0],
        "explicit_symbol_reasoning_prompt": [0, 0, 0, 0, 0],
        "hint_prompt": [0, 0, 0, 0, 0],
    }

    ind = 0
    with open(f"../response_evaluation/{args.dataset}/{args.name}/resultsEvaluations.jsonl") as f:
        for line in f:
            ind += 1
            # print(ind)
            data = json.loads(line)

            prompttype = data["PromptType"]
            total_len_dict[prompttype] += 1
            if data['correctness'] == "1":
                correct_count_dict[prompttype][0] += 1
                token_num_dict[prompttype][0] += get_token_num(data['Response'])
                if total_len_dict[prompttype] <= 50:
                    correct_count_dict[prompttype][1] += 1
                    token_num_dict[prompttype][1] += get_token_num(data['Response'])
                if total_len_dict[prompttype] > 25 and total_len_dict[prompttype] <= 50:
                    correct_count_dict[prompttype][2] += 1
                    token_num_dict[prompttype][2] += get_token_num(data['Response'])
                if total_len_dict[prompttype] <= 25:
                    correct_count_dict[prompttype][3] += 1
                    token_num_dict[prompttype][3] += get_token_num(data['Response'])
                if total_len_dict[prompttype] >50:
                    correct_count_dict[prompttype][4] += 1
                    token_num_dict[prompttype][4] += get_token_num(data['Response'])
            

        acc_dict = {k: [v[0]/250, v[1]/50, v[2]/25, v[3]/25, v[4]/200] for k, v in correct_count_dict.items()} # still need to handle missing items
        token_dict = {k: [v[0]/250, v[1]/50, v[2]/25, v[3]/25, v[4]/200] for k, v in token_num_dict.items()}
        print(acc_dict)
        print(token_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)
    
    args = parser.parse_args()

    get_acc(args)

    
if __name__ == "__main__":
    main()