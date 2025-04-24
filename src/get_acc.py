import pandas as pd
import argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)
    
    args = parser.parse_args()

    correct_count_dict = {
        "basicprompt": [0, 0, 0, 0],
        "mathPrompt": [0, 0, 0, 0],
        "implicit_symbol_reasoning_prompt": [0, 0, 0, 0],
        # "explicit_symbol_reasoning_prompt": [0, 0, 0, 0],
        "hint_prompt": [0, 0, 0, 0],
    }

    with open(f"../response_evaluation/{args.dataset}/{args.name}/resultsEvaluations.jsonl") as f:
        total_len = 0
        for line in f:
            data = json.loads(line)

            if data['correctness'] == "1":
                correct_count_dict[data["PromptType"]][0] += 1
                if total_len <= 50:
                    correct_count_dict[data["PromptType"]][1] += 1
                if total_len >= 25 and total_len <= 50:
                    correct_count_dict[data["PromptType"]][2] += 1
                if total_len <= 25:
                    correct_count_dict[data["PromptType"]][3] += 1
            total_len += 1

        acc_dict = {k: [v[0]/250, v[1]/50, v[2]/25, v[3]/25] for k, v in correct_count_dict.items()} # still need to handle missing items
        print(acc_dict)
    
if __name__ == "__main__":
    main()