import pandas as pd
import argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)
    
    args = parser.parse_args()

    with open(f"../response_evaluation/{args.dataset}/{args.name}/resultsEvaluations.jsonl") as f:
        total_len = 0
        count = 0
        for line in f:
            data = json.loads(line)
            if data['correctness'] == "1":
                count += 1
            total_len += 1
        acc = count / total_len
        print(acc)
    
if __name__ == "__main__":
    main()