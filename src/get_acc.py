import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic"], required=True)
    
    args = parser.parse_args()

    with open(f"../response_evaluation/{args.dataset}/{args.name}-evaluation.csv", index=False) as f:
        df = pd.read_csv(f)
        count = (df["correctness"] == 1).sum()
        acc = count / len(df)
        print(acc)
    
if __name__ == "__main__":
    main()