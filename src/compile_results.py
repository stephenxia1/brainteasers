
import os
import argparse
import datetime, time
import pandas as pd
import numpy as np




def compile_results(data_file_path, results_folder_path, metric = 'Correct'):

    # MODE: 'Correct', 'ModelBruteForce', 'HumanBruteForce'
    df = pd.read_csv(data_file_path)


    for root, _, files in os.walk(results_folder_path):
        for file in files:
            if file.endswith(".csv"):
                results_df = pd.read_csv(os.path.join(root, file))
                for index, row in results_df.iterrows():
                    col_label = row['Model'] + "+" + row['PromptType']
                    problem_id = row['ID']
                    if row['Response'] == '':
                        result = 0 # None
                    elif row['Status'] == False:
                        result = 0 # GradingError
                    else:
                        result = row[metric]
                    if col_label not in df.keys():
                        df.insert(df.shape[1], col_label, None)
                    df.loc[problem_id, col_label] = result

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Experiment Name", required=True)
    parser.add_argument("--dataset", help="Dataset to run on", choices=["Math", "Logic", "Math_rewritten", "Math_with_categories", "Logic_with_categories"], required=True)
    parser.add_argument("--results_folder", help="Folder with csv results", required=True)
    parser.add_argument("--metric", help="Metric to compile results for", default='Correct', choices=['Correct', 'ModelBruteForce', 'HumanBruteForce'])


    args = parser.parse_args()

    data_file_path = f'../data/braingle/braingle_{args.dataset}.csv'

    df = compile_results(data_file_path, args.results_folder)
    print(df)
    df.to_csv(f'{args.name}.csv')


if __name__ == "__main__":
    main()