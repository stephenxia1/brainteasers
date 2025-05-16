from typing import Optional, Tuple
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def load_braingle_data(category: str, topk: Optional[int] = None) -> pd.DataFrame:

    # Load the dataframe.
    csv_path = f'../../data/braingle/braingle_{category}_all.csv'
    df = pd.read_csv(csv_path)

    # Sort by difficulty, descendingly.
    df = df.sort_values(by='Difficulty', ascending=False).reset_index()

    # Only select the top k difficult entries, if applicable.
    if topk:
        df = df[:topk]

    return df


def compute_hint_percentage(df: pd.DataFrame) -> float:
    '''
    Find the percentage of questions with hints.
    '''
    count_no_hint = df['Hint'].isna().sum().item()
    count_all = len(df)
    return 100 * (count_all - count_no_hint) / count_all


def compute_statistics(df: pd.DataFrame, attribute: str) -> Tuple[float]:
    '''
    Compute the population statistics of a given attribute
    '''
    scores = df[attribute].to_numpy()

    statistics = {
        'mean': scores.mean(),
        'std': scores.std(),
        'max': scores.max(),
        '3rd quartile': np.percentile(scores, 75),
        'median': np.percentile(scores, 50),
        '1st quartile': np.percentile(scores, 25),
        'min': scores.min(),
    }

    for key in statistics.keys():
        if isinstance(statistics[key], np.generic) or isinstance(statistics[key], np.ndarray):
            statistics[key] = statistics[key].item()

    return statistics

def compute_word_statistics(df: pd.DataFrame) -> Tuple[float]:
    '''
    Find how many words each answer has, and compute the population statistics.
    '''
    df['AnswerWords'] = count_words(df['Answer'])
    return compute_statistics(df, attribute='AnswerWords')

def compute_sentence_statistics(df: pd.DataFrame) -> Tuple[float]:
    '''
    Find how many sentence each answer has, and compute the population statistics.
    '''
    df['AnswerSentences'] = count_sentences(df['Answer'])
    return compute_statistics(df, attribute='AnswerSentences')

def count_words(series: pd.Series) -> pd.Series:
    '''
    Count how many words each item in the series has.
    '''
    word_counts = []
    for item in series:
        word_counts.append(len(word_tokenize(item)))
    word_counts = pd.Series(word_counts)
    return word_counts

def count_sentences(series: pd.Series) -> pd.Series:
    '''
    Count how many sentences each item in the series has.
    '''
    sentence_counts = []
    for item in series:
        sentence_counts.append(len(sent_tokenize(item)))
    sentence_counts = pd.Series(sentence_counts)
    return sentence_counts


if __name__ == '__main__':
    nltk.download('punkt_tab')

    for category in ['Math', 'Logic']:
        for topk in [None, 250]:
            df = load_braingle_data(category=category, topk=topk)

            hint_percentage = compute_hint_percentage(df)
            stats_difficulty = compute_statistics(df, 'Difficulty')
            word_statistics = compute_word_statistics(df)
            sentence_statistics = compute_sentence_statistics(df)

            print(f'\n\nCategory: {category}, topk={topk}, count={len(df)}')
            print(f'Hint Percentage: {hint_percentage}%')
            print(f'Difficulty Stats: {stats_difficulty}')
            print(f'Answer Word Stats: {word_statistics}')
            print(f'Answer Sentence Stats: {sentence_statistics}')
