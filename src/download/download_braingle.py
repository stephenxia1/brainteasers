from typing import List
import os
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm


def scrape_braingle_math_QAs(base_url: str = 'https://www.braingle.com',
                             category_url: str = 'https://www.braingle.com/brainteasers/Math.html',
                             verbose: bool = False):
    '''
    Webscrape the braingle math website and get all question-answer pairs and metadata.
    '''
    all_question_page_urls = get_all_question_page_urls(base_url=base_url,
                                                        front_page_url=category_url,
                                                        verbose=verbose)
    question_answer_and_metadata = []
    if verbose:
        print(f'Scraping the question-answer pairs and metadata from {len(all_question_page_urls)} questions.')

    for url in tqdm(all_question_page_urls):
        title, question, answer, hint, popularity, difficulty = \
            get_question_answer_and_metadata(base_url=base_url, question_page_url=url)
        if title and question and answer:
            question_answer_and_metadata.append((title, question, answer, hint, popularity, difficulty))

        # Be nice to the server.
        time.sleep(0.1)

    return question_answer_and_metadata

def get_all_question_page_urls(base_url: str,
                               front_page_url: str,
                               verbose: bool = False):
    '''
    Get the list of urls of all question pages.
    '''
    all_question_page_urls = []
    webpage_url = front_page_url

    if verbose:
        print(f'Finding all question links by crawling {front_page_url}.')
        page_count = 1

    while True:
        response = requests.get(webpage_url)
        if response.status_code != 200:
            break

        # Get all contents from the main page.
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the links to all question pages.
        question_page_items = soup.select('td.nowrap.col_max')
        if not question_page_items:
            break

        # Add all question page urls to the list.
        for item in question_page_items:
            question_page_url = base_url + item.find('a')['href']
            if question_page_url not in all_question_page_urls:
                all_question_page_urls.append(question_page_url)

        if verbose:
            print(f'Found {len(all_question_page_urls)} questions after crawling {page_count} pages.')
            page_count += 1

        # Find the 'next page' button and crawl the next page.
        next_page_button = find_button_starting_with_phrase(soup, start_phrase='Next')
        if next_page_button is None:
            break
        webpage_url = base_url + next_page_button['href']

        # Be nice to the server.
        time.sleep(0.1)

    return all_question_page_urls

def find_button_starting_with_phrase(soup, start_phrase: str):
    '''
    Find a button that starts with a given phrase.
    '''
    for a in soup.find_all('a'):
        if a.text.strip().startswith(start_phrase):
            return a
    return None

def get_question_answer_and_metadata(base_url: str, question_page_url: str):
    '''
    Get all question-answer pairs and metadata given a question page url.
    '''

    # Open the question page.
    response_question = requests.get(question_page_url)

    # Parse and find the link to the question-answer page.
    soup = BeautifulSoup(response_question.content, 'html.parser')
    show_answer_link = find_button_starting_with_phrase(soup, start_phrase='Show Answer')
    question_answer_url = base_url + show_answer_link['href']

    # Open the question-answer page.
    response_show_answer = requests.get(question_answer_url)

    # Parse the question-answer page.
    soup = BeautifulSoup(response_show_answer.content, 'html.parser')

    # 1. Extract title.
    title_block = soup.find('h1', {'property': 'name headline'})
    title = title_block.get_text('\n', strip=True)

    # 2. Extract question.
    question_block = soup.find('div', class_='textblock', property='text')
    question = question_block.get_text('\n', strip=True)

    # 3. Extract answer.
    answer_block = soup.find('div', class_='ans_s').find('span', class_='textblock')
    answer = answer_block.get_text('\n', strip=True)

    # 4. Extract (optional) hint.
    hint = ''
    hint_button = soup.find('a', id='bt_hideHint')
    if hint_button:
        hint_block = hint_button.find_previous('div', class_='hint_s')
        hint = hint_block.get_text('\n', strip=True).lstrip('Hint\n')

    # 5. Extract popularity and difficulty.
    popularity, difficulty = extract_popularity_and_difficulty(soup)

    return title, question, answer, hint, popularity, difficulty

def extract_popularity_and_difficulty(soup):
    '''
    Extract the popularity and difficulty from a given question-answer page.
    '''
    metadata_block = soup.find('div', class_='box_strip')
    popularity, difficulty = '', ''

    if metadata_block:
        metadata = metadata_block.find_all('span', class_='item')
        for item in metadata:
            text = item.get_text(strip=True)
            if text.startswith('Fun:'):
                rating = item.find('span', class_='hide_400')
                if rating:
                    popularity = rating.text.strip('()')
            elif text.startswith('Difficulty:'):
                rating = item.find('span', class_='hide_400')
                if rating:
                    difficulty = rating.text.strip('()')

    return float(popularity), float(difficulty)


def save_to_csv(data: List[List[str]], output_csv_path: str):
    df = pd.DataFrame(data, columns=['Title', 'Question', 'Answer', 'Hint', 'Popularity/Fun', 'Difficulty'])
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    return


if __name__ == '__main__':
    for category in ['Math', 'Logic']:
        category_url = f'https://www.braingle.com/brainteasers/{category}.html'

        output_csv_path = f'../../data/braingle/braingle_{category}_all.csv'
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        question_answer_and_metadata = scrape_braingle_math_QAs(base_url='https://www.braingle.com',
                                                                category_url=category_url,
                                                                verbose=True)

        save_to_csv(data=question_answer_and_metadata, output_csv_path=output_csv_path)

        print(f'Saved {len(question_answer_and_metadata)} question-answer pairs and metadata to {output_csv_path}.')