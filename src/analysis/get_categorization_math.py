import os
import time
import pandas as pd
from openai import OpenAI
import openai
# ───────── CONFIG ────────────────

# 2) Input/output paths
INPUT_CSV  = "../../data/braingle/braingle_Math.csv"
OUTPUT_CSV = "../../data/braingle/braingle_Math_with_categories.csv"

# 3) Model and rate-limit pacing
MODEL_NAME     = "o3-2025-04-16"
REQUEST_DELAY  = 1.0    # seconds between calls to avoid rate-limit

# ───────── TAXONOMY PROMPT ─────────

CATEGORY_DEFINITIONS = """
Please read the following 10 category labels and their descriptions, then assign the given math brain‐teaser to one or more of these buckets.  Respond with a comma-separated list of category numbers only (e.g. “1,4”):

1. Arithmetic & Mental-Calculation  
   Puzzles where the main task is to juggle +, -, ×, ÷ quickly or cleverly (e.g. “Make 24” style).

2. Number-Theory & Divisibility  
   Teasers hinging on primes, factors, remainders, digital roots, etc.

3. “Where’s the Dollar?” — Algebraic Word Puzzles  
   Word riddles whose twist is a hidden algebraic mismatch or variable.

4. Geometry & Spatial Reasoning  
   Problems about areas, angles, tilings, or “think outside the box” dot puzzles.

5. Sequences, Series & Patterns  
   “Find the next term” or “explain the rule” puzzles (e.g. Fibonacci).

6. Probability & Expected-Value  
   Dice, cards, urns, Monty Hall-style counterintuitive odds questions.

7. Combinatorics & Optimization  
   “How many ways…?” or “minimum moves/time” challenges (e.g. bridge-and-torch).

8. Logic-Driven Math Teasers  
   Numeric logic–constraint puzzles (Zebra-style, knights-and-knaves with numbers).

9. Measurement, Time & Rates  
   Clock puzzles, rope-timers, filling/draining tanks, travellers-meet questions.

10. Recreational Algorithms & Move-Counting Games  
    Recursive/algorithmic games (Tower of Hanoi, peg-solitaire, Rubik’s-cube).
"""

# ───────── FUNCTION TO CALL o3 ─────────

def categorize_with_o3(text, answer):
    """
    Sends the problem text to o3 with the taxonomy definitions,
    returns the comma-separated category numbers.
    """
    prompt = CATEGORY_DEFINITIONS + "\n\nProblem:\n" + text + "\n\nAnswer: \n" + answer + "\n\nCategories:"
    retries = 0
    MAX_RETRIES=5
    RETRY_DELAY=5
    client = OpenAI(
        api_key= os.getenv("OPENAI_API_KEY"),
        base_url = "https://api.openai.com/v1"
    )
    while retries < MAX_RETRIES:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    # {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            cats = resp.choices[0].message.content
            print(cats)
            return cats
            
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

# ───────── MAIN WORKFLOW ─────────

def main():
    # Load your CSV of brain-teasers
    df = pd.read_csv(INPUT_CSV)

    # Ensure there's a column named e.g. 'question' or 'text' – adjust if needed
    PROBLEM_FIELD = "Question"  # or "title", or however your column is named
    ANSWER_FIELD = "Answer"

    # Create a new column for the categories
    df["categories"] = ""

    for idx, row in df.iterrows():
        problem_text = str(row.get(PROBLEM_FIELD, "")).strip()
        answer_text = str(row.get(ANSWER_FIELD, "")).strip()
        if not problem_text:
            df.at[idx, "categories"] = ""
            continue

        try:
            cats = categorize_with_o3(problem_text, answer_text)
            df.at[idx, "categories"] = cats
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            df.at[idx, "categories"] = ""
        
        # pace ourselves to avoid rate-limits
        time.sleep(REQUEST_DELAY)

    # Write out the augmented CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
