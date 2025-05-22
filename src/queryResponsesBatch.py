#!/usr/bin/env python3
# batch_runner.py
# pip install --upgrade openai pandas numpy

import os, json, time, argparse, datetime, textwrap, uuid, tempfile
import pandas as pd
from openai import OpenAI

# ─────────────────────────── helpers ────────────────────────────
def read_txt_files(directory):
    """Return {prompt_name: prompt_text} for every .txt in directory."""
    result = {}
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.endswith(".txt"):
                with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                    result[fn[:-4]] = f.read()
    return result

def write_jsonl(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def wait_for_batch(client, batch_id, sleep_sec=10):
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        if status in ("completed", "failed", "expired", "cancelling", "cancelled"):
            return batch
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
              f"Batch {batch_id} still {status} …")
        time.sleep(sleep_sec)

# ───────────────────────── model catalogue ──────────────────────
# Keep ONLY OpenAI‑endpoint models for batches.
modelInfo = {
    "GPT-o3": {
        "key": "OPENAI_API_KEY",
        "modelName": "o3-2025-04-16",
        "url": "https://api.openai.com/v1",
    },
    # Gemini / DeepSeek / Together rows removed – batch can’t hit those hosts
}

# ───────────────────────────── main ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", choices=["Math", "Logic", "Math_rewritten"], required=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    # --- paths ---
    out_dir = f"../responses/{args.dataset}/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    data_path = f"../data/braingle/braingle_{args.dataset}.csv"

    # --- read data & prompts ---
    df = pd.read_csv(data_path).head(args.rows)
    prompts = read_txt_files("../prompting/brainteaserPrompts")

    # --- Stage JSONL requests ---
    requests = []
    metadata_rows = []   # will turn into DataFrame later

    for row_idx, row in df.iterrows():
        q = row["Question"]
        a = row["Answer"]
        h = row["Hint"]

        if q == None:
            print("No question found, skipping")

        for _ in range(args.samples):
            for prompt_name, prompt_text in prompts.items():
                for model_key, spec in modelInfo.items():
                    # Compose the user message (optionally w/ hint)
                    user_msg = f"Question: {q}"
                    if "hint" in prompt_text.lower():
                        user_msg += f"\nHint: {h}"

                    # One request line for the batch JSONL
                    custom_id = str(uuid.uuid4())
                    requests.append({
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": spec["modelName"],
                            "messages": [
                                {"role": "system", "content": prompt_text},
                                {"role": "user",   "content": user_msg}
                            ],
                            "temperature": 0.0,
                            "stream": False
                        }
                    })

                    # Store everything we’ll need to rebuild the CSV
                    metadata_rows.append({
                        "custom_id": custom_id,
                        "ID": row_idx,
                        "Question": q,
                        "Model": model_key,
                        "PromptType": prompt_name,
                        "Human-written solution": a,
                        "Hint": h
                    })

    # Write to a temp JSONL file
    tmp_jsonl = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl").name
    write_jsonl(requests, tmp_jsonl)
    print(f"Wrote {len(requests):,} requests to {tmp_jsonl}")

    # --- kick off the batch job ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
    timeout=None  )                       # uses OPENAI_API_KEY env‑var
    file_obj = client.files.create(file=open(tmp_jsonl, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Submitted batch {batch.id}. Waiting for completion …")

    batch = wait_for_batch(client, batch.id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch finished with status {batch.status}")

    # --- download results JSONL ---
    output_file = client.files.retrieve(batch.output_file_id)
    out_path = os.path.join(out_dir, "raw_results.jsonl")
    with open(out_path, "wb") as f:
        f.write(client.files.content(output_file.id).read())
    print(f"Batch output saved to {out_path}")

    # --- build the DataFrame we need ---
    results = pd.read_json(out_path, lines=True)
    meta_df = pd.DataFrame(metadata_rows).set_index("custom_id")
    results = results.set_index("custom_id")
    merged = meta_df.join(results, how="left")

    # unpack the assistant message
    merged["Response"] = merged["response"].apply(
        lambda r: r["choices"][0]["message"]["content"]
    )
    merged["Status"] = merged["status"]
    final_cols = ["ID", "Question", "Model", "PromptType",
                  "Response", "Status", "Human-written solution", "Hint"]
    merged[final_cols].to_csv(f"{out_dir}/results.csv", index=False)
    print(f"All done → {out_dir}/results.csv")

if __name__ == "__main__":
    main()
