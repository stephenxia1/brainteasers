import json
import csv
import sys

def jsonl_to_csv(jsonl_file, csv_file):
    with open(jsonl_file, 'r') as infile, open(csv_file, 'w', newline='') as outfile:
        writer = None
        for line in infile:
            record = json.loads(line.strip())
            if writer is None:
                # Initialize CSV writer with fieldnames from the first record
                writer = csv.DictWriter(outfile, fieldnames=record.keys())
                writer.writeheader()
            writer.writerow(record)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jsonltocsv.py <input_jsonl_file> <output_csv_file>")
        sys.exit(1)

    input_jsonl = sys.argv[1]
    output_csv = sys.argv[2]
    jsonl_to_csv(input_jsonl, output_csv)
    print(f"Converted {input_jsonl} to {output_csv}")