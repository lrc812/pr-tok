import csv

def extract_captions_to_prompts(input_csv, output_csv):
    """
    Extract the 'Caption' column from input_csv and save it to output_csv.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """
    with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        
        # Write header for prompts.csv
        writer.writerow(['Prompt'])
        
        # Extract and write the 'Caption' column
        for row in reader:
            writer.writerow([row['Caption']])

# Example usage:
if __name__ == "__main__":
    input_csv = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/test_captions_modified.csv"
    output_csv = "/home/disk1/lihaoran/vq-gan/taming-transformers/evaluation/t2i/prompts.csv"
    extract_captions_to_prompts(input_csv, output_csv)
