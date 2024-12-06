import gzip
import os
import re
from collections import Counter
from fpdf import FPDF


def read_gzip_file(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return f.read()


def tokenize(text):
    # Remove punctuation except for internal hyphens
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    return tokens


def count_token_types(tokens):
    # Count the occurrences of each token type, and the total amount of occurrences
    token_counts = Counter(tokens)
    total_all_count = sum(token_counts.values())
    return token_counts, total_all_count


def write_tokens_to_file(token_counts, original_file_path):
    # Extract the original file name without extension
    base_name = os.path.basename(original_file_path)
    name_without_extension = os.path.splitext(base_name)[0]

    # Create the new file name with "tokens_" prefix
    output_file_name = f"tokens_{name_without_extension}.txt"

    # Write token counts to the file, each entry on a new line
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for token, count in token_counts.items():
            f.write(f"{token}: {count}\n")

    print(f"Token counts written to {output_file_name}")
    return output_file_name


# count tokens types
def main():
    # print("tokens:", tokenize(read_gzip_file("english.txt.gz")))
    print("English : ")
    file_path = "../english.txt.gz"
    # Step 1: Read content from the gzip file
    content = read_gzip_file(file_path)
    # Step 2: Tokenize the content
    tokens = tokenize(content)
    # Step 3: Count token types
    token_counts, all_tokens_count = count_token_types(tokens)
    distinct_tokens_count = len(token_counts)
    # Step 4: Write token counts to a new file
    write_tokens_to_file(token_counts, file_path)

    print("1 - tokens count:", all_tokens_count)
    print("2 - distinct tokens count: ", distinct_tokens_count)
    print("3 - tokens_counts / distinct tokens count :", all_tokens_count / distinct_tokens_count)

    # print("tokens:", tokenize(read_gzip_file("english.txt.gz")))
    print("Hebrew : ")
    file_path = "../hebrew.txt.gz"
    # Step 1: Read content from the gzip file
    content = read_gzip_file(file_path)
    # Step 2: Tokenize the content
    tokens = tokenize(content)
    # Step 3: Count token types
    token_counts, all_tokens_count = count_token_types(tokens)
    distinct_tokens_count = len(token_counts)
    # Step 4: Write token counts to a new file
    write_tokens_to_file(token_counts, file_path)

    print("1 - tokens count:", all_tokens_count)
    print("2 - distinct tokens count: ", distinct_tokens_count)
    print("3 - tokens_counts / distinct tokens count :", all_tokens_count / distinct_tokens_count)

if __name__ == "__main__":
    main()
