import gzip
from collections import Counter
from tqdm import tqdm
import string


def tokenize(text):
    print("in tokenize ...")
    # Define all punctuation to strip, including UTF-8 general punctuation
    utf8_general_punctuation = "".join([chr(i) for i in range(8192, 8303)])
    punctuation = utf8_general_punctuation + string.punctuation

    tokens = []
    for token in text.split():
        stripped_token = token.strip(punctuation)
        if stripped_token:
            tokens.append(stripped_token)

    return tokens


def generate_ngrams(tokens, n):
    print("in generate ...")

    """
    Generate n-grams from a list of tokens.
    """
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    # return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(tokens, n):
    print("in count_ngrams ...")

    """
    Count the frequencies of n-grams in the token list.
    """
    print("start generate ...")
    ngrams = generate_ngrams(tokens, n)
    ngram_counts = Counter(ngrams)
    return ngram_counts


def find_all_segments(tokens, min_freq=2):
    """
    Find all segments where all tokens appear at least `min_freq` times.
    """
    token_freq = Counter(tokens)
    segments = []
    current_segment = []

    for token in tokens:
        if token_freq[token] >= min_freq:
            current_segment.append(token)
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    if current_segment:
        segments.append(current_segment)

    return segments


def longest_ngram_with_freq(tokens, min_freq=2):
    """
    Find the longest n-gram that appears at least `min_freq` times using index-based splitting.
    """
    ngram_to_indices = {}
    max_length = 0
    longest_ngram = None

    # Initialize with unigrams
    for i, token in enumerate(tokens):
        ngram = (token,)
        if ngram not in ngram_to_indices:
            ngram_to_indices[ngram] = []
        ngram_to_indices[ngram].append(i)

    # Iteratively extend n-grams
    current_length = 1
    while ngram_to_indices:
        next_ngram_to_indices = {}

        for ngram, indices in ngram_to_indices.items():
            # Filter n-grams by frequency
            if len(indices) >= min_freq:
                if current_length > max_length:
                    max_length = current_length
                    longest_ngram = ngram

                # Extend the n-gram
                for idx in indices:
                    if idx + current_length < len(tokens):
                        extended_ngram = ngram + (tokens[idx + current_length],)  # הרחבה כ-tuple
                        if extended_ngram not in next_ngram_to_indices:
                            next_ngram_to_indices[extended_ngram] = []
                        next_ngram_to_indices[extended_ngram].append(idx)

        ngram_to_indices = next_ngram_to_indices
        current_length += 1

    return longest_ngram, max_length


def read_gzip_file(file_path):
    print("in read ...")

    """Reads a gzip file and returns its text content."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return f.read()


def process_file(file_path):
    """
    Process a gzipped input file and compute required n-gram statistics.
    """
    print("start read ...")
    text = read_gzip_file(file_path)
    print("finish read ...")

    # Tokenize text
    print("start tokenize ...")
    tokens = tokenize(text)
    print("finished tokenize ...")

    print("start longest repeated ngram ...")
    longest_ngram, freq = longest_ngram_with_freq(tokens)

    print("finished longest repeated ngram ...")
    results = {"Longest n-gram": (longest_ngram, freq)}

    return results


def print_results(results):
    """
    Print the results to the console.

    """
    for file_name, data in results.items():
        print(f"\nResults for {file_name}:\n" + "-" * 40)
        for key, value in data.items():
            if key == "Longest n-gram":
                ngram, freq = value
                print(f"{key}: {ngram} (len: {freq})")
            else:
                print(f"{key}:")
                for ngram, freq in value:
                    print(f"  {ngram}: {freq}")
        print("-" * 40)


def main():
    # Define input files (compressed .gz files)
    input_files = ["english.txt.gz"]  # , "english.txt.gz"]  # Replace with your actual .gz file paths

    # Process each file
    all_results = {}
    print("start process ...")

    for file_path in tqdm(input_files, desc="Processing files"):
        all_results[file_path] = process_file(file_path)
    print("finish process ...")

    # Print results

    print_results(all_results)


if __name__ == "__main__":
    main()
