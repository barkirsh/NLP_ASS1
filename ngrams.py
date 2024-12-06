import gzip
from collections import Counter

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


def count_ngrams(tokens, n):
    print("in count_ngrams ...")

    """
    Count the frequencies of n-grams in the token list.
    """
    print("start generate ...")
    ngrams = generate_ngrams(tokens, n)
    ngram_counts = Counter(ngrams)
    return ngram_counts


def find_longest_repeated_ngram(tokens):
    print("in longest repeated ngrams ...")

    """
    Find the longest n-gram that appears more than once.
    """
    tokens_freq_more_1 =[]
    token_counts = Counter(tokens)

    for t in tokens:
        if token_counts[t] > 1:
            tokens_freq_more_1.append(t)

    max_length = len(tokens_freq_more_1)
    for n in range(max_length, 1, -1):
        ngram_counts = count_ngrams(tokens, n)
        for ngram, freq in ngram_counts.items():
            if freq > 1:
                return ngram, freq
    return None, 0


# def find_longest_repeated_ngram(tokens):
#     """
#     Find the longest n-gram that appears more than once, optimized to prune unnecessary computations.
#     """
#     print("in optimized longest repeated ngrams ...")
#
#     # Step 1: Filter tokens with frequency > 1
#     token_counts = Counter(tokens)
#     tokens = [token for token in tokens if token_counts[token] > 1]
#     if not tokens:
#         return None, 0  # No repeated tokens, so no repeated n-grams
#
#     # Step 2: Iteratively build and count n-grams
#     max_length = len(tokens)
#     ngram_counts = None
#     for n in range(2, max_length + 1):
#         ngram_counts = count_ngrams(tokens, n)
#         repeated_ngrams = {ngram: freq for ngram, freq in ngram_counts.items() if freq > 1}
#
#         if repeated_ngrams:
#             # Keep only the repeated n-grams for further analysis
#             tokens = [token for ngram in repeated_ngrams for token in ngram.split()]
#         else:
#             break  # No repeated n-grams of this length, stop early
#
#     # Find the longest repeated n-gram from the last valid ngram_counts
#     if ngram_counts:
#         for ngram, freq in sorted(ngram_counts.items(), key=lambda x: -len(x[0])):
#             if freq > 1:
#                 return ngram, freq
#
#     return None, 0


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
    print("finised tokenize ...")
    results = {
         "5-grams": count_ngrams(tokens, 5).most_common(10),
         "10-grams": count_ngrams(tokens, 10).most_common(10),
    }
   # print("start longest repeated ngram ...")
   # longest_ngram, freq = find_longest_repeated_ngram(tokens)
   # print("finished longest repeated ngram ...")
   # results["Longest n-gram"]= (longest_ngram, freq)

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
                print(f"{key}: {ngram} (Frequency: {freq})")
            else:
                print(f"{key}:")
                for ngram, freq in value:
                    print(f"  {ngram}: {freq}")
        print("-" * 40)


def main():
    # Define input files (compressed .gz files)
    input_files = ["hebrew.txt.gz","english.txt.gz"]  # Replace with your actual .gz file paths

    # Process each file
    all_results = {}
    print("start process ...")

    for file_path in input_files:
        all_results[file_path] = process_file(file_path)
    print("finish process ...")

    # Print results
    print_results(all_results)


if __name__ == "__main__":
    main()
