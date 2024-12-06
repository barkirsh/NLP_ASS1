import gzip
from collections import Counter
from tqdm import tqdm
import string


def tokenize(text):
   # print("in tokenize ...")
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


# finding all sequences where each one has only tokens that their freq in text is >=2.
def find_all_segments(tokens, min_freq=2):
    """
    Find all segments where all tokens appear at least `min_freq` times.
    """
    token_freq = Counter(tokens)
    segments = set()
    current_segment = []

    for token in tokens:
        if token_freq[token] >= min_freq:
            current_segment.append(token)
        else:
            if current_segment:
                segments.add(tuple(current_segment))
                current_segment = []
    if current_segment:
        segments.add(tuple(current_segment))

    return segments


def find_longest_ngram(tokens):
    segments = find_all_segments(tokens, 2)  # extracting all optional segments
    max_length = max(len(segment) for segment in segments)  # max len of segments #TODO count
    print("max optional n:", max_length)
    left, right = 0, max_length
    longest_ngram = ""
    longest_freq = 0
    while left <= right:
        mid = (left + right) // 2
        print("mid is :", mid)
        seen_ngrams = set()
        found = False
        for segment in segments:  #
            for i in range(len(segment) - mid + 1):
                ngram = (segment[i:i + mid])
                # seen_ngrams[ngram] += 1
                if ngram in seen_ngrams:
                    left = mid + 1
                    found = True
                    break
                else:
                    seen_ngrams.add(ngram)
            if found:
                break
        else:
            right = mid - 1
    #  for segements in len more than right all segments len right counter and then most common
    counter = Counter()
    for segment in segments:
        if len(segment) >= right:
            for i in range(len(segment) - right + 1):
                ngram = (segment[i:i + right])
                counter[ngram] += 1

    longest_ngram = counter.most_common(1)[0]

    return longest_ngram


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
    longest_ngram, freq = find_longest_ngram(tokens)

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
                print(f"{key}: {ngram} (Frequency: {freq})")
            else:
                print(f"{key}:")
                for ngram, freq in value:
                    print(f"  {ngram}: {freq}")
        print("-" * 40)


def main():
    # Define input files (compressed .gz files)
    input_files = ["hebrew.txt.gz"]  # , "english.txt.gz"]  # Replace with your actual .gz file paths

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
