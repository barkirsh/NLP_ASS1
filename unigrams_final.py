import gzip
import os
import re
from collections import Counter, defaultdict
import string

def read_gzip_file(file_path):
    """Reads a gzip file and returns its text content."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        return f.read()


def tokenize(text):
    # Define all punctuation to strip, including UTF-8 general punctuation
    utf8_general_punctuation = "".join([chr(i) for i in range(8192, 8303)])
    punctuation = utf8_general_punctuation + string.punctuation

    tokens = []
    for token in text.split():
        stripped_token = token.strip(punctuation)
        if stripped_token:
            tokens.append(stripped_token)

    return tokens

def count_token_types(tokens):
    """Counts occurrences of each token."""
    return Counter(tokens)


def analyze_file(file_path):
    """Analyzes token counts, splits halves, and computes required metrics."""
    content = read_gzip_file(file_path)
    tokens = tokenize(content)
    total_count = len(tokens)

    # Split into halves
    mid_index = total_count // 2
    first_half = tokens[:mid_index]
    second_half = tokens[mid_index:]

    # Count token types in each half
    first_half_counts = count_token_types(first_half)
    second_half_counts = count_token_types(second_half)
    full_counts = count_token_types(tokens)

    # Distinct types
    distinct_first_half = set(first_half_counts.keys())
    distinct_full = set(full_counts.keys())
    added_types = distinct_full - distinct_first_half

    # Identify tokens meeting frequency thresholds in the second half
    more_than_two = [t for t, c in second_half_counts.items() if t not in distinct_first_half and c > 2]
    more_than_five = [t for t, c in second_half_counts.items() if t not in distinct_first_half and c > 5]
    more_than_ten = [t for t, c in second_half_counts.items() if t not in distinct_first_half and c > 10]

    return {
        "total_tokens": total_count,
        "distinct_first_half": len(distinct_first_half),
        "distinct_full": len(distinct_full),
        "added_types": added_types,
        "more_than_two": more_than_two,
        "more_than_five": more_than_five,
        "more_than_ten": more_than_ten,
        "full_counts": full_counts
    }


def top_n_tokens(counts, n=50):
    """Returns the top N tokens sorted by frequency."""
    return counts.most_common(n)


def frequency_buckets(counts):
    """Counts tokens by frequency buckets."""
    freq_buckets = defaultdict(int)
    for token, count in counts.items():
        if count == 1:
            freq_buckets[1] += 1
        elif count == 2:
            freq_buckets[2] += 1
        elif count == 3:
            freq_buckets[3] += 1
        elif count == 5:
            freq_buckets[5] += 1
        if count >= 10:
            freq_buckets["10+"] += 1
        if count >= 100:
            freq_buckets["100+"] += 1
        if count >= 1000:
            freq_buckets["1,000+"] += 1
        if count >= 10000:
            freq_buckets["10,000+"] += 1
    return freq_buckets


def main():
    # Analyze English and Hebrew files
    english_path = "english.txt.gz"
    hebrew_path = "hebrew.txt.gz"

    english_data = analyze_file(english_path)
    hebrew_data = analyze_file(hebrew_path)

    for lang, data in [("English", english_data), ("Hebrew", hebrew_data)]:
        print(f"\nAnalysis for {lang}:")
        print(f"1. Total tokens: {data['total_tokens']}")
        print(f"2. Distinct types in full file: {data['distinct_full']}")
        print(f"3. Tokens/Types: {data['total_tokens']/data['distinct_full']}")
        print(f"4. Distinct types in first half: {data['distinct_first_half']}")
        print(f"5. Types added in second half: {len(data['added_types'])}")
        print(f"6. Examples of types appearing only in second half:")
        print(f"   > More than 2 times: {data['more_than_two'][:5]}")
        print(f"   > More than 5 times: {data['more_than_five'][:5]}")
        print(f"   > More than 10 times: {data['more_than_ten'][:5]}")

        # Top 50 tokens
        top_50 = top_n_tokens(data["full_counts"])
        print(f"\nTop 50 tokens:")
        for token, count in top_50:
            print(f"{token}: {count}")

        # Frequency buckets
        buckets = frequency_buckets(data["full_counts"])
        print(f"\nFrequency Buckets:")
        for bucket, count in buckets.items():
            print(f"{bucket}: {count}")


if __name__ == "__main__":
    main()
