import collections
import re
from tqdm import tqdm
import gzip
import time

DEBUG = False  # Set to False to disable debug prints


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def train_bpe(filename, num_merges):
    # Read and preprocess the file
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        lines = file.read().splitlines()

    # Tokenize: characters separated by spaces, with '§' as end-of-word marker
    tokens_list = [
        ' '.join(list(word) + ['§'])
        for line in tqdm(lines, desc="Reading and tokenizing lines")
        for word in line.split()
    ]
    debug_print("token list:", tokens_list)
    token_frequencies = collections.Counter(tokens_list)
    debug_print("token_freq :", token_frequencies)

    # Initialize vocabulary as unique characters
    vocab = set(char for token in tokens_list for char in token.split())
    debug_print("initial vocab:", vocab)
    print("init vocab size: ", len(vocab))

    pair_to_indexes = collections.defaultdict(lambda: (set(), int))
    pairs_freq = collections.defaultdict(int)

    # Count pair frequency
    def count_freq_in_token(token, pair):
        freq_in_token = 0
        for i in range(len(token) - 1):
            if token[i] == pair[0] and token[i + 1] == pair[1]:
                freq_in_token += 1
        return freq_in_token

    # Calculate frequencies for all pairs
    def pair_freq_update_calc(pair_to_indexes, new_pairs):
        #    print("Calculating pair frequencies...")
        # pairs_freq = collections.defaultdict(int)
        for pair in new_pairs:  # tqdm(pair_to_indexes.items(), desc="Calculating pair frequencies"):
            indexes, frequency = pair_to_indexes[pair]
            for idx in indexes:
                token = tokens_list[idx].split()  # Split the token into symbols
                pair_count = count_freq_in_token(token, pair)
                pairs_freq[pair] += pair_count  # Accumulate the frequency

        debug_print("Pair frequencies:", dict(pairs_freq))
        return pairs_freq

    # changes indexes is a list of all indexes of tokens that has changed.
    def update_pair_to_indexes(tokens, changed_indexes=None):
        #    print("updating pair_to_indexes...")
        if changed_indexes is None:
            changed_indexes = range(len(tokens))
        list_of_new_pairs = []
        # Update pair-to-index mappings for the specified indices
        for idx in changed_indexes:
            word = tokens[idx]
            symbols = word.split()  # Split token into characters or sub-tokens
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair not in pair_to_indexes:
                    pair_to_indexes[pair] = (set(),int)
                    list_of_new_pairs.append(pair)
                indexes, frequency = pair_to_indexes[pair]
                indexes.add(idx)  # Correctly add the index to the set
                pair_to_indexes[pair] = (indexes, frequency)
        debug_print("changed indexes:", changed_indexes)
        debug_print("pair_to_indexes:", dict(pair_to_indexes))
        return list_of_new_pairs

    def merge_vocab(pair, tokens_list):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        #   changed_tokens = []
        changed_indexes = []

        # Merge the pair in the tokens list
        indexes_words, frequency = pair_to_indexes[pair]
        for i_word in indexes_words:
            word = tokens_list[i_word]
            symbols = word.split()
            merged_word = pattern.sub(''.join(pair), word)

            # Update frequencies of adjacent pairs
            for i in range(len(symbols) - 1):
                if (symbols[i], symbols[i + 1]) == pair:
                    if i > 0:  # Update left neighbor pair
                        left_pair = (symbols[i - 1], symbols[i])
                        pairs_freq[left_pair] = 0
                    if i + 2 < len(symbols):  # Update right neighbor pair
                        right_pair = (symbols[i+1], symbols[i + 2])
                        pairs_freq[right_pair] = 0

            tokens_list[i_word] = merged_word
            pairs_freq[pair] = 0  # Mark this pair as merged
            if merged_word != word:
                changed_indexes.append(i_word)

        return tokens_list, changed_indexes

    new_pairs = update_pair_to_indexes(tokens_list)
    debug_print("new pairs: ", new_pairs)
    pairs_freq = pair_freq_update_calc(pair_to_indexes, new_pairs)

    # Perform BPE merges
    for i in tqdm(range(num_merges), desc="Performing BPE merges"):
        #        pairs = get_stats(pair_to_indexes)
        if not pairs_freq:
            break
        best = max(pairs_freq, key=pairs_freq.get)
        vocab.add(''.join(best))
        debug_print(f"Step {i + 1}: Merged pair {best} freq {pairs_freq[best]} ")
        debug_print("vocab: ", vocab)
        pairs_freq[best] = 0
        tokens_list, changed_indexes = merge_vocab(best, tokens_list)
        new_pairs = update_pair_to_indexes(tokens_list, changed_indexes)

    #    print("new pairs: ", new_pairs)

        debug_print("token list :", tokens_list, "\n----------------------------------------------\n")

        pairs = pair_freq_update_calc(pair_to_indexes, new_pairs)

    # sort vocab by a-b
    sorted_vocab = sorted(vocab)

    return sorted_vocab


if __name__ == "__main__":
    filename = "english.txt.gz"
    N = 30000
    # Start the timer
 #   start_time = time.time()

    # Execute the function
    vocab = train_bpe(filename, N)

    # End the timer
 #   end_time = time.time()

    # Calculate elapsed time
 #   elapsed_time = end_time - start_time
  #  debug_print(f"Elapsed time: {elapsed_time:.6f} seconds")
    # vocab = train_bpe(filename, N)
    print("Final Vocabulary Size:", len(vocab))
    print("Sample Vocabulary:", list(vocab))
    with open("vocab_english.txt", 'w', encoding='utf-8') as file:
        # Write each vocabulary item on a new line
        for word in vocab:
            file.write(word + '\n')
    print(f"Vocabulary written to {filename}")
    print("Final Vocabulary Size:", len(vocab))
