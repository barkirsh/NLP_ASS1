import collections
import re
import sys

from tqdm import tqdm
import gzip
import time

DEBUG = True  # Set to False to disable debug prints


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def train_bpe(filename, num_merges):
    # Read and preprocess the file
    with open(filename, 'rt', encoding='utf-8') as file:
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

    pair_to_indexes = collections.defaultdict(lambda: (set(), 0))
    # changes indexes is a list of all indexes of tokens that has changed.
    def update_pair_to_indexes(tokens, changed_indexes=None, new_symbol=None):
        if changed_indexes is None:
            changed_indexes = range(len(tokens))  # Process all tokens initially
        #        new_pairs = set()
        debug_print("changed indexes:", changed_indexes)
        len_changed_indexes = len(changed_indexes)
        # Update pair-to-index mappings for the specified indices
        with tqdm(total=len_changed_indexes, desc="Performing Update pair_indexes") as pbar:
            update_counter_up = 0
            j = 0
            for idx in changed_indexes:
                word = tokens[idx]
                symbols = word.split()
                # Then, add the new pairs for the updated token
                for i in range(len(symbols) - 1):
                    new_pair = (symbols[i], symbols[i + 1])
                    if (new_symbol is None) or (symbols[i] == new_symbol or symbols[i + 1] == new_symbol):
                        #  new_pairs.add(new_pair)  # Track new pairs
                        if new_pair not in pair_to_indexes:
                            pair_to_indexes[new_pair] = (set(), 0)
                            indexes, freq = pair_to_indexes[new_pair]
                            indexes.add(idx)
                            pair_to_indexes[new_pair] = (indexes, 1)

                        else:
                            indexes, freq = pair_to_indexes[new_pair]
                            indexes.add(idx)  # Add the index of the updated token
                            pair_to_indexes[new_pair] = (indexes, freq + 1)
                j += 1
                update_counter_up += 1
                if update_counter_up == 10000 or (j + 1) == len_changed_indexes:
                    remaining_bar = len_changed_indexes - pbar.n
                    pbar.update(min(update_counter_up, remaining_bar))
                    update_counter_up = 0

        debug_print("Updated Pair-to-Indexes:", {k: (list(v[0]), v[1]) for k, v in pair_to_indexes.items()})

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
            new_symbol = ''.join(pair)
            merged_word = pattern.sub(new_symbol, word)

            # Update frequencies of adjacent pairs
            for i in range(len(symbols) - 1):
                if (symbols[i], symbols[i + 1]) == pair:
                    if i > 0:  # Update left neighbor pair
                        left_pair = (symbols[i - 1], symbols[i])
                        indexes, frequency = pair_to_indexes[left_pair]
                        frequency = 0
                        pair_to_indexes[left_pair] = (indexes, frequency)
                    if i + 2 < len(symbols):  # Update right neighbor pair
                        right_pair = (symbols[i + 1], symbols[i + 2])
                        indexes, frequency = pair_to_indexes[right_pair]
                        frequency = 0
                        pair_to_indexes[right_pair] = (indexes, frequency)

            tokens_list[i_word] = merged_word
            indexes, frequency = pair_to_indexes[pair]
            frequency = 0
            pair_to_indexes[pair] = (indexes, frequency)
            if merged_word != word:
                changed_indexes.append(i_word)

        return tokens_list, changed_indexes, new_symbol

    update_pair_to_indexes(tokens_list)
    #   debug_print("new pairs: ", new_pairs)
    #    pairs_freq = pair_freq_update_calc(pair_to_indexes, new_pairs)

    # Perform BPE merges
    with tqdm(total=num_merges, desc="Performing BPE merges") as pbar:
        update_counter = 0
        for i in range(num_merges):
            debug_print("*************","size of dict: ", sys.getsizeof(pair_to_indexes),"***************")
            #        pairs = get_stats(pair_to_indexes)
            if not pair_to_indexes:
                break
            best = max(pair_to_indexes.items(), key=lambda item: item[1][1])
            best_pair = best[0]
            best_indexes = best[1][0]
            best_freq = best[1][1]

            vocab.add(''.join(best_pair))
            debug_print(f"Step {i + 1}: Merged pair {best_pair} freq {best_freq} ")
            debug_print("vocab: ", vocab)
            pair_to_indexes[best_pair] = (best_indexes, best_freq)
            tokens_list, changed_indexes, new_symbol = merge_vocab(best_pair, tokens_list)
            update_pair_to_indexes(tokens_list, changed_indexes, new_symbol)

            debug_print("new pairs: ", pair_to_indexes)

            debug_print("token list :", tokens_list, "\n----------------------------------------------\n")

            update_counter += 1
            if update_counter == 1000 or (i + 1) == num_merges:
                remaining = num_merges - pbar.n
                pbar.update(min(update_counter, remaining))
                update_counter = 0

    #    pairs = pair_freq_update_calc(pair_to_indexes, new_pairs)

    # sort vocab by a-b
    sorted_vocab = sorted(vocab)

    return sorted_vocab


if __name__ == "__main__":
    filename = "test_video.txt"
    N = 8
    vocab = train_bpe(filename, N)
    with open("eng_vocab.txt", 'w', encoding='utf-8') as file:
        # Write each vocabulary item on a new line
        for word in vocab:
            file.write(word + '\n')
    print(f"Vocabulary written to {filename}")
    print("Final Vocabulary Size:", len(vocab))
