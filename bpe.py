import heapq
import collections
import re
from tqdm import tqdm
import gzip
import time

DEBUG = True  # Set to False to disable debug prints


def debug_print(*args, **kwargs):
    if re.DEBUG:
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
    print("init vocab size: ", len(vocab),"\n----------------------------------------------\n","\n----------------------------------------------\n")

    pair_to_indexes = collections.defaultdict(set)
    # pairs_freq = collections.defaultdict(int)
    heap = []

    def count_freq_in_token(token, pair):
        freq_in_token = 0
        for i in range(len(token) - 1):
            if token[i] == pair[0] and token[i + 1] == pair[1]:
                freq_in_token += 1
        return freq_in_token

    def pair_freq_update_calc(pair_to_indexes, new_pairs):
        visited_pairs = set()  # To avoid duplicate recalculations
        for pair in new_pairs:
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)

            freq = 0
            for idx in pair_to_indexes[pair]:
                token = tokens_list[idx].split()
                freq += count_freq_in_token(token, pair)

            if freq > 0:
                heapq.heappush(heap, (-freq, pair))

        return heap

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
                    pair_to_indexes[pair] = set()
                    list_of_new_pairs.append(pair)
                pair_to_indexes[pair].add(idx)
        debug_print("changed indexes:", changed_indexes)
        debug_print("pair_to_indexes:", dict(pair_to_indexes))
        return list_of_new_pairs

    def merge_vocab(pair, tokens_list):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        changed_indexes = []

        for i_word in pair_to_indexes[pair]:
            word = tokens_list[i_word]
            symbols = word.split()
            merged_word = pattern.sub(''.join(pair), word)
            tokens_list[i_word] = merged_word

            if merged_word != word:
                changed_indexes.append(i_word)

                # Remove neighbors and set their frequency to 0
                for i in range(len(symbols) - 1):
                    current_pair = (symbols[i], symbols[i + 1])

                    if current_pair == pair:  # Focus on neighbors
                        if i > 0:  # Update left neighbor
                            left_pair = (symbols[i - 1], symbols[i])
                            if left_pair in pair_to_indexes:
                                pair_to_indexes[left_pair].discard(i_word)
                                if not pair_to_indexes[left_pair]:
                                    del pair_to_indexes[left_pair]
                            # Set frequency to 0 for left pair
                            heapq.heappushpop()
                            heapq.heappush(heap, (0, left_pair))

                        if i + 2 < len(symbols):  # Update right neighbor
                            right_pair = (symbols[i + 1], symbols[i + 2])
                            if right_pair in pair_to_indexes:
                                pair_to_indexes[right_pair].discard(i_word)
                                if not pair_to_indexes[right_pair]:
                                    del pair_to_indexes[right_pair]
                            # Set frequency to 0 for right pair
                            heapq.heappush(heap, (0, right_pair))

        # Remove the merged pair itself
        pair_to_indexes.pop(pair, None)
        return tokens_list, changed_indexes

    new_pairs = update_pair_to_indexes(tokens_list)
    pair_freq_update_calc(pair_to_indexes, new_pairs)

    # Perform BPE merges
    for i in tqdm(range(num_merges), desc="Performing BPE merges"):
        while heap:
            freq, best = heapq.heappop(heap)
            if best in pair_to_indexes and pair_to_indexes[best]:
                break
        else:
            break

        vocab.add(''.join(best))
        debug_print(f"Step {i + 1}: Merged pair {best} freq {freq} ")
        debug_print("vocab: ", vocab)
        tokens_list, changed_indexes = merge_vocab(best, tokens_list)
        new_pairs = update_pair_to_indexes(tokens_list, changed_indexes)
        pair_freq_update_calc(pair_to_indexes, new_pairs)

        debug_print(f"After merging {best}, tokens_list: {tokens_list}")
        debug_print(f"Updated pair_to_indexes: {dict(pair_to_indexes)}")
        debug_print(f"Updated heap: {heap}","\n----------------------------------------------\n")

    sorted_vocab = sorted(vocab)
    return sorted_vocab


if __name__ == "__main__":
    filename = "test_video.txt"
    N = 8
    vocab = train_bpe(filename, N)
    print("Final Vocabulary Size:", len(vocab))
    print("Sample Vocabulary:", list(vocab))
