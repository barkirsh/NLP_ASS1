

import collections
import re
import gzip
from tqdm import tqdm


def train_bpe(filename, num_merges):
    # Read and preprocess the file
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        lines = file.read().splitlines()

    # Pre-tokenization: split into characters and add end-of-word marker
    vocab = collections.Counter(
        ' '.join(list(word) +['§']) for line in tqdm(lines, desc="Reading and tokenizing lines")
        for word in line.split()
    )

    pair_to_words = collections.defaultdict(set)

    def update_pair_to_words(vocab):
        pair_to_words.clear()
        for word in tqdm(vocab, desc="Updating pair-to-words mapping"):
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_to_words[pair].add(word)

    def get_stats():
        pairs = collections.defaultdict(int)
        for pair, words in tqdm(pair_to_words.items(), desc="Calculating pair frequencies"):
            for word in words:
                pairs[pair] += vocab[word]
        return pairs

    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in tqdm(v_in, desc=f"Merging pair {pair}"):
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    update_pair_to_words(vocab)

    # Perform BPE merges
    for i in tqdm(range(num_merges), desc="Performing BPE merges"):
        pairs = get_stats()
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        update_pair_to_words(vocab)
        print(f"Step {i + 1}: Merged pair {best}")

    return vocab


if __name__ == "__main__":
    filename = "hebrew.txt.gz"
    N = 30000
    train_bpe(filename, N)

# def train_bpe(filename, N):
#     # Read and preprocess the file
#     with gzip.open(filename, 'rt', encoding='utf-8') as file:
#         lines = file.read().splitlines()
#
#     # Pre-tokenization: split into characters and add end-of-word marker as its own token
#     word_list = [' '.join(list(word) + ['§']) for line in tqdm(lines, desc="Reading and tokenizing lines")
#                  for word in line.split()]
#     vocab = collections.Counter(word_list)
#
#     pair_to_wordsindexes = collections.defaultdict(set)
#
#     def update_pair_to_wordindexes(vocab, word_list):
#         """Update pair-to-word indexes mapping."""
#         pair_to_wordsindexes.clear()
#         for idx, word in enumerate(word_list):
#             symbols = word.split()
#             for i in range(len(symbols) - 1):
#                 pair = (symbols[i], symbols[i + 1])
#                 pair_to_wordsindexes[pair].add(idx)
#
#     def get_stats():
#         """Calculate frequency of pairs in the vocabulary."""
#         pairs = collections.defaultdict(int)
#         for pair, indexes in pair_to_wordsindexes.items():
#             for idx in indexes:
#                 pairs[pair] += vocab[word_list[idx]]
#         return pairs
#
#     def merge_vocab(pair, v_in, word_list):
#         """Merge the most frequent pair in the vocabulary."""
#         v_out = {}
#         bigram = re.escape(' '.join(pair))
#         p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
#         for idx in pair_to_wordsindexes[pair]:
#             word = word_list[idx]
#             merged_word = p.sub(''.join(pair), word)  # Merge the pair
#             word_list[idx] = merged_word  # Update the word list
#             v_out[merged_word] = v_in[word]
#         return v_out
#
#     update_pair_to_wordindexes(vocab, word_list)
#
#     # Perform BPE merges
#     for i in tqdm(range(N), desc="Performing BPE merges"):
#         pairs = get_stats()
#         if not pairs:
#             break
#         best = max(pairs, key=pairs.get)
#         vocab = merge_vocab(best, vocab, word_list)
#         update_pair_to_wordindexes(vocab, word_list)
#         print(f"Step {i + 1}: Merged pair {best}")
#
#     return vocab
#
#
# if __name__ == "__main__":
#     filename = "hebrew.txt.gz"
#     N = 30000
#     train_bpe(filename, N)

#
# def train_bpe(filename, N):
#     # Read and preprocess the file
#     with gzip.open(filename, 'rt', encoding='utf-8') as file:
#         lines = file.read().splitlines()
#
#         # Pre-tokenization: split into characters and add end-of-word marker as its own token
#     vocab = collections.Counter(
#         ' '.join(list(word) + ['§']) for line in tqdm(lines, desc="Reading and tokenizing lines")
#         for word in line.split()
#     )
#
#     pair_to_words = collections.defaultdict(set)
#
#     def update_pair_to_words(vocab):
#         pair_to_words.clear()
#         for word in vocab:
#             symbols = word.split()
#             for i in range(len(symbols) - 1):
#                 pair = (symbols[i], symbols[i + 1])
#                 pair_to_words[pair].add(word)
#
#     def get_stats():
#         pairs = collections.defaultdict(int)
#         for pair, words in pair_to_words.items():
#             for word in words:
#                 pairs[pair] += vocab[word]
#         return pairs
#
#     def merge_vocab(pair, v_in):
#         v_out = {}
#         bigram = re.escape(' '.join(pair))
#         p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
#         for word in pair_to_words[pair]:
#             w_out = p.sub(''.join(pair), word)  # Merge the pair
#             v_out[w_out] = v_in[word]
#         return v_out
#
#     update_pair_to_words(vocab)
#
#     # Perform BPE merges
#     for i in tqdm(range(N), desc="Performing BPE merges"):
#         pairs = get_stats()
#         if not pairs:
#             break
#         best = max(pairs, key=pairs.get)
#         vocab = merge_vocab(best, vocab)
#         update_pair_to_words(vocab)
#         print(f"Step {i + 1}: Merged pair {best}")
#
#     return vocab
#
#
# if __name__ == "__main__":
#     filename = "hebrew.txt.gz"
#     N = 30000
#     train_bpe(filename, N)
