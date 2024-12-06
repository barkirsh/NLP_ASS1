import collections
import re
import gzip
from tqdm import tqdm


def train_bpe(filename, num_merges):
    # Read and preprocess the file
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        lines = file.read().splitlines()

        # Pre-tokenization: split into characters and add end-of-word marker as its own token
    vocab = collections.Counter(
        ' '.join(list(word) + ['§']) for line in tqdm(lines, desc="Reading and tokenizing lines")
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
        for word in pair_to_words[pair]:
            w_out = p.sub(''.join(pair), word)  # Merge the pair
            v_out[w_out] = v_in[word]
        return v_out

    update_pair_to_words(vocab)




    # Perform BPE merges until the vocabulary size reaches the target
#    while len(vocab) > vocab_size:

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
    filename = "../english.txt.gz"
    N = 30000
    train_bpe(filename, N)
