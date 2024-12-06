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
        ' '.join(list(word) + ['§']) for line in lines
        for word in line.split()
    )

    pair_to_words = collections.defaultdict(set)
    pairs = collections.defaultdict(int)

    def initialize_pairs(vocab):
        """Initializes pair counts and word mappings."""
        pair_to_words.clear()
        pairs.clear()
        for word in vocab:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_to_words[pair].add(word)
                pairs[pair] += vocab[word]

    def merge_and_update(pair, vocab):
        """Merges the best pair in the vocabulary and updates pair counts and word mappings."""
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        updated_words = set()

        for word in pair_to_words[pair]:
            # Merge the pair
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = vocab[word]
            updated_words.add(w_out)

        # Update pair_to_words for new words
        for word in updated_words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                new_pair = (symbols[i], symbols[i + 1])
                pair_to_words[new_pair].add(word)
                pairs[new_pair] += v_out[word]  # Update pair counts

        # Remove old pair from mappings and counts
        for word in pair_to_words[pair]:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                old_pair = (symbols[i], symbols[i + 1])
                pairs[old_pair] -= vocab[word]
                if pairs[old_pair] <= 0:
                    del pairs[old_pair]
        pair_to_words.pop(pair, None)

        return v_out

    # Initialize pairs
    initialize_pairs(vocab)

    # Perform BPE merges
    for i in tqdm(range(num_merges), desc="Performing BPE merges"):
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_and_update(best, vocab)
        print(f"Step {i + 1}: Merged pair {best}")

    return vocab


if __name__ == "__main__":
    filename = "../hebrew.txt.gz"
    N = 30000
    final_vocab = train_bpe(filename, N)
    print("Final Vocabulary:")
    for word, freq in list(final_vocab.items())[:10]:
        print(f"{word}: {freq}")
