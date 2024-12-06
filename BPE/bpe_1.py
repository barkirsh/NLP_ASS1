import re
from collections import Counter, defaultdict
from tqdm import tqdm  # For the progress bar
import gzip


def train_bpe(filename, n):
    """
    Train a BPE vocabulary from a file.

    Args:
        filename (str): Path to the training text file.
        n (int): Number of subwords to include in the vocabulary.

    Returns:
        List[str]: List of subwords in the learned vocabulary.
    """
    # Read file and tokenize
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        text = file.read()
    tokens = [token + ' ' for token in text.split()]

    # Count initial token frequencies
    token_counts = Counter(tokens)

    # Initialize mappings
    pair_freq = defaultdict(int)
    subword_map = {token: list(token) for token in token_counts}

    # Function to calculate pair frequencies
    def update_pair_freq(token_counts, subword_map):
        pair_freq.clear()
        for token, count in token_counts.items():
            subwords = subword_map[token]
            for i in range(len(subwords) - 1):
                pair_freq[(subwords[i], subwords[i + 1])] += count

    # Perform N merges with a progress bar
    vocab = set()
    with tqdm(total=n, desc="Training BPE", unit="merge") as pbar:
        for _ in range(n):
            update_pair_freq(token_counts, subword_map)
            if not pair_freq:
                break
            # Find the most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            vocab.add(''.join(best_pair))

            # Update token decompositions
            for token in list(subword_map):
                if best_pair in zip(subword_map[token], subword_map[token][1:]):
                    new_token = []
                    skip = False
                    for i in range(len(subword_map[token])):
                        if skip:
                            skip = False
                            continue
                        if i < len(subword_map[token]) - 1 and \
                                (subword_map[token][i], subword_map[token][i + 1]) == best_pair:
                            new_token.append(''.join(best_pair))
                            skip = True
                        else:
                            new_token.append(subword_map[token][i])
                    subword_map[token] = new_token

            # Recompute token frequencies
            token_counts = Counter([''.join(subword_map[token]) for token in token_counts])

            # Update progress bar
            pbar.update(1)

    return list(vocab)


def main():
    filename = "../english.txt.gz"
    n = 30000
    vocab = train_bpe(filename, n)
    print(f"Learned Vocabulary: {vocab[:100]}")


if __name__ == "__main__":
    main()
