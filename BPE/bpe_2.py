import collections
from tqdm import tqdm  # ספרייה לניטור התקדמות, אפשר גם בלי
import gzip


def train_bpe_with_indices(filename, N):
    """
    Trains a BPE vocabulary using two dictionaries to track pairs and their indices.

    Args:
        filename (str): Path to the training file.
        N (int): Number of merge operations (size of the vocabulary).

    Returns:
        list: A list of N strings representing the learned vocabulary.
    """
    # Step 1: Read the file and tokenize the input
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        text = file.read()
    words = text.split()

    # Convert words into list of characters with </w>
    vocab = {' '.join(list(word) + ['§']): count for word, count in collections.Counter(words).items()}

    # Dictionaries for pair counts and indices
    pair_counts = collections.defaultdict(int)  # Tracks frequency of pairs
    token_indices = collections.defaultdict(set)  # Tracks indices of each token in the vocab

    # Helper functions
    def get_pairs_and_indices():
        """Update pair counts and token indices based on current vocab."""
        pair_counts.clear()
        token_indices.clear()
        for idx, (word, freq) in enumerate(vocab.items()):
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
                token_indices[pair].add(idx)

    def merge_pair(pair):
        """Merge a pair of symbols into one and update vocab."""
        a, b = pair
        new_symbol = a + b
        new_vocab = {}
        for idx, (word, freq) in enumerate(vocab.items()):
            if idx in token_indices[pair]:
                # Replace the pair in the word
                new_word = word.replace(" ".join(pair), new_symbol)
                new_vocab[new_word] = freq
            else:
                new_vocab[word] = freq
        return new_vocab, new_symbol

    # Main training loop
    learned_vocab = []  # Keeps track of learned vocabulary items
    for i in tqdm(range(N), desc="Training BPE"):
        get_pairs_and_indices()
        if not pair_counts:
            print("No more pairs to merge!")
            break
        most_frequent = max(pair_counts, key=pair_counts.get)  # Find most frequent pair
        vocab, new_symbol = merge_pair(most_frequent)
        learned_vocab.append(new_symbol)

        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{N}: Learned symbol {new_symbol}")

    return learned_vocab


if __name__ == "__main__":
    # Example usage
    filename = "../hebrew.txt.gz"  # Replace with actual file
    N = 30000
    vocab = train_bpe_with_indices(filename, N)
    print(vocab)
