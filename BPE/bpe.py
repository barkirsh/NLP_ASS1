import collections
import gzip

def train_bpe(filename, N):
    """
    Trains a BPE vocabulary from a given file.

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

    # Convert words into list of characters with special end token </w>
    vocab = {" ".join(list(word)) + " </w>": count for word, count in collections.Counter(words).items()}

    # Helper dictionaries
    pair_counts = collections.defaultdict(int)  # Tracks frequency of symbol pairs
    vocab_indices = {}  # Tracks where each symbol appears in the vocab

    # Function to get pairs of symbols from a word
    def get_pairs(word):
        """Get consecutive pairs of symbols from a word."""
        symbols = word.split()
        return [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

    # Update pair counts
    def update_pair_counts():
        pair_counts.clear()
        for word, freq in vocab.items():
            pairs = get_pairs(word)
            for pair in pairs:
                pair_counts[pair] += freq

    # Merge the most frequent pair
    def merge_pair(pair):
        """Merge a pair of symbols into one."""
        a, b = pair
        new_symbol = a + b
        for word in list(vocab.keys()):
            if pair in get_pairs(word):
                # Replace pair in the word
                new_word = word.replace(" ".join(pair), new_symbol)
                freq = vocab.pop(word)
                vocab[new_word] = freq

    # Main training loop
    learned_vocab = set()  # Keeps track of learned vocabulary items
    for _ in range(N):
        update_pair_counts()
        if not pair_counts:
            break
        most_frequent = max(pair_counts, key=pair_counts.get)  # Find most frequent pair
        merge_pair(most_frequent)
        learned_vocab.add("".join(most_frequent))

    return list(learned_vocab)


if __name__ == "__main__":
    # Example usage
    filename = "../hebrew.txt.gz"  # Replace with actual file
    N = 30000
    vocab = train_bpe(filename, N)
    print(vocab)
