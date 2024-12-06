import collections
import re
import gzip
from tqdm import tqdm
import heapq


def train_bpe(filename, num_merges):
    # Read and preprocess the file
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        lines = file.read().splitlines()

    # Extract formatted tokens: characters separated by spaces, plus end-of-word marker
    tokens_list = [
        ' '.join(list(word) + ['§'])  # Convert word into spaced characters and append '§'
        for line in tqdm(lines, desc="Reading and tokenizing lines")
        for word in line.split()
    ]
    token_frequencies = collections.Counter(tokens_list)
    vocab = set()  # init vocab with all unique chars
    for token in tokens_list:
        vocab.update(token.split())

    # Dict from pair to all indexes of tokens where tokens include the pair
    pair_to_indexes = collections.defaultdict(set)

    # Update pair_to_indexes mapping to include pairs from tokens
    def update_pair_to_indexes(tokens_list, changed_tokens=None):
        if changed_tokens is None:
            changed_tokens = tokens_list  # Update all words if no specific changes
        for word in tqdm(changed_tokens, desc="Updating pair-to-indexes mapping"):
            symbols = word.split()  # Split token into characters
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_to_indexes[pair].add(word)  # Add the word to the pair's set

    # Function to count the frequency of a pair in a token
    def count_freq_in_token(token, pair):
        count = 0
        for i in range(len(token) - 1):  # Loop over token's characters
            if token[i] == pair[0] and token[i + 1] == pair[1]:
                count += 1
        return count

    # Function to calculate the frequencies of each pair
    def get_stats():
        pairs_freq = collections.defaultdict(int)
        # Create a max-heap to store pairs with their frequencies
        heap = []

        # Iterate through each pair in the pair_to_indexes dictionary
        for pair, words in tqdm(pair_to_indexes.items(), desc="Calculating pair frequencies"):
            for word in words:
                # Get the frequency of the token in the corpus
                token_freq = token_frequencies[word]
                # Calculate the frequency of the pair in the token
                pair_count = count_freq_in_token(word, pair)
                # Multiply token frequency by the pair count and add to the total pair frequency
                pairs_freq[pair] += token_freq * pair_count

        # Push pairs and their frequencies into the heap
        for pair, freq in pairs_freq.items():
            heapq.heappush(heap, (-freq, pair))  # Use negative frequency for max-heap behavior

        return heap

    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        changed_tokens = []  # To track which tokens were changed
        for word in tqdm(pair_to_indexes[pair], desc=f"Merging pair {pair}"):
            word_out = p.sub(''.join(pair), word)
            v_out[word_out] = v_in[word]
            if word_out != word:  # If the word changed after merging
                changed_tokens.append(word_out)
        return v_out, changed_tokens

    update_pair_to_indexes(tokens_list)

    # Perform BPE merges
    for i in tqdm(range(num_merges), desc="Performing BPE merges"):
        heap = get_stats()
        if not heap:
            break
        # Extract the most frequent pair from the heap
        freq, best = heapq.heappop(heap)
        best = tuple(best)  # Convert back to tuple
        vocab, changed_tokens = merge_vocab(best, vocab)
        update_pair_to_indexes(tokens_list, changed_tokens)  # Update only for changed tokens
        print(f"Step {i + 1}: Merged pair {best}")

    return vocab


if __name__ == "__main__":
    filename = "english.txt.gz"
    N = 30000
    train_bpe(filename, N)
