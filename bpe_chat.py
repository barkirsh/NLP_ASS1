# working on this now:
# token is word

# read file into tokens
# count tokens
# we want to create dict pair_indexes of tokens that include pair TODO
# calc pair frequency : some for all words(pair freq in a word * freq of word ) TODO


import collections
import re
import gzip
from tqdm import tqdm


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
    # init vocab with all unique chars
    # vocab = set()
    # for token in tokens_list:
    #     vocab.update(token.split())

    # # dict from pair to all indexes of tokens where tokens include the pair
    # pair_to_words = collections.defaultdict(set)
    # def update_pair_to_words(vocab):
    #     pair_to_words.clear()
    #     for word in tqdm(vocab, desc="Updating pair-to-words mapping"):
    #         symbols = word.split()
    #         for i in range(len(symbols) - 1):
    #             pair = (symbols[i], symbols[i + 1])
    #             pair_to_words[pair].add(word)
    #

    # def gen_pairs(vocab):
    #     pairs = set()
    #     for symbol in vocab:
    #         for symbol2 in vocab:
    #             pair = (symbol, symbol2)
    #             pairs.add(pair)
    #     return pairs

    # Dict from pair to all indexes of tokens where tokens include the pair

    pair_to_indexes = collections.defaultdict(set)
    # Update pair_to_indexes mapping to include pairs from tokens
    def update_pair_to_indexes(tokens_list):
        pair_to_indexes.clear()
        for idx, word in tqdm(enumerate(tokens_list), desc="Updating pair-to-indexes mapping"):
            symbols = word.split()  # Split token into characters
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_to_indexes[pair].add(idx)  # Add the word index to the pair's set

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

        # Iterate through each pair in the pair_to_indexes dictionary
        for pair, indexes in tqdm(pair_to_indexes.items(), desc="Calculating pair frequencies"):
            for i in indexes:
                # Get the frequency of the token in the corpus
                token_freq = token_frequencies[tokens_list[i]]
                # Calculate the frequency of the pair in the token
                pair_count = count_freq_in_token(tokens_list[i], pair)
                # Multiply token frequency by the pair count and add to the total pair frequency
                pairs_freq[pair] += token_freq * pair_count

        return pairs_freq

    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for i_word in tqdm(pair_to_indexes[pair], desc=f"Merging pair {pair}"):
            word = tokens_list[i_word]
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    update_pair_to_indexes(tokens_list)

    # Perform BPE merges
    for i in tqdm(range(num_merges), desc="Performing BPE merges"):
        pairs = get_stats()
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        update_pair_to_indexes(tokens_list)
        print(f"Step {i + 1}: Merged pair {best}")

    return vocab


if __name__ == "__main__":
    filename = "hebrew.txt.gz"
    N = 30000
    train_bpe(filename, N)
