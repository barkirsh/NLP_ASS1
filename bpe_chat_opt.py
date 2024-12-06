import collections
import re
from tqdm import tqdm
import gzip

def train_bpe(filename, num_merges):
    # Read and preprocess the file
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        lines = file.read().splitlines()

    # Tokenize: characters separated by spaces, with '§' as end-of-word marker
    tokens_list = [
        ' '.join(list(word) + ['§'])
        for line in tqdm(lines, desc="Reading and tokenizing lines")
        for word in line.split()
    ]
  #  print("token list:", tokens_list)
    token_frequencies = collections.Counter(tokens_list)
  #  print("token_freq :", token_frequencies)

    # Initialize vocabulary as unique characters
    vocab = set(char for token in tokens_list for char in token.split())
  #  print("initial vocab:", vocab)
    print("init vocab size: ",len(vocab))

    pair_to_indexes = collections.defaultdict(set)

    # Count pair frequency
    def count_freq_in_token(token, pair):
        freq_in_token = 0
        for i in range(len(token) - 1):
            if token[i] == pair[0] and token[i + 1] == pair[1]:
                freq_in_token += 1
        return freq_in_token

    # Calculate frequencies for all pairs
    def get_stats(pair_to_indexes):
    #    print("Calculating pair frequencies...")
        pairs_freq = collections.defaultdict(int)
        for pair, indexes in tqdm(pair_to_indexes.items(), desc="Calculating pair frequencies"):
            for idx in indexes:
                token = tokens_list[idx].split()  # Split the token into symbols
                pair_count = count_freq_in_token(token, pair)
                pairs_freq[pair] += pair_count  # Accumulate the frequency

    #    print("Pair frequencies:", dict(pairs_freq))
        return pairs_freq

    def update_pair_to_indexes(tokens, changed_tokens=None):
    #    print("updating pair_to_indexes...")
     #   if changed_tokens is None:
     #       changed_tokens = tokens
     #   else:
     #       print("changed tokens in updating:", changed_tokens)

        # Clear pair_to_indexes before re-building
        pair_to_indexes.clear()

        for idx, word in tqdm(enumerate(tokens), desc="Updating pair-to-indexes mapping"):
            symbols = word.split()  # Split token into characters
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_to_indexes[pair].add(idx)  # Add the word index to the pair's set

    def merge_vocab(pair, tokens_list):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        changed_tokens = []

        # Merge the pair in the tokens list
        for i_word in pair_to_indexes[pair]:
            word = tokens_list[i_word]
            merged_word = pattern.sub(''.join(pair), word)
            tokens_list[i_word] = merged_word
            if merged_word != word:
                changed_tokens.append(merged_word)

        return tokens_list, changed_tokens

    update_pair_to_indexes(tokens_list)

    # Perform BPE merges
    for i in tqdm(range(num_merges), desc="Performing BPE merges"):
        pairs = get_stats(pair_to_indexes)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab.add(''.join(best))
        print(f"Step {i + 1}: Merged pair {best} freq {pairs[best]} ")
#        print("vocab: ", vocab,"\n----------------------------------------------\n")

        tokens_list, changed_tokens = merge_vocab(best, tokens_list)
        update_pair_to_indexes(tokens_list, changed_tokens)

    # sort vocab by a-b
    sorted_vocab = sorted(vocab)

    return sorted_vocab


if __name__ == "__main__":
    filename = "english.txt.gz"
    N = 30000
    vocab = train_bpe(filename, N)
    print("Final Vocabulary Size:", len(vocab))
    print("Sample Vocabulary:", list(vocab))
