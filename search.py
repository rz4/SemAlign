from SemAlign import SemAlign

# Paths
text_path = 'data/text/alice_ch1.txt'
embeddings_path = 'data/embeddings/glove.6B.50d.txt'

# Parameters
search_str = 'drop a bunny'
kernel_size = len(search_str) # Fuzzy radius for inter-word dependencies
w = (0.25,0.25) # Higher weights penalized gaps
k = 10 # Return top k results

################################################################################

if __name__ == '__main__':

    # Load text file
    with open(text_path, 'r') as f: text_str = f.read().replace('\n', '')

    # Initiate aligner
    aligner = SemAlign(embeddings_path, kernel_size)

    # Search for alignments
    print("Searching for:", search_str)
    alignments, scores = aligner.align(text_str, search_str, k, w)
