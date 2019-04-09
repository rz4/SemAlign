# SemAlign: Semantic Alignment Using Word Embeddings
Author: Rafael Zamora-Resendiz

## Overview
In this quick project, we explore the use of pre-trained word embeddings for
sequence-based semantic alignment of text. The algorithm used for sequence-alignment
used in this short exploration is inspired by the
[Smith-Waterman algorithm](https://en.wikipedia.org/wiki/Smith–Waterman_algorithm)
commonly used in bioinformatics for local alignment of genetic sequences.

The Smith-Waterman algorithm uses a substitution matrix which encodes whether
specific genes are the same or not. Here, we explore the use of a similarity
matrix between words in two texts instead of a substitution matrix to perform the sequence alignment.
The similarity matrix is calculated by finding the cosine distance between the embedded vectors of the words.

In order to allow for a more fuzzy alignment of the two texts, a gaussian
kernel of a set n-gram size is used to convolve over the sequences in order
to better match inter-word dependencies.

For more information about how the alignment search works take a look at
the project notebook [here]().

## Installation
The following ***Pip*** command can be used to install **SemAlign**:

```
$ pip3 install git+https://github.com/rz4/SemAlign
```

## Demo
The following is a quick demo showing how to use SemAlign to search for
phrases in a text document. Here we load in glove word embeddings and perform
an alignment search on the first chapter of Alice in Wonderland:

```python
from SemAlign import SemAlign

# Paths
text_path = 'data/text/alice_ch1.txt'
embeddings_path = 'data/embeddings/glove.6B.50d.csv'

# Parameters
search_str = 'drop a bunny'
kernel_size = len(search_str) # Fuzzy radius for inter-word dependencies
w = (0.25,0.25) # Higher weights penalized gaps for (text1, text2)
k = 10 # Return top k results

################################################################################

if __name__ == '__main__':

    # Load text file
    with open(text_path, 'r') as f: text_str = f.read().replace('\n', '')

    # Initiate aligner
    aligner = SemAlign(embeddings_path, kernel_size, delimiter=' ')

    # Search for alignments
    print("Searching for:", search_str)
    alignments, scores = aligner.align(text_str, search_str, k, w)

```

Output:
```
Loading Embeddings...
Searching for: drop a bunny
Top  10 :
Match 1 : Score: 0.8383031045242969
['on', 'a', 'little']
['drop', 'a', 'bunny']

Match 2 : Score: 0.819361822670485
['fell', 'on', 'a']
['drop', 'a', 'bunny']

Match 3 : Score: 0.8123755785450688
['down', 'the', 'rabbit']
['drop', 'a', 'bunny']

Match 4 : Score: 0.7911079667163303
['a']
['bunny']

Match 5 : Score: 0.7878661060780158
['the', 'rabbit', '-']
['drop', 'a', 'bunny']

Match 6 : Score: 0.7401924102470062
['from', 'a']
['drop', 'a']

Match 7 : Score: 0.6894943043002824
['i', 'down', 'the']
['drop', 'a', 'bunny']

Match 8 : Score: 0.6846408918275982
['chapter', 'i', 'down']
['drop', 'a', 'bunny']

Match 9 : Score: 0.6532867158379392
['cake', '.']
['drop', 'a']

Match 10 : Score: 0.6489316583896311
['actually', 'took']
['drop', 'a']

```
