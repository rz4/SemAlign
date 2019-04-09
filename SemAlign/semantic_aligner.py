import numpy as np
from .preprocessing import *
from .sequence_alignment import *

class SemAlign(object):

    def __init__(self, embeddings_path, kernel_size=1, verbose=True):
        self.lookup = load_embeddings(embeddings_path)
        self.kernel_size = kernel_size
        self.verbose = verbose

    def align(self, text_str1, text_str2, k=1, w=(0.25,0.25)):

        # Preprocess strings
        tokens_list = preprocess_strs([text_str1, text_str2])

        # Gather embeddings for tokens in each string
        token_embeddings = [get_embeddings(_, self.lookup) for _ in tokens_list]

        # Search for sequence alignments for each search str along text file
        all_alignments = []
        alignment_scores = []

        # Apply sequence kernels of radius len(search_phrase) to search phrase and text
        text1 = apply_sequence_kernel(token_embeddings[0], self.kernel_size)
        text2 = apply_sequence_kernel(token_embeddings[1], self.kernel_size)

        # Calculate cosine similarity between search phrase and text
        cos_dist = distance_matrix(text1, text2)

        # Calculate scoring matrix for sequence alignment
        score = scoring_matrix(cos_dist, wi=w[0], wj=w[1])

        # Find first k alignments of len > 1
        alignments = traceback(score, k=None)
        for j, _ in enumerate(alignments):
            all_alignments.append(_)
            alignment_scores.append(score_alignment(_, token_embeddings[0], text2, 1-(j/len(alignments))))

        # Sort
        sorted_scores = np.argsort(alignment_scores)[::-1]

        # Display results
        if self.verbose:
            if k>1: print("Top ", k,':')
            for i in range(k):
                alignment = all_alignments[sorted_scores[i]]
                ss1 = []
                ss2 = []
                l = -1
                j = -1
                for _ in reversed(alignment):
                    if _[0] != l:
                        ss1.append(tokens_list[0][_[0]])
                        l = _[0]
                    else: ss1.append('GAP')
                    if _[1] != j:
                        ss2.append(tokens_list[1][_[1]])
                        j = _[1]
                    else: ss2.append('GAP')
                print('Match', i+1, ':', 'Score:',alignment_scores[sorted_scores[i]])
                print(ss1)
                print(ss2,'\n')

        # Compile Top results
        alignments = np.array(alignments)
        alignment_scores = np.array(alignment_scores)
        top_alignments = alignments[sorted_scores[:k].astype('int')]
        top_scores = alignment_scores[sorted_scores[:k].astype('int')]

        return top_alignments, top_scores
