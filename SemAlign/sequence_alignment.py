import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

def apply_sequence_kernel(seq_emb, kernel_radius=1, eplison=0.01):
    '''
    Method applies fuzzy graph kernels to sequence vectors. This is used to naively propagate
    local information of neighbors within a specified kernel radius. All graph kernels
    sum to 1.

    '''
    # Generate sequence distance matrix
    kernels = squareform(pdist(np.expand_dims(np.arange(len(seq_emb)),axis=-1)))

    # Calculate graph kernels with cutoff of epsilon at the kernel_radius.
    kernels = np.exp((kernels*np.log(eplison))/kernel_radius)
    kernels[kernels<eplison] = 0

    # Normalize kernels by dividing by row sums.
    kernels = kernels / np.expand_dims(np.sum(kernels, axis=-1), axis=-1)

    # Updated sequence embeddings using kernel
    seq_emb_prime = np.dot(kernels, seq_emb)

    return seq_emb_prime

def distance_matrix(s1, s2):
    '''
    Method calcualtes cosine distance matrix between two sequences. Returns
    distances between values of -1.0 and 1.0.

    '''
    a = 1 - cdist(s1,s2,'cosine')
    return a

def scoring_matrix(a, wi=1.0, wj=1.0, epsilon=0.01):
    '''
    Method generates scoring matrix used to align sequences. This algorithm is
    inspired by the Smith-Waterman local sequence-alignment algorithm used
    in bioinformatics. Source: https://en.wikipedia.org/wiki/Smith–Waterman_algorithm

    The gap weights are adpatively assigned according to fuzzy graph kernels defined
    by wi, wj and eplison. Gap weights vary from (0.0, 0.0) to (wi, wj) where
    small gaps are closer to 0.0.

    '''
    # Pad distance matrix
    sa = np.pad(a, ((1,0),(1,0)), 'constant', constant_values=0)

    # Calculate gap weight kernels
    dims = a.shape
    wi_ = [wi*np.exp((i*np.log(epsilon))/dims[0]) for i in reversed(range(dims[0]+1))]
    wj_ = [wj*np.exp((j*np.log(epsilon))/dims[1]) for j in reversed(range(dims[1]+1))]

    # Updated scoring matrix according to policy
    for i in range(1,dims[0]+1):
        for j in range(1,dims[1]+1):

            inputs = [(sa[i,j]+sa[i-1,j-1]), # Top Left + Bottom Right
                      np.max(sa[:i,j])-wi_[i-np.argmax(sa[:i,j])], # Max of all previous values in column - column gap weight
                      np.max(sa[i,:j])-wj_[j-np.argmax(sa[i,:j])], # Max of all previous values in row - row gap weight
                      0] # Zero
            sa[i,j] = np.max(inputs)
    return sa

def traceback(sa, k=100):
    '''
    Method preforms traceback path finding on scoring matrix to find first k alignments
    of length greater than 1.

    '''
    # Sort scoring matrix values in descending order; Save coordinates in look up table.
    sorted_args = np.argsort(sa.flatten())[::-1]
    coords = [(i,j) for i in range(sa.shape[0]) for j in range(sa.shape[1])]

    # Perform traceback until all coords have been visted
    tracebacks = []
    seen = []
    route = []
    for ind in sorted_args:
        i, j = coords[ind]

        flag = True
        score = sa[i,j]
        while(flag):

            # Route connects to other traceback
            if (i,j) in seen:
                tracebacks.append([route,(i,j)])
                route = []
                break

            route.append((i,j))
            seen.append((i,j))

            # Route terminates at zero
            if sa[i,j] == 0:
                tracebacks.append([route,[]])
                route = []
                break

            # Select path direction
            kernel = [sa[i-1,j],sa[i,j-1],sa[i-1,j-1],sa[i,j]]
            m = np.argmax(kernel)

            # Move to next gap
            if m == 0:
                # Terminate route if score is less than gap value
                if score > sa[i-1,j]:
                    i -= 1
                    score += sa[i,j]
                else:
                    tracebacks.append([route,[]])
                    route = []
                    break
            elif m==1:
                # Terminate route if score is less than gap value
                if score > sa[i,j-1]:
                    j -= 1
                    score += sa[i,j]
                else:
                    tracebacks.append([route,[]])
                    route = []
                    break

            # Move to next hit
            elif m==2:
                i -= 1
                j -= 1
                score += sa[i,j]
            elif m==3:
                i -= 1
                j -= 1
                score += sa[i,j]

    # Return alignments with length greater than 1 in order as discovered
    if k == None: k = len(tracebacks)
    alignments = []
    for _ in tracebacks:
        if len(_[0]) > 1:
            r = [(i-1,j-1) for i,j in _[0]]
            alignments.append(r[:-1])
        if len(alignments) == k: break

    return alignments

def score_alignment(alignment, s1, s2, k):
    '''
    This method is used to calculate a global score for aligmnets, to sort
    alignments from multiple search queries of the same topic. This is still
    a work in progress, but has shown good prelimanary results on the note example.

    '''
    # Find gaps and hits, and gather feature vectors
    temp_i = []
    temp_j = []
    i = -1
    j = -1
    s1_ = []
    s2_ = []
    for _ in alignment:
        if _[0] != i:
            temp_i.append(1)
            i = _[0]
        else: temp_i.append(0.0)
        if _[1] != j:
            temp_j.append(1)
            j = _[1]
        else: temp_j.append(0.0)
        s1_.append(s1[_[0]])
        s2_.append(s2[_[1]])

    # Calculate similarity score
    mask = np.array(temp_i) * np.array(temp_j)
    similarity = 2 - cdist(s1_,s2_,'cosine').diagonal()
    score = (similarity*mask)/(2*len(alignment)) * (np.sum(mask)/len(s2)) * k * len(s2)

    return score[0]
