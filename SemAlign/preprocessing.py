import numpy as np
import re

def preprocess_strs(str_list):
    '''
    Method lower cases all characters, adds spaces between alphanumerical characters
    and non-alphanumerical, and tokenizes along spaces for all strings in list.

    '''
    token_list = []
    for str_ in str_list:
        s = str_.lower()
        s = re.sub(r'([0-9a-zA-Z])([^0-9a-zA-Z])', r'\1 \2',s)
        s = re.sub(r'([^0-9a-zA-Z])([0-9a-zA-Z])', r'\1 \2',s)
        s = re.sub(r'([^0-9a-zA-Z])([^0-9a-zA-Z])', r'\1 \2 ',s)
        token_list.append(s.split())
    return token_list

def load_embeddings(path):
    '''
    Method loads embeddings into dictionary look-up table. Embeddings vectors
    are stored as numpy arrays.

    '''
    embeddings = {}
    print("Loading Embeddings...")
    with open(path, 'r') as f:
        for _ in f:
            row = _.split()
            embeddings[str(row[0])] = np.array(row[1:],dtype=float)
    return embeddings

def get_embeddings(str_tokens, embeddings):
    '''
    Method retrieves embedding vectors for each token in a tokenized string
    using an embedding look up table.
    Returns numpy array of emedded vectors.

    '''
    v = []
    for _ in str_tokens:
        if str(_) in embeddings: v.append(embeddings[_])
        else: v.append(np.ones(embeddings['.'].shape))
    return np.array(v)
