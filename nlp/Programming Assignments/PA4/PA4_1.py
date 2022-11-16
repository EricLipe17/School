import os
import numpy as np
import pandas as pd
import scipy


# • Compute the co-occurrence matrix (read: Numpy array) C, such that:
# C[w, c] = the number of (w, c) and (c, w) bigrams in the corpus. Each
# row in the array should represent the number of co-occurrences of w
# with each c.

def load_data(data_set):
    # iterate over documents
    vocab = set()
    lines = list()
    for root, dirs, files in os.walk(data_set):
        for name in files:
            with open(os.path.join(root, name)) as f:
                for line in f:
                    lines.append(line)
                    for word in line.split():
                        vocab.add(word)
    vocab = sorted(list(vocab))
    vocab_dict = {vocab[i]: i for i in range(len(vocab))}
    inv_vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    return vocab, vocab_dict, inv_vocab_dict, lines


def compute_cm(vocab, vocab_dict, lines):
    cm = np.zeros((len(vocab), len(vocab)))
    for word1 in vocab:
        for word2 in vocab:
            if word1 != word2:
                for sent in lines:
                    split = sent.split()
                    w1_indices = [i for i, word in enumerate(split) if word == word1]
                    w2_indices = [i for i, word in enumerate(split) if word == word2]
                    # if word1 in split and word2 in split and abs(split.index(word1) - split.index(word2)) == 1:
                    if len(w1_indices) > 0 and len(w2_indices) > 0:
                        for w1_index, w2_index in zip(w1_indices, w2_indices):
                            if abs(w1_index - w2_index) == 1:
                                cm[vocab_dict[word1], vocab_dict[word2]] += 1
    return cm


# • Multiply your entire matrix by 10 (to pretend that we see these sentences 10 times) and then smooth
# the counts by adding 1 to all cells.

def smooth_cm(c_matrix):
    c_matrix = c_matrix * 10 + 1
    return c_matrix


# Compute the positive pointwise mutual information (PPMI) for each
# word w and context word c:
def compute_ppmi(c_matrix, vocab_dict):
    ppmi = np.zeros(c_matrix.shape)
    c_sum = np.sum(c_matrix)
    word_probs = np.sum(c_matrix, axis=1) / c_sum
    words = vocab_dict.keys()
    for w in words:
        for c in words:
            i = vocab_dict[w]
            j = vocab_dict[c]
            num = c_matrix[i][j] / c_sum
            den = word_probs[i] * word_probs[j]
            arg = np.log(num / den)
            ppmi[i][j] = max(arg, 0)
    return ppmi


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    return abs(scipy.linalg.norm(a) - scipy.linalg.norm(b))


def svd(ppmi):
    return scipy.linalg.svd(ppmi, full_matrices=False)


voc, voc_dict, inv_voc_dict, data_lines = load_data("sim_data")
cm = compute_cm(voc, voc_dict, data_lines)
cm = smooth_cm(cm)
df = pd.DataFrame(data=cm, columns=list(voc_dict.values()))
print(voc_dict)
print()
print(df)
print()
mat_ppmi = compute_ppmi(cm, voc_dict)
print(mat_ppmi)

print()
print()
print()

print(cm[voc_dict["dogs"]])
print(mat_ppmi[voc_dict["dogs"]])

print()
print()
print()

print("women & men:", euclidean_distance(mat_ppmi[voc_dict["women"]], mat_ppmi[voc_dict["men"]]))
print("women & dogs:", euclidean_distance(mat_ppmi[voc_dict["women"]], mat_ppmi[voc_dict["dogs"]]))
print("men & dogs:", euclidean_distance(mat_ppmi[voc_dict["men"]], mat_ppmi[voc_dict["dogs"]]))
print("feed & like:", euclidean_distance(mat_ppmi[voc_dict["feed"]], mat_ppmi[voc_dict["like"]]))
print("feed & bite:", euclidean_distance(mat_ppmi[voc_dict["feed"]], mat_ppmi[voc_dict["bite"]]))
print("like & bite:", euclidean_distance(mat_ppmi[voc_dict["like"]], mat_ppmi[voc_dict["bite"]]))

print()
print()
print()

U, E, Vt = svd(mat_ppmi)
U = np.matrix(U)
E = np.matrix(np.diag(E))
Vt = np.matrix(Vt)
V = Vt.T
reconstructed_ppmi = U @ E @ Vt
print(reconstructed_ppmi)
print(np.allclose(mat_ppmi, reconstructed_ppmi))

print()
print()
print()

reduced_ppmi = mat_ppmi * V[:, 0:3]
print(reduced_ppmi)
print("women & men:", euclidean_distance(reduced_ppmi[voc_dict["women"]], reduced_ppmi[voc_dict["men"]]))
print("women & dogs:", euclidean_distance(reduced_ppmi[voc_dict["women"]], reduced_ppmi[voc_dict["dogs"]]))
print("men & dogs:", euclidean_distance(reduced_ppmi[voc_dict["men"]], reduced_ppmi[voc_dict["dogs"]]))
print("feed & like:", euclidean_distance(reduced_ppmi[voc_dict["feed"]], reduced_ppmi[voc_dict["like"]]))
print("feed & bite:", euclidean_distance(reduced_ppmi[voc_dict["feed"]], reduced_ppmi[voc_dict["bite"]]))
print("like & bite:", euclidean_distance(reduced_ppmi[voc_dict["like"]], reduced_ppmi[voc_dict["bite"]]))