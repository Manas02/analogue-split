"""
Analogue Split
--------------

:: Parameters ::
threshold := defaults to 0; is used to convert similarity matrix into adjacency matrix
sim_mat := similarity matrix or adjacency matrix if passed with a threshold
gamma := fraction of molecule from total set of activity pairs within range [0, 1]
"""

import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataManip.Metric import GetTanimotoSimMat


def calculate_fp(smi):
    """
    Calculate Morgan fingerprint for a SMILES string
    :param smi: SMILES string
    :return: 2048 bit ECFP4 as ExplicitBitVect
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except:
        return None


def _validate_data(X, y, sim_mat, gamma, test_ratio):
    assert 0 <= gamma <= 1, (
        "Gamma (Fraction of Activity Cliff in"
        f"Test set should be in range [0, 1]) was given {gamma}"
    )
    assert sim_mat.shape[0] == sim_mat.shape[1], (
        "Similarity matrix should be a " "symmetrical square matrix"
    )
    assert len(X) == len(
        y
    ), "Number of Molecules should be equal to number of label entries"
    assert 0 < test_ratio <= 1, "Test ratio has to be in range (0,1]"


def tri2mat(tri_arr):
    n = len(tri_arr)
    m = int((np.sqrt(1 + 4 * 2 * n) + 1) / 2)
    arr = np.ones([m, m])
    for i in range(m):
        for j in range(i):
            arr[i][j] = tri_arr[i + j - 1]
            arr[j][i] = tri_arr[i + j - 1]
    return arr


def analogue_split(
    X, y, similarity_matrix, cliff_threshold, cliff_fraction, test_size, random_seed
):
    """
    Analogue Split
    """
    random.seed(random_seed)

    _validate_data(X, y, similarity_matrix, cliff_fraction, test_size)
    # Initialize a set to store activity cliffs
    activity_cliff_pairs = set()
    test_set = set()
    train_set = set(range(len(y)))
    annot = {}
    # Number of molecules
    n = len(y)

    num_test = int(test_size * len(y))

    # Iterate over all pairs of molecules
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] > cliff_threshold and y[i] != y[j]:
                activity_cliff_pairs.add(frozenset((i, j)))

    # Calculate the number of cliffs to return based on the fraction
    num_cliffs_to_return = int(cliff_fraction * len(activity_cliff_pairs))

    # Sample the required number of cliffs
    sampled_cliffs = random.sample(list(activity_cliff_pairs), num_cliffs_to_return)
    train_set_without_cliffs = list(
        train_set.difference(
            {item for subset in activity_cliff_pairs for item in subset}
        )
    )

    for pair in sampled_cliffs:
        pair = list(pair)
        selected = random.choice([0, 1])
        if pair[selected] not in test_set:
            test_set.add(pair[selected])
        else:
            test_set.add(pair[selected - 1])

    assert num_cliffs_to_return == len(
        test_set
    ), f"Unexpected number of activity cliff nodes selected [Expected {num_cliffs_to_return} Found {len(test_set)}]"

    more_test = num_test - len(test_set)
    if more_test > 0:
        rest_of_test_set = set(random.sample(train_set_without_cliffs, k=more_test))
        annot["cliff"] = list(test_set)
        annot["noncliff"] = list(rest_of_test_set)
        test_set = test_set.union(rest_of_test_set)
    else:
        test_set = set(random.sample(list(test_set), k=num_test))
        annot["cliff"] = list(test_set)
        annot["noncliff"] = []

    test_list = list(test_set)
    train_list = list(train_set.difference(test_set))

    return X[train_list], X[test_list], y[train_list], y[test_list], annot
