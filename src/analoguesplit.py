"""
Analogue Split

:::parameters:::
    gamma := Fraction of test set comprising of activity cliffs
    omega := Threshold of Similarity to create edges between molecules
    test_size := Fraction of Dataset to be put in test set
    train_size := Fraction of Dataset to be put in train set
    X := Feature vector
    y := Label vector
    debug := Log Details
    smi := list of SMILES
    simmat := Similarity matrix
"""

from loguru import logger
