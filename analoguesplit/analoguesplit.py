"""
Analogue Split

:::parameters:::
    gamma := Fraction of test set comprising of activity cliffs
    omega := Threshold of Similarity to create edges between molecules
    test_size := Fraction of Dataset to be put in test set
    X := Feature vector
    y := Label vector
"""

import random

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def set_random_seed(seed: int) -> None:
    """
    Sets a random seed for random and numpy
    """
    logger.debug(f"Setting Random Seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)


def calculate_fp(mols: list[Chem.rdchem.Mol], fp: str = "ecfp4") -> np.ndarray:
    """
    Calculate molecular fingerprints for a list of molecules.

    Parameters:
    mols (list[Chem.rdchem.Mol]): List of RDKit molecule objects.
    fp (str): Fingerprint type to calculate. Supported values are
              "ecfp4", "ecfp4_chiral", "ecfp6", "ecfp6_chiral",
              "atom_pair", "topological_torsion", and "rdkit".

    Returns:
    np.ndarray: Array of calculated fingerprints.
    """
    fp2fn = {
        "ecfp4": rdFingerprintGenerator.GetMorganGenerator(
            radius=2, includeChirality=False
        ).GetFingerprintAsNumPy,
        "ecfp4_chiral": rdFingerprintGenerator.GetMorganGenerator(
            radius=2, includeChirality=True
        ).GetFingerprintAsNumPy,
        "ecfp6": rdFingerprintGenerator.GetMorganGenerator(
            radius=3, includeChirality=False
        ).GetFingerprintAsNumPy,
        "ecfp6_chiral": rdFingerprintGenerator.GetMorganGenerator(
            radius=3, includeChirality=True
        ).GetFingerprintAsNumPy,
        "atom_pair": rdFingerprintGenerator.GetAtomPairGenerator().GetFingerprintAsNumPy,
        "topological_torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator().GetFingerprintAsNumPy,
        "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator().GetFingerprintAsNumPy,
    }

    if fp not in fp2fn:
        raise ValueError(f"Unsupported fingerprint type: {fp}")

    fingerprint_fn = fp2fn[fp]
    fingerprints = [fingerprint_fn(mol) for mol in mols]

    return np.array(fingerprints)


def convert_smiles_to_mol(smis: list[str]) -> Chem.rdchem.Mol:
    """
    Convert a list of SMILES strings to RDKit molecule objects.

    Parameters:
    smis (list[str]): List of SMILES strings to convert.

    Returns:
    list[Chem.rdchem.Mol]: List of RDKit molecule objects converted from SMILES strings.
    """
    return [Chem.MolFromSmiles(smi) for smi in smis]


def calculate_simmat(fps: np.ndarray, similarity_function) -> np.ndarray:
    """
    Calculate the lower triangle of a similarity matrix using a specified similarity function
    and uses that to construct a symmetrical matrix [Assuming similarity fn is symmetrical].

    Parameters:
    fps (np.ndarray): Input data array of shape (n_mols, fp_shape).
    similarity_function (callable): Function that computes similarity between two data points.

    Returns:
    np.ndarray: similarity matrix, a 2D array
    """
    n_mols = fps.shape[0]
    similarity_matrix = np.zeros((n_mols, n_mols))

    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            similarity_matrix[i, j] = similarity_function(fps[i], fps[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]

    np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Calculate the Tanimoto similarity coefficient between two binary vectors.

    Parameters:
    fp1 (np.ndarray): Bit vector 1.
    fp2 (np.ndarray): Bit vector 2.

    Returns:
    float: Tanimoto similarity coefficient.
    """
    intersection = np.count_nonzero(fp1 & fp2)
    union = np.count_nonzero(fp1 | fp2)

    if union == 0:
        return 0.0  # Handle edge case where both vectors are empty

    return intersection / union


def find_activity_cliffs(
    fps: np.ndarray, labels: np.ndarray, threshold: float
) -> list[tuple[int, int]]:
    """
    Identify activity cliffs in the dataset.

    Parameters:
    fps (np.ndarray): Array of molecular fingerprints.
    labels (np.ndarray): Array of activity labels.
    threshold (float): Similarity threshold to consider a pair as an activity cliff.

    Returns:
    list[tuple[int, int]]: List of pairs of indices representing activity cliffs.
    """
    n_mols = fps.shape[0]
    cliffs = []

    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            similarity = tanimoto_similarity(fps[i], fps[j])
            if similarity >= threshold and labels[i] != labels[j]:
                cliffs.append((i, j))

    return cliffs


def analogue_split(
    fps: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    gamma: float,
    omega: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform analogue split to create training and test sets with a fraction of test set molecules involved in activity cliffs.

    Parameters:
    fps (np.ndarray): Array of molecular fingerprints.
    labels (np.ndarray): Array of activity labels.
    test_size (float): Fraction of molecules in the test set.
    gamma (float): Fraction of test set molecules to be involved in activity cliffs.
    omega (float): Similarity threshold to consider a pair as an activity cliff.
    random_seed (int): Seed the PRNG

    Returns:
    tuple[np.ndarray, np.ndarray]: Indices of training and test set molecules.
    """
    # Set random seed for reproducibility
    set_random_seed(random_seed)

    activity_cliffs = find_activity_cliffs(fps, labels, omega)
    num_test_size = int(len(fps) * test_size)
    n_cliff_molecules = int(num_test_size * gamma)

    # Collect all activity cliff molecules
    all_activity_cliff_molecules: set[int] = set()
    for i, j in activity_cliffs:
        all_activity_cliff_molecules.add(i)
        all_activity_cliff_molecules.add(j)

    # Check if we have enough activity cliff molecules
    if len(all_activity_cliff_molecules) >= n_cliff_molecules:
        test_cliff_molecules = random.sample(
            list(all_activity_cliff_molecules), n_cliff_molecules
        )
    else:
        logger.critical(
            "The desired fraction of molecules involved in activity cliffs cannot be achieved."
            "Sampling remaining molecules randomly."
        )
        test_cliff_molecules = list(all_activity_cliff_molecules)

    # Ensure remaining test set does not include any activity cliff molecules
    remaining_test_size = num_test_size - len(test_cliff_molecules)
    all_indices = set(range(len(fps)))
    non_cliff_molecules = all_indices - all_activity_cliff_molecules
    if len(non_cliff_molecules) <= remaining_test_size:
        remaining_test_size = (
            num_test_size - len(test_cliff_molecules) - len(non_cliff_molecules)
        )
        logger.critical(
            f"Test set will have {remaining_test_size} more than expected"
            " number of activity cliff molecules.\nConsider increasing omega."
        )

        all_without_test_cliff_and_non_cliff = (
            all_indices - non_cliff_molecules - set(test_cliff_molecules)
        )
        test_cliff_and_non_cliff_molecules = non_cliff_molecules
        test_cliff_and_non_cliff_molecules = random.sample(
            list(all_without_test_cliff_and_non_cliff), k=remaining_test_size
        )

        test_indices = test_cliff_molecules + test_cliff_and_non_cliff_molecules

    else:
        test_non_cliff_molecules = random.sample(
            list(non_cliff_molecules), k=remaining_test_size
        )
        test_indices = test_cliff_molecules + test_non_cliff_molecules

    train_indices = list(all_indices - set(test_indices))
    return np.array(train_indices), np.array(test_indices)


def train_and_evaluate_models(
    gammas: list[float],
    fps: np.ndarray,
    labels: np.ndarray,
    models: dict,
    test_size: float,
    omega: float,
    random_seed: int,
) -> dict:
    """
    Train and evaluate models using analogue split and return evaluation results.

    Parameters:
    gammas (list[float]): List of Fraction of test set molecules to be involved in activity cliffs.
    fps (np.ndarray): Array of molecular fingerprints.
    labels (np.ndarray): Array of activity labels.
    models (dict): Dict of Model name as key and model object
    test_size (float): Fraction of molecules in the test set.
    omega (float): Similarity threshold to consider a pair as an activity cliff.
    random_seed (int): Seed the RNG

    Returns:
    dict: Dictionary with key as "gamma_model_name" and value as list of evaluation results.
    """

    # Dictionary to store the evaluation results
    eval_results = {}

    for gamma in gammas:
        # Perform the analogue split
        train_indices, test_indices = analogue_split(
            fps, labels, test_size, gamma, omega, random_seed
        )

        # Split the data into training and test sets
        X_train, X_test = fps[train_indices], fps[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]

        # Train and evaluate each model
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="binary")
            recall = recall_score(y_test, y_pred, average="binary")
            f1 = f1_score(y_test, y_pred, average="binary")

            # Store the results in the dictionary
            eval_results[f"{gamma}_{model_name}"] = [accuracy, precision, recall, f1]

    return eval_results


def plot_evaluation_results(results: dict, gammas: list[float], title: str) -> None:
    """
    Plot evaluation results for different gamma values.

    Parameters:
    results (dict): Dictionary with key as "gamma_model_name" and value as list of evaluation results.
    gammas (list[float]): List of gamma values used in the evaluation.
    title (str): Title of the entire figure.
    """
    # Initialize dictionaries to store evaluation metrics for each model and gamma
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_metrics = {
        metric: {model: [] for model in set(k.split("_", 1)[1] for k in results.keys())}
        for metric in metrics
    }

    # Populate the metrics for each model and gamma
    for gamma in gammas:
        for model in model_metrics["accuracy"].keys():
            key = f"{gamma}_{model}"
            if key in results:
                for i, metric in enumerate(metrics):
                    model_metrics[metric][model].append(results[key][i])
            else:
                for metric in metrics:
                    model_metrics[metric][model].append(None)

    # Plot the metrics for each model
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    axs = axs.ravel()

    for idx, metric in enumerate(metrics):
        for model, values in model_metrics[metric].items():
            axs[idx].plot(gammas, values, label=model)

        axs[idx].set_title(f"{metric}")
        axs[idx].set_xlabel("$\gamma$ (fraction of test set made of activity cliffs)")
        axs[idx].set_ylabel(metric)
        axs[idx].set_xticks(gammas)
        axs[idx].tick_params(axis="x", rotation=45)
        axs[idx].set_ylim(0, 1)
        axs[idx].legend(ncol=2, fontsize="x-small")
        axs[idx].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
