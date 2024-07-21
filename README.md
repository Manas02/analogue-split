# Analogue Split
A Chemically Biased Parametric Data Splitting Method

![](https://github.com/Manas02/analogue-split/raw/main/banner.png?raw=True)

## Overview

The Analogue Split method is designed to analyze and improve the robustness of machine learning models by considering activity cliffs in molecular datasets. Activity cliffs are pairs of similar molecules with significantly different biological activities, which can challenge the performance of predictive models.

This package provides tools to:

1. Ensure a specified fraction of the test set molecules are involved in activity cliffs.
2. Analyze model performance as a function of the proportion of activity cliffs in the test set.
3. Visualize these analyses through gamma plots.

## Installation

You can install the package from PyPI using:
```bash
pip install analoguesplit
```

## Usage

### Parameters
---
- **gamma**: Fraction of the test set comprising of activity cliffs.
- **omega**: Similarity threshold to create edges between molecules.
- **test_size**: Fraction of the dataset to be used as the test set.
- **X**: Feature vector (molecular fingerprints).
- **y**: Label vector (biological activities).

### API
---
#### `func` set_random_seed
Sets a random seed for reproducibility.

```python
def set_random_seed(seed: int) -> None:
```

#### `func` calculate_fp
Calculates molecular fingerprints for a list of molecules.

```python
def calculate_fp(mols: list[Chem.rdchem.Mol], fp: str = "ecfp4") -> np.ndarray:
```

#### `func`  convert_smiles_to_mol
Converts a list of SMILES strings to RDKit molecule objects.

```python
def convert_smiles_to_mol(smis: list[str]) -> list[Chem.rdchem.Mol]:
```

#### `func` calculate_simmat
Calculates the similarity matrix for molecular fingerprints using a specified similarity function.

```python
def calculate_simmat(fps: np.ndarray, similarity_function) -> np.ndarray:
```

#### `func` tanimoto_similarity
Calculates the Tanimoto similarity coefficient between two binary vectors.

```python
def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
```

#### `func` find_activity_cliffs
Identifies activity cliffs in the dataset.

```python
def find_activity_cliffs(fps: np.ndarray, labels: np.ndarray, threshold: float) -> list[tuple[int, int]]:
```

#### `func` analogue_split
Splits the dataset into training and test sets, ensuring a specified fraction of the test set molecules are activity cliffs.

```python
def analogue_split(fps: np.ndarray, labels: np.ndarray, test_size: float, gamma: float, omega: float) -> tuple[np.ndarray, np.ndarray]:
```

#### `func` train_and_evaluate_models
Trains and evaluates models using the analogue split and returns evaluation results.

```python
def train_and_evaluate_models(gammas: list[float], fps: np.ndarray, labels: np.ndarray, models: dict, test_size: float, omega: float) -> dict:
```

#### `func` plot_evaluation_results
Plots evaluation results for different gamma values.

```python
def plot_evaluation_results(results: dict, gammas: list[float], title: str) -> None:
```

## How to use `analoguesplit` ? 

1. **Identify Activity Cliff Molecules**: Determine which molecules are part of activity cliffs based on their similarity and class labels.
2. **Generate Test Sets**: For each gamma value, create test sets with the desired proportion of activity cliff molecules.
3. **Evaluate Model Performance**: Train models on the training set and evaluate them on the test sets, calculating metrics such as accuracy, precision, recall, and F1 score.
4. **Create Gamma Plot**: Visualize the model performance metrics against gamma values to understand the impact of activity cliffs on model robustness.

## Example

Please check [Notebook](https://github.com/Manas02/analogue-split/blob/main/notebook/) to learn how to use `analoguesplit`.

## License

This project is licensed under the MIT License.

## Acknowledgments

This package relies on several excellent Python libraries including RDKit, scikit-learn, NumPy, and Matplotlib.
