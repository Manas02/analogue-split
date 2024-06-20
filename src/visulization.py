""" Visualization API for Analogue Split """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from split import analogue_split, tri2mat, calculate_fp, GetTanimotoSimMat
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from tqdm import tqdm


def gamma_plot(clfs, X, y, simmat, gammas, cliff_threshold, test_size, random_seed):
    scores = {}
    scores_cliff = {}
    for clf in tqdm(clfs):
        for gamma in gammas:
            df = pd.read_csv("../data/simpd/CHEMBL1267245.csv")
            fps = df["canonical_smiles"].apply(calculate_fp)
            simmat = tri2mat(GetTanimotoSimMat(fps))
            X = np.array(list(fps))
            y = df["active"].values

            X_train, X_test, y_train, y_test, annot = analogue_split(
                X,
                y,
                simmat,
                cliff_fraction=gamma,
                cliff_threshold=cliff_threshold,
                test_size=test_size,
                random_seed=random_seed,
            )

            clf.fit(X_train, y_train)
            scores[gamma] = clf.score(X_test, y_test)
            if annot["cliff"]:
                scores_cliff[gamma] = clf.score(X[annot["cliff"]], y[annot["cliff"]])

        # Extract keys and values from the dictionary
        x_overall = list(scores.keys())
        y_overall = list(scores.values())
        x_cliff = list(scores_cliff.keys())
        y_cliff = list(scores_cliff.values())

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(
            x_cliff,
            y_cliff,
            marker="o",
            linestyle="-",
            # color="b",
            label=f"{clf.__class__()} Activity Cliff",
        )
        plt.plot(
            x_overall,
            y_overall,
            marker="*",
            linestyle="--",
            # color="r",
            label=f"{clf.__class__()} Overall",
        )

        # Add titles and labels
        plt.title(
            "Plot of Effect of Activity Cliff on Model Performance\nAnalogue Split"
        )
        plt.xlabel("Fraction of Activity Cliffs")
        plt.ylabel("Model Performance")
        plt.legend()

    # Show the plot
    plt.show()

    return scores


df = pd.read_csv("../data/simpd/CHEMBL3705542.csv")
fps = df["canonical_smiles"].apply(calculate_fp)
simmat = tri2mat(GetTanimotoSimMat(fps))
X = np.array(list(fps))
y = df["active"].values

clfs = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    SVC(),
]

data = gamma_plot(
    clfs,
    X,
    y,
    simmat,
    gammas=[i / 10 for i in range(10)],
    cliff_threshold=0.5,
    test_size=0.2,
    random_seed=42,
)
