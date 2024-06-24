# Analogue Split
Chemically Biased Parametric Data Splitting Method

## Analogues, Activity Cliffs & Molecular Networks
The analogue split method ensures that a fraction $\gamma$ of the test set molecules 
are involved in activity cliff edges. Specifically, if the test set contains $N$ molecules, 
then $int(N \times \gamma)$ molecules will be selected from the activity cliff set. 
An activity cliff set consists of molecules (nodes in a molecular network) that are connected 
by an edge (due to their similarity being above a specified threshold) but have different 
class labels.

However, this requirement may not always be achievable depending on the data set. If 
$\gamma$ exceeds the maximum possible fraction of such molecules in the test set, the
code will issue a warning to the user. In this scenario, the remaining molecules in the test
set will be randomly sampled from the available data points to meet the total required number of 
$N$ molecules.

# Gamma Plot Analysis for Activity Cliffs

This repository implements a method to analyze the performance of a machine learning model as a 
function of the proportion ($\gamma$) of activity cliffs in the test set. This method is useful 
for understanding how the presence of activity cliffs in the test set impacts model performance. 
The resulting analysis is visualized using a gamma plot, where the x-axis represents $\gamma$ 
(ranging from 0 to 1) and the y-axis represents a chosen performance metric (e.g., precision, recall).

## Definitions

- **Activity Cliff**: Molecules (nodes in a molecular network) that are connected by an edge (due 
to similarity above a specified threshold) but have different class labels.
- **Gamma ($\gamma$)**: A parameter representing the fraction of the test set made up of activity cliffs.

## Methodology

1. **Identify Activity Cliff Molecules**: Determine which molecules in the data set are part of activity 
cliffs based on their similarity and class labels.

2. **Generate Test Sets**: For each $\gamma$ value from 0 to 1:
    - Calculate the number of activity cliff molecules needed in the test set as 
    $\text{int}(N \times \gamma)$, where $N$ is the total number of molecules in the test set.
    - If $\gamma = 0$, the test set will contain no activity cliffs (all activity cliffs are 
    in the training set).
    - If $\gamma = 1$ and the number of activity cliff molecules is less than $N$, include all 
    activity cliff molecules in the test set, and fill the remaining spots by randomly sampling 
    from the non-activity cliff molecules.

3. **Evaluate Model Performance**: For each test set generated with different $\gamma$ values, 
    train your model on the remaining data (training set) and evaluate it on the test set. Calculate 
    performance metrics such as precision, recall, etc.

4. **Create Gamma Plot**: Plot the performance metrics against $\gamma$ values. The x-axis represents
    $\gamma$ (ranging from 0 to 1), and the y-axis represents the model performance metric.

This method aims to provide a systematic way to evaluate how the presence of activity cliffs in the test 
set impacts model performance. By generating gamma plots, you can gain insights into the robustness 
and reliability of your model in different scenarios.
