""" Analogue split """

####################################################
# ANALOGUE SPLIT
# ~ Chemically biased parametric data splitting
# -------------------------------------------------
#
# A dataset of molecules can be represented as a
# connected graph often reffered to as molecular
# network. This graph is constructed such that
# molecules are nodes (implying removal of duplicates
# sterioisomers, tautomer etc. depending upon molecular
# standardization methods used) and edges between
# nodes have associated property of similarity
# between molecules (depends upon fingerprint /
# descriptor and distance metrics). Additionally,
# a similarity threshold parameter could be imposed
# on the graph to reduce the number of edges to only
# those with similarity property greater than the
# parameter threshold. We use this molecular network
# to create a biased data split (create test set).
# Guiding principle for biasing the data split is
# molecules in test set that are structurally similar
# to molecules in train set tend to have similar
# activities (labels). We first report whether this
# hypothesis is true / false for a given dataset,
# then, create a range of datasets with varying
# the fraction of activity cliffs and isolated nodes.
# The goal is to get cross validation splits that
# can be informative visually, as we plot from low to high,
# the fraction of activity cliffs and isolated nodes
# against performance metrics and the AUC shows how
# well the model performed across datasplits and
# bins of fractions represents if the model is able
# to interpolate or extrapolate well for the given
# dataset and how generalizable the performance of the
# model truely is in chemically reasonable (biased)
# data splits.
####################################################

# TODO: Check significance of activity cliffs in the data set and log it
# TODO: Answer: What is Interpolation vs Extrapolation in terms of chemistry ?
