# Programmable-Quantum-Matter
Codes for our paper: "Programmable Quantum Matter: Harnessing Qubit Inhomogeneity in Solid-State Spin Ensembles for Scalable Cluster State Generation"

We have organized the Github repo in terms of the sections in the paper and Supplements. Below we describe the workflow and code used to generate a particular figure in the main text or supplements.

## Section I: SAFE-GRAPE
This section corresponds to folder "/Error-Correcting Pulses_/ which contains the notebook SAFE_GRAPE.ipynb. This code implements the numeric pulse engineering and is used to generate Fig. 2 in the main text.

## Section II: Filter-Function
This section corresponds to folder "/Dynamical Decoupling - Filter Function/" which contains the following codes:

1. Filter_Function_I.ipynb: This notebook takes the numeric values from SAFE-GRAPE optimization and numerically evaluates the Filter function for pulse sequences A, and B. This is used to generate Fig. 3(b-d) in the main text.

2. Filter_Function_II.py: This code is similar to above, and further takes into account a noise-spectrum for SiV. Further, it numberically evaluates the value of chi, T2, Fidelity, z (scaliing factor) for pulse sequences A and B, and stores them in .csv files.

