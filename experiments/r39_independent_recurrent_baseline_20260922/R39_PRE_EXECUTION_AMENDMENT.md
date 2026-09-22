# R39 pre-execution infrastructure amendment

The first R39 workflow attempt failed before model training, development evaluation,
or confirmatory-seed access.

Failure:
`ModuleNotFoundError: No module named 'scipy'`

Cause:
the frozen R38 controller imports the R27 numerical helper, which requires
`scipy.linalg.solve_discrete_are`.

Amendment:
- add `scipy==1.17.0` to the pinned runtime;
- include the installed SciPy version in package identity checks.

Unchanged:
- experiment source logic;
- RecurrentPPO hyperparameters;
- training seeds;
- development evaluation seeds;
- confirmatory evaluation seeds;
- environments;
- all thresholds and adjudication rules.

No development or confirmatory R39 seed was opened before this amendment.
