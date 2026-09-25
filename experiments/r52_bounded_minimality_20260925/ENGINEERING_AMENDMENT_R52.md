# R52 engineering amendment — integrity job environment

Canonical scientific run:

`36191159816`

Scientific head SHA:

`79a1de4d43147c041e0b606656e8b7769c8cfd59`

## Scientific execution status

Engineering, development and confirmatory jobs completed successfully.

Development:
- seeds 2226001–2226016;
- 16/16 seed guards;
- `R52_DEVELOPMENT_AUTHORIZE_CONFIRM`.

Confirmation:
- seeds 2227001–2227032;
- 32/32 seed guards;
- `R52_BOUNDED_DCR_NECESSITY_PASS`.

## Integrity-job defect

The original integrity job failed before running `verify_r52.py`.

Cause:

- integrity job used the runner's default Python 3.12;
- it attempted to install frozen `numpy==1.23.5`;
- that NumPy version does not provide the required Python 3.12 wheel and its
  legacy build path fails with `pkgutil.ImpImporter` missing.

This is a CI environment defect only. It occurred after scientific development
and confirmation artifacts were already immutable.

## Repair rule

The repair audit must:

1. download exactly the development and confirmatory artifacts from canonical
   run `36191159816`;
2. use Python 3.10;
3. run the unchanged frozen `verify_r52.py`;
4. perform no environment execution, no scientific seed generation and no
   endpoint/threshold modification.

No R52 scientific run may be re-selected or pooled with the canonical run.
