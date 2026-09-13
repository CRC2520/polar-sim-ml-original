# Statistical method validation

`method.py` implements the fixed-sample bounds, conservative prospective N calculation, and positive-claim gates. `build_design.py` independently checks the immutable development trial schedule using exact fractions, then constructs the versioned design. `simulate.py` runs bounded synthetic distributions only. No script runs Polar Dynamics or generates final experiment seeds.

The current artifacts use QA revision 2. `SIMULATION_RESULTS.v1.json`, `SIMULATION_DESIGN.json.v1`, and `simulate.py.v1` preserve the first statistical QA pass. Independent reviewers requested a negative equivalence-boundary scenario, context range rescaling, and an exact-binomial nominal-error regression gate. Revision 2 adds these checks without changing N, the confidence method, or any model result.

Full validation: 40 cells, 20,000 synthetic replications per cell. CI smoke: 40 cells, 200 replications per cell; no output file by default.

```sh
python -B -m unittest b1.v1_1.statistics.test_statistics -v
python -B -m b1.v1_1.statistics.simulate --smoke
```

Method, simulator, and design SHA-256 values are embedded in `SIMULATION_RESULTS.json`. The protocol JSON separately hashes the result and all frozen development planning inputs. Eleven tests verify the statistical contracts and saved provenance. These are method-validation results, not evidence about the model.
