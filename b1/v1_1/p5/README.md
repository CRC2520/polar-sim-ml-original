# P5 prospective development benchmark

The policy disposition and protocol are at `../P5_C4_DISPOSITION.json` and `../P5_PROTOCOL_V1_1.json`. They preserve the original inventory task and separately name the full-refill ceiling witness. The v1 code and results remain unchanged.

All adaptable arms use the prospectively specified `cfg00` without tuning. Both C2 cycles are retained and averaged. C6 verifies exact actions, complete controller states, observations and conjugated task events. The common resource envelope is read from the v1.1 accounting module and pinned with the policy source. C4 may use fewer resources; it receives no dummy graph work or memory history.

Before collection, publish `P5_SOURCE_FREEZE.json` and its complete source/configuration/resource hash inventory. Pass that published commit to the collection command. The orchestration/CI layer must additionally verify that the supplied commit actually contains the frozen bytes and precedes the results commit; the local runner cannot prove GitHub publication by checking a hexadecimal identifier.

```sh
python -B -m unittest b1.v1_1.p5.test_p5 -v
python -B -m b1.v1_1.p5.benchmark --freeze b1/v1_1/p5/P5_SOURCE_FREEZE.json --source-commit PUBLISHED_SOURCE_COMMIT --output b1/v1_1/p5/results
python -B -m b1.v1_1.p5.verify_results b1/v1_1/p5/results
```

The first command uses only explicit QA fixtures and `PD-B1-D-v1.1-QA`; it is not scientific evidence. The benchmark permits exactly 32 development bundles and 9 controller realizations per bundle, including both C2 cycles and the separate witness. Its namespace is `PD-B1-D-v1.1`. It refuses to overwrite an existing output directory. Every event is retained, with deterministic gzip shards, per-shard hashes, source/configuration/task identity and a manifest. Controller failures retain the episode and receive loss one; hard task or resource-envelope violations fail closed.

The task cannot support positive utility or pairing superiority. Any apparent performance beyond its per-realization physical ceiling is a bug or unfairness signal. Equality is descriptive only. No final seeds, confirmatory execution, P7 retuning or combination with v1 observations is available here.
