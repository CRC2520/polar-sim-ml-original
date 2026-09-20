# R11 pilot choices allowed before freeze
Only these hyperparameters may be selected from pilots:
- adapter forgetting: {0.97, 0.985, 0.995}
- adapter L1 soft-threshold: {0.0, 0.0001, 0.0005}
- minimum action samples before adapter use: {3, 5, 8}
- reliability margin (adapter EMA vs base EMA): {0.0, 0.002, 0.005}
- adapter score weight: {0.40, 0.60, 0.75}
- risk floor: {0.20, 0.25, 0.30}
Selection objective on pilots only: maximize the median of standardized E4 post-adaptation return and E5 counterfactual accuracy,
subject to no pilot alive fraction below 0.60 in the R11 base condition.
No confirmatory threshold may change.
