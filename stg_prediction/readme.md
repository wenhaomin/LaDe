## Run

1. Place the raw data CSV file in the `./data` directory.

2. Run the following code to generate the dataset and adjacency metrics:

```bash
    python gen_dataset.py
    python gen_adj.py
```

3. Run different models

Please execute the following commands to run different models:

```bash
    python ./experiments/agcrn/main.py

    python ./experiments/astgcn/main.py

    python ./experiments/gwnet/main.py

    python ./experiments/mtgnn/main.py

    python ./experiments/stgcn/main.py

    python ./experiments/stgncde/main.py

    python ./experiments/dcrnn/main.py
```

# Leaderboard

| Method | MAE         | RMSE        |
|-------|-------------|-------------|
| HA    | 4.63        | 9.91        |
| DCRNN | 3.69 ± 0.09 | 7.08 ± 0.12 |
| STGCN | 3.04 ± 0.02 | 6.42 ± 0.05 |
| GWNET | 3.16 ± 0.06 | 6.56 ± 0.11 |
| ASTGCN | 3.12 ± 0.06 | 6.48 ± 0.14 |
| MTGNN | 3.13 ± 0.04 | 6.51 ± 0.13 |
| AGCRN  | 3.93 ± 0.03 | 7.99 ± 0.08 |
| STGNCDE  | 3.74 ± 0.15 | 7.27 ± 0.16 |