# FMVSC-DCAG

MATLAB implementation of **Fast Multi-view Subspace Clustering with
Discriminative Consensus Anchor Guidance**.

## Requirements

- MATLAB
- Optimization Toolbox (`quadprog`)
- Deep Learning Toolbox (`mapstd`)
- Parallel Computing Toolbox (`parfor`)

## Run the example

Run:

```matlab
demo
```

The example uses the included `3Sources` dataset with `m = 12`, `d = 12`,
`beta = 0.0625`, and `lambda = 0.125`. It reports ACC, NMI, Purity, and
running time.

## Main files

- `demo.m`: example entry point
- `FMVSC_DCAG.m`: optimization algorithm
- `inG.m`: initialization of the anchor-cluster indicator matrix
- `data/3Sources.mat`: example dataset
