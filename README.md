# Validation Diagnostics for SBI-algorithms and Normalizing Flows

This repository includes the implementation of several validation methods:
- `multiPIT`: code for [[Linhart et al. (2022)]](https://arxiv.org/abs/2211.09602), Validation Diagnostics for SBI algorithms based on Normalizing Flows
- `lc2st`: local classifier two-sample tests, code for [[Linhart et al. (2023)]](https://arxiv.org/abs/2306.03580), L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference
- PIT and its local version `localPIT`[[Zhao et al. (2021)]](https://arxiv.org/abs/2102.10473)
- HPD and its local version `lhpd` [[Zhao et al. (2021)]](https://arxiv.org/abs/2102.10473)
- `waldo` [[Masserano et al. (2023)]](https://arxiv.org/abs/2205.15680)
- `c2st` [[Lopez et al. (2016)]](https://arxiv.org/abs/1610.06545) and [[Lee et al. (2018)]](https://arxiv.org/abs/1812.08927)

The `notebooks` folder contains code with numerical illustrations and comparisons of these methods on
- easy toy-examples in 1D, 2D or 3D
- the Jansen and Rit Neural Mass Model (JRNMM) in 4D (includes the code to reproduce the figures in `multiPIT` paper)
including posterior correction with `lc2st`.

Several scripts provide the code to
- evaluate `c2st` gaussian toy-examples as in [[Lee et al. (2018)]](https://arxiv.org/abs/1812.08927): `c2st_emp_power_lee_2018_fig_2.py`
- evaluate `lc2st` (w.r.t. other methods) on sbi benchmark examples from `sbibm` and the JRNMM (reproduces figures in `lc2st` paper)
- perform additional experiments including classifier calibration for `lc2st`
- plot estimated posteriors on `sbibm` examples

See `reproduce_lc2st2023.md` for instructions on how to reproduce experiments from the `lc2st` paper [[Linhart et al. (2023)]](https://arxiv.org/abs/2306.03580).

