# Validation Diagnostics for SBI-algorithms and Normalizing Flows
Official code for L-C2ST: Local Diagnostics for Posterior Approximations in
Simulation-Based Inference

Dependencies: 
- python 3.10 conda environment
- sbi
- lampe
- zuko
- sbibm
- tueplots
- seaborn

Make sure to `pip install -e .` within the `valDiags_flows` folder.

## 1. Generate Figures from paper

### Figure 1: Single-class evaluation
```
python figure1_lc2st_2023.py --plot
```

### Figure 2: Method comparison on SBIBM tasks
```
python figure2_lc2st_2023.py --task {task_name} --plot
```
For `task_name = two_moons` or `slcp`.

### Figure 3: Global vs. Local Coverage Test (JRNMM)
```
python figure3_lc2st_2023.py --plot
```

### Figure 4: Interpretability of L-C2ST (graphical diagnostics for JRNMM)
```
python figure3_lc2st_2023.py --plot --lc2st_interpretability
```

## 2. Reproduce experiment results from paper

### Results for Figure 1
```
python figure1_lc2st_2023.py --opt_bayes --t_shift
```
```
python figure1_lc2st_2023.py --power_shift
```
### Results for Figure 2
1. Varying N_train (Columns 1 and 2):
```
python figure2_lc2st_2023.py --t_res_ntrain --n_train 100 1000 10000 100000
```
```
python figure2_lc2st_2023.py --t_res_ntrain --n_train 100 1000 10000 100000 --power_ntrain
```
2. Varing N_cal (Columns 3 and 4):
```
python figure2_lc2st_2023.py --power_ncal --n_cal 100 500 1000 2000 5000 10000
```
By default this will compute results for the Two Moons task. For the SLCP task add `--task slcp`.

### Results of Appendix A.5 (Runtimes)
```
python figure2_lc2st_2023.py --runtime -nt 0 --n_cal 5000 10000 --task slcp
```

### Results for Figures 3 and 4:
1. Global Coverage results (left panel of Figure 3):
```
python figure3_lc2st_2023.py --global_ct
```
2. Local Coverage results for varying gain (right panel of Figure 4 and Figure 4):
```
python figure3_lc2st_2023.py --local_ct_gain
```


