# Validation Diagnostics for (conditional) Flows

## Reporoduce NEURIPS 2023 Figures

### Figure 1 
```
python figure1_neurips_2023.py --plot
```

### Figure 2
```
python figure2_neurips_2023.py --task {task_name} --plot
```

Option to add  `--box_plots` to generate boxplots showing the variance over test runs for every observation (and every method).

### Figure 3

1. Global Coverage Test
```
python figure3_neurips_2023.py --global_ct
```

2. Local Coverage Test
```
python figure3_neurips_2023.py --local_ct_gain
```
Option to add  `--pp_plots` to generate local pp-plots for every observation. 

3. Interpretability of L-C2ST (pairplots with predicted probability)
```
python figure3_neurips_2023.py -lct_g -lc2st_i
```
