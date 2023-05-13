# Validation Diagnostics for (conditional) Flows

Dependencies: 
- python 3.10 conda environment
- sbi
- lampe
- zuko
- sbibm
- tueplots
- seaborn

Make sure to `pip install -e .` within the `valDiags_flows` folder.

## Generate NEURIPS 2023 Figures

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
Option to add  `--pp_plots` to generate local pp-plots for every observation. In that case, the null hypothesis will be computed.

3. Final Figures: 
- Global vs Local 
```
python figure3_neurips_2023.py --plot 
```
- Interpretability of L-C2ST (pairplots with predicted probability)
```
python figure3_neurips_2023.py --plot --lc2st_interpretability
```

