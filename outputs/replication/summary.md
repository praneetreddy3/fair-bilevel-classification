# Reported results vs. independent rerun on different hardware

| Dataset | Run | EO off | EO on | paired diff | seeds better | p | p (Bonferroni x4) | acc off | acc on |
|---|---|---|---|---|---|---|---|---|---|
| Credit | reported | 0.083 | 0.073 | -0.010 | 6/10 | 0.348 | 1.000 | 0.729 | 0.714 |
| Credit | replication | 0.075 | 0.066 | -0.010 | 6/10 | 0.361 | 1.000 | 0.719 | 0.705 |
| Adult | reported | 0.099 | 0.033 | -0.066 | 8/10 | 0.022 | 0.090 | 0.752 | 0.747 |
| Adult | replication | 0.090 | 0.029 | -0.061 | 9/10 | 0.021 | 0.085 | 0.753 | 0.748 |
| Law | reported | 0.304 | 0.197 | -0.107 | 10/10 | 0.001 | 0.005 | 0.703 | 0.669 |
| Law | replication | 0.306 | 0.224 | -0.082 | 9/10 | 0.015 | 0.061 | 0.703 | 0.666 |
| COMPAS | reported | 0.144 | 0.114 | -0.031 | 9/10 | 0.009 | 0.035 | 0.634 | 0.635 |
| COMPAS | replication | 0.144 | 0.134 | -0.011 | 7/10 | 0.279 | 1.000 | 0.634 | 0.635 |
