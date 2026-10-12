Options: {'float64': True, 'adult_log1p': False}
| Dataset | selected rho_o | Acc | F1 | EO gap | DemP gap | EOD gap | diverged clients (of 50) |
|---|---|---|---|---|---|---|---|
| credit | 0 | 0.756±0.015 | 0.495±0.004 | 0.056±0.014 | 0.094±0.023 | 0.096±0.026 | 0 |
|   credit grid | rho_o=0 | val acc 0.809 | val EO 0.042 | | | | 0 |
|   credit grid | rho_o=10 | val acc 0.806 | val EO 0.063 | | | | 0 |
|   credit grid | rho_o=100 | val acc 0.795 | val EO 0.054 | | | | 0 |
|   credit grid | rho_o=1000 | val acc 0.788 | val EO 0.061 | | | | 0 |
|   credit grid | rho_o=10000 | val acc 0.786 | val EO 0.059 | | | | 0 |
| adult | 10 | 0.788±0.012 | 0.450±0.055 | 0.180±0.119 | 0.046±0.043 | 0.185±0.113 | 2 |
|   adult grid | rho_o=0 | val acc 0.795 | val EO 0.262 | | | | 5 |
|   adult grid | rho_o=10 | val acc 0.791 | val EO 0.166 | | | | 2 |
|   adult grid | rho_o=100 | val acc 0.752 | val EO 0.212 | | | | 3 |
|   adult grid | rho_o=1000 | val acc 0.734 | val EO 0.329 | | | | 2 |
|   adult grid | rho_o=10000 | val acc 0.746 | val EO 0.170 | | | | 1 |
| law | 0 | 0.766±0.022 | 0.852±0.017 | 0.362±0.012 | 0.430±0.013 | 0.362±0.012 | 0 |
|   law grid | rho_o=0 | val acc 0.901 | val EO 0.094 | | | | 0 |
|   law grid | rho_o=10 | val acc 0.891 | val EO 0.005 | | | | 0 |
|   law grid | rho_o=100 | val acc 0.889 | val EO 0.012 | | | | 0 |
|   law grid | rho_o=1000 | val acc 0.887 | val EO 0.016 | | | | 0 |
|   law grid | rho_o=10000 | val acc 0.887 | val EO 0.016 | | | | 0 |
| compas | 10 | 0.640±0.005 | 0.609±0.017 | 0.060±0.024 | 0.039±0.030 | 0.093±0.036 | 0 |
|   compas grid | rho_o=0 | val acc 0.677 | val EO 0.282 | | | | 0 |
|   compas grid | rho_o=10 | val acc 0.665 | val EO 0.076 | | | | 0 |
|   compas grid | rho_o=100 | val acc 0.623 | val EO 0.071 | | | | 0 |
|   compas grid | rho_o=1000 | val acc 0.597 | val EO 0.077 | | | | 0 |
|   compas grid | rho_o=10000 | val acc 0.590 | val EO 0.081 | | | | 0 |
