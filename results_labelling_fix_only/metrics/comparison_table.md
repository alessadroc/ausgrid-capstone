| substation                |   horizon_h | family      |   MAE |   RMSE |   MASE |   mae_seed_sd |
|:--------------------------|------------:|:------------|------:|-------:|-------:|--------------:|
| Hurstville North 132_11kV |           1 | Transformer | 0.434 |  0.626 |  0.43  |         0.003 |
| Hurstville North 132_11kV |           1 | LSTM        | 0.452 |  0.642 |  0.447 |         0.006 |
| Hurstville North 132_11kV |           1 | ARIMA       | 0.486 |  0.674 |  0.48  |               |
| Hurstville North 132_11kV |           1 | Best naive  | 0.709 |  0.89  |  0.701 |               |
| Hurstville North 132_11kV |          24 | LSTM        | 0.967 |  1.463 |  0.958 |         0.008 |
| Hurstville North 132_11kV |          24 | Transformer | 0.97  |  1.459 |  0.961 |         0.017 |
| Hurstville North 132_11kV |          24 | Best naive  | 1.153 |  1.784 |  1.139 |               |
| Hurstville North 132_11kV |          24 | ARIMA       | 1.471 |  1.974 |  1.454 |               |
| Telarah 33_11kV           |           1 | Transformer | 0.355 |  0.593 |  0.442 |         0.008 |
| Telarah 33_11kV           |           1 | LSTM        | 0.361 |  0.588 |  0.45  |         0.009 |
| Telarah 33_11kV           |           1 | ARIMA       | 0.419 |  0.647 |  0.524 |               |
| Telarah 33_11kV           |           1 | Best naive  | 0.611 |  0.853 |  0.764 |               |
| Telarah 33_11kV           |          24 | Transformer | 0.787 |  1.249 |  0.98  |         0.002 |
| Telarah 33_11kV           |          24 | LSTM        | 0.805 |  1.276 |  1.003 |         0.006 |
| Telarah 33_11kV           |          24 | Best naive  | 0.924 |  1.553 |  1.155 |               |
| Telarah 33_11kV           |          24 | ARIMA       | 1.052 |  1.447 |  1.316 |               |
| West Gosford 132_11kV     |           1 | LSTM        | 0.73  |  1.121 |  0.318 |         0.005 |
| West Gosford 132_11kV     |           1 | Transformer | 0.736 |  1.13  |  0.321 |         0.051 |
| West Gosford 132_11kV     |           1 | ARIMA       | 0.929 |  1.313 |  0.406 |               |
| West Gosford 132_11kV     |           1 | Best naive  | 1.251 |  1.602 |  0.547 |               |
| West Gosford 132_11kV     |          24 | Transformer | 1.847 |  2.891 |  0.806 |         0.018 |
| West Gosford 132_11kV     |          24 | LSTM        | 1.876 |  2.97  |  0.819 |         0.094 |
| West Gosford 132_11kV     |          24 | Best naive  | 2.489 |  3.91  |  1.088 |               |
| West Gosford 132_11kV     |          24 | ARIMA       | 2.632 |  3.549 |  1.15  |               |