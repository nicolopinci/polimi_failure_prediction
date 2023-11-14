# PoliMi - failure prediction
## Wrapper machines
This repository contains a wrapping machine real data set, which can be read using the "Read dataset.ipynb" notebook.
The following table summarizes the meaning of each variable:

| Variable name | Meaning |
|---|---|
| Timestamp | The current timestamp, expressed as date and time |
| Flag roping | Flag indicating whether the roping functionality is active and which thickness is configured |
| Platform Position [°] | Angular position of the platform |
| Platform Motor frequency [HZ] | Frequency of the platform motor |
| Temperature platform drive [°C] | Temperature of the platform device drive  |
| Temperature slave drive [°C] | Temperature of the slave device drive  |
| Temperature hoist drive [°C] | Temperature of the hosit device drive  |
| Tensione totale film [%] | Total film tension |
| Current speed cart [%] | Current speed of the cart, expressed in % of the maximum value |
| Platform motor speed [%] | Speed of the platform motor |
| Lifting motor speed [RPM] | Speed of the lifting motor |
| Platform rotation speed [RPM] | Rotational speed of the platform |
| Slave rotation speed [M/MIN] | Rotational speed of the slave |
| Lifting speed rotation [M/MIN] | Lifting speed rotation |
| session_counter | Number of sessions |
| time_to_failure | Time to the next alert, expressed in seconds |
| alert_11 | Label indicating the presence of a failure (alert 11) |


The following table summarizes the names of the available algorithms scripts, to reproduce the results presented in the paper:

| Algorithm | Hyperparameters | Batch run script | Single run script |
|---|---|---|---|
| LSTM | loss: F1, type: Unidirectional | run_all_LSTM_UnidirectionalF1.py | LSTM_arch52.py |
| LSTM | loss: F1, type: Bidirectional | run_all_LSTM_BidirectionalF1.py | LSTM_archR1.py |
| LSTM | loss: BCE, type: Unidirectional | run_all_LSTM_UnidirectionalBCE.py | LSTM_archBCEUnidirectional.py |
| LSTM | loss: BCE, type: Bidirectional | run_all_LSTM_BidirectionalBCE.py | LSTM_archBCEBidirectional.py |
| LR | regularization: L1 | run_all_LR.py | LR_liblinear.py |
| LR | regularization: L2 | run_all_LR.py | LogisticRegression.py |
| SVC | all | run_all_SVC.py | SVC.py |
| RF | all | run_all_RF.py | RandomForest.py |
| ConvLSTM | _different architectures_ | run_all_ConvLSTM_A<_x_>.py | ConvLSTM_A<_x_>.py |
| Transformer | _different architectures_ | run_all_Transformer_A<_x_>.py | Transformer_A<_x_>.py |
