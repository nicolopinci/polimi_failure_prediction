## Blood Refrigerators
This repository contains a blood refrigerator real data set which is split in different prediction window (PW) in the folders PW_*. Each folder Each folder contains the train, test and validation split by which the following algorithms are trained and tested.
The following table summarizes the meaning of each variable:

| Variable name | Meaning |
|---|---|
| Timestamp | The current timestamp, expressed as date and time |
| Product Temperature Base [°C] | Temperature of the product inserted into the fridge |
| Evaporator Temperature Base [°C] | Temperature of the evaporator inside the refrigerator  |
| Condenser Temperature Base [°C] | Temperature of the condenser inside the refrigerator |
| Power Supply [V] | Power necessary to let the refrigerator works  |
| Instant Power Consumtpion [W] | Current power consumption of the refrigerator  |
| Signal [dBm] | Current level of power expressed in dBm |
| Door alert | Binary value that indicates if there is something wrong in the door of the fridge |
| Door close | Binary value that indicates if there door of the refrigerator is close (1) or no (0) |
| Door open | Binary value that indicates if there door of the refrigerator is open (1) or no (0) |
| Machine cooling | Binary value to indicate if the refrigerator is in the status 'cooling' |
| Machine defrost | Binary value to indicate if the refrigerator is in the status 'defrost' |
| Machine pause | Binary value to indicate if the refrigerator is in the status 'pause'|
| PW_Xh | Flag if in the prediction window with size X there is a fault|

Notice that the variable 'Evaporator Temperature Base [°C]' in the dataset is reported as 'Evaportator Temperature Base [°C]'.


The following table summarizes the names of the available algorithms scripts, to reproduce the results presented in the paper:

| Algorithm | Batch run script | Single run script |
|---|---|---|
| LSTM | run_all_LSTM.py | LSTM.py |
| ConvLSTM | run_all_ConvLSTM.py | ConvLSTM.py |
| Transformer | run_all_Transformer.py | Transformer.py |
| LR | run_all_LR.py | LR.py |
| SVM | run_all_SVM.py | SVM.py |
| RF | run_all_RF.py | RF.py |

