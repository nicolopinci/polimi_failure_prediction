## Nitrogen Generators
This repository contains a nitrogen generators real data set which is split in different prediction window (PW) in the folders PW_*. Each folder Each folder contains the train, test and validation split by which the following algorithms are trained and tested.
The following table summarizes the meaning of each variable:

| Variable name | Meaning |
|---|---|
| Timestamp | The current timestamp, expressed as date and time |
| CMS air pressure [bar] | Pressure of the Carbon Molecular Sieves (CMS) |
| Oxygen base concentration | Concentration of the oxygen inside the machine  |
| Nitrogen pressure [bar] | Pressure of the nitrogen inside the machine |
| Oxygen over threshold | Flag if the oxygen concentration is under or over a specific threshold|
| PW_X.h | Flag if in the prediction window with size X there is a fault|  

Notice that in the dataset, the variable 'Oxygen base concentration' is in Italian: 'Concentrazione ossigeno base'.

The following table summarizes the names of the available algorithms scripts, to reproduce the results presented in the paper:

| Algorithm | Hyperparameters | Batch run script | Single run script |
|---|---|---|---|
| LSTM | loss: F1, type: Unidirectional | run_all_LSTM.py | LSTM.py |
| ConvLSTM | loss: F1, type: Unidirectional | run_all_ConvLSTM.py | ConvLSTM.py |
| Transformer | loss: F1, type: Unidirectional | run_all_Transformer.py | Transformer.py |
| LR | regularization: L1, L2 | run_all_LR.py | LR.py |
| SVM | all | run_all_SVM.py | SVM.py |
| RF | all | run_all_RF.py | RF.py |
