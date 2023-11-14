## Nitrogen Generators
This repository contains a nitrogen generators real data set, which can be read using the "Read dataset.ipynb" notebook. The following table summarizes the meaning of each variable:

| Variable name | Meaning |
|---|---|
| Timestamp | The current timestamp, expressed as date and time |
| CMS air pressure [bar] | Pressure of the Carbon Molecular Sieves (CMS) |
| Oxygen base concentration | Concentration of the oxygen inside the machine  |
| Nitrogen pressure [bar] | Pressure of the nitrogen inside the machine |
| Oxygen over threshold | Flag if the oxygen concentration is under or over a specific threshold|

The folders PW_* contains different splitting of the dataset based on the desidered prediction window (PW). Each folder contains the train, test and validation split by which the following algorithms are trained and tested:
| Algorithm | Hyperparameters | Batch run script | Single run script |
|---|---|---|---|
| LSTM | loss: F1, type: Unidirectional | run_all_LSTM.py | LSTM.py |
| ConvLSTM | loss: F1, type: Unidirectional | run_all_ConvLSTM.py | ConvLSTM.py |
| Transformer | loss: F1, type: Unidirectional | run_all_Transformer.py | Transformer.py |
| LR | regularization: L1, L2 | run_all_LR.py | LR.py |
| SVM | all | run_all_SVM.py | SVM.py |
| RF | all | run_all_RF.py | RF.py |
