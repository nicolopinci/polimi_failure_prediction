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
