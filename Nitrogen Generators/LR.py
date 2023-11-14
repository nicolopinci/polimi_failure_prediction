import numpy as np
import pandas as pd
import os
from datetime import timedelta
import sys
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

folder = "PW_" + sys.argv[1] + "h"
RW_minutes = int(sys.argv[2])

train_data = pd.read_csv(os.path.join(folder, "train.csv"))
train_data["timestamp"] = pd.to_datetime(train_data["timestamp"])
train_data.set_index("timestamp", inplace = True)

val_data = pd.read_csv(os.path.join(folder, "val.csv"))
val_data["timestamp"] = pd.to_datetime(val_data["timestamp"])
val_data.set_index("timestamp", inplace = True)

test_data = pd.read_csv(os.path.join(folder, "test.csv"))
test_data["timestamp"] = pd.to_datetime(test_data["timestamp"])
test_data.set_index("timestamp", inplace = True)

def split_into_windows(df, window_size, step):
    start_time = df.index[0]
    end_time = df.index[-1]
    window_end_time = start_time + window_size

    while window_end_time <= end_time:
        window = df.loc[start_time:window_end_time]
        yield window
        start_time += step
        window_end_time += step


window_size = timedelta(minutes=RW_minutes)
step = train_data.index[1] - train_data.index[0]

result = []
for window in split_into_windows(train_data, window_size, step):
    window_features = window.iloc[:, :-1]
    last_class_value = window.iloc[-1][folder]
    result.append((window_features, last_class_value))

val_result = []
for window in split_into_windows(val_data, window_size, step):
    window_features = window.iloc[:, :-1]
    last_class_value = window.iloc[-1][folder]
    val_result.append((window_features, last_class_value))
    
test_result = []
for window in split_into_windows(test_data, window_size, step):
    window_features = window.iloc[:, :-1]
    last_class_value = window.iloc[-1][folder]
    test_result.append((window_features, last_class_value))

random.seed(42)

# Separate the windows for each class
class_0_windows = [window for window, last_class_value in result if last_class_value == 0]
class_1_windows = [window for window, last_class_value in result if last_class_value == 1]

# Determine the minimum number of windows for each class
min_windows_per_class = min(len(class_0_windows), len(class_1_windows))

# Randomly select the required number of windows for each class
selected_class_0_windows = random.sample(class_0_windows, min_windows_per_class)
selected_class_1_windows = random.sample(class_1_windows, min_windows_per_class)

# Combine the randomly selected windows for each class
balanced_result = [(window, 0) for window in selected_class_0_windows] + [(window, 1) for window in selected_class_1_windows]

# Shuffle the balanced_result to randomize the order
random.shuffle(balanced_result)

class_0_count = 0
class_1_count = 0

for window, last_class_value in balanced_result:
    if last_class_value == 0:
        class_0_count += 1
    elif last_class_value == 1:
        class_1_count += 1


# Prepare features and labels from the balanced_result list
features = []
labels = []

for window, last_class_value in balanced_result:
    feature_values = window.values
    features.append(feature_values)
    labels.append(last_class_value)

features = np.array(features)
labels = np.array(labels)


test_features = []
test_labels = []

for window, last_class_value in test_result:
    test_feature_values = window.values
    test_features.append(test_feature_values)
    test_labels.append(last_class_value)

test_features = np.array(test_features)
test_labels = np.array(test_labels)



val_features = []
val_labels = []

for window, last_class_value in val_result:
    val_feature_values = window.values
    val_features.append(val_feature_values)
    val_labels.append(last_class_value)

val_features = np.array(test_features)
val_labels = np.array(test_labels)

original_shape = features.shape
flattened_shape = (original_shape[0], original_shape[1] * original_shape[2])
features = features.reshape(flattened_shape)

val_original_shape = val_features.shape
val_flattened_shape = (val_original_shape[0], val_original_shape[1] * val_original_shape[2])
val_features = val_features.reshape(val_flattened_shape)

test_original_shape = test_features.shape
test_flattened_shape = (test_original_shape[0], test_original_shape[1] * test_original_shape[2])
test_features = test_features.reshape(test_flattened_shape)

best_f1_score, best_model = None, None

weight_class_1 = (len(class_1_windows) + len(class_0_windows))/len(class_1_windows)
weight_class_0 = (len(class_1_windows) + len(class_0_windows))/len(class_0_windows)
sample_weights = [weight_class_0 if label == 0 else weight_class_1 for label in test_labels]

for C in [0.5, 1.0, 1.5]:
    for penalty in ["l1", "l2"]:
        solver = None
        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "l2":
            solver = "lbfgs"
    
        logreg_model = LogisticRegression(max_iter = 20000, random_state = 42 , C = C, penalty = penalty, solver = solver)
        logreg_model.fit(features, labels)

        y_val_pred = logreg_model.predict(val_features)
        accuracy_val = accuracy_score(val_labels, y_val_pred)
        precision_val = precision_score(val_labels, y_val_pred)
        recall_val = recall_score(val_labels, y_val_pred)

        f1_score_val = f1_score(val_labels, y_val_pred, average = 'macro', sample_weight=sample_weights)
        print("Validation F1", f1_score_val)

        if best_f1_score is None:
            best_f1_score = f1_score_val
            best_model = logreg_model
        elif f1_score_val > best_f1_score:
            best_f1_score = f1_score_val
            best_model = logreg_model

y_test_pred = best_model.predict(test_features)

# Calculate metrics for the training set
accuracy_test = accuracy_score(test_labels, y_test_pred)
precision_test = precision_score(test_labels, y_test_pred)
recall_test = recall_score(test_labels, y_test_pred)

f1_score_test = f1_score(test_labels, y_test_pred, average = 'macro', sample_weight=sample_weights)
conf_matrix_test = confusion_matrix(test_labels, y_test_pred)

print("Testing Set Metrics:")
print("Accuracy:", accuracy_test)
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1 Score:", f1_score_test)
print("Confusion Matrix:")
print(conf_matrix_test)


data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Training Set": [accuracy_test, precision_test, recall_test, f1_score_test]
}


# Create a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to CSV
df.to_csv("LR" + "_" + folder + "_RW_" + str(RW_minutes) + "min.csv", index=False)