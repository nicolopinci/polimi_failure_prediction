import pandas as pd
import os
from datetime import timedelta
import sys
import random


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


# Separate the windows for each class
val_class_0_windows = [window for window, last_class_value in val_result if last_class_value == 0]
val_class_1_windows = [window for window, last_class_value in val_result if last_class_value == 1]

# Determine the minimum number of windows for each class
val_min_windows_per_class = min(len(val_class_0_windows), len(val_class_1_windows))

# Randomly select the required number of windows for each class
val_selected_class_0_windows = random.sample(val_class_0_windows, val_min_windows_per_class)
val_selected_class_1_windows = random.sample(val_class_1_windows, val_min_windows_per_class)

# Combine the randomly selected windows for each class
val_balanced_result = [(window, 0) for window in val_selected_class_0_windows] + [(window, 1) for window in val_selected_class_1_windows]

# Shuffle the balanced_result to randomize the order
random.shuffle(val_balanced_result)


class_0_count = 0
class_1_count = 0

for window, last_class_value in balanced_result:
    if last_class_value == 0:
        class_0_count += 1
    elif last_class_value == 1:
        class_1_count += 1

print("Class 0 count:", class_0_count)
print("Class 1 count:", class_1_count)

val_class_0_count = 0
val_class_1_count = 0

for window, last_class_value in val_balanced_result:
    if last_class_value == 0:
        val_class_0_count += 1
    elif last_class_value == 1:
        val_class_1_count += 1

print("Class 0 count:", val_class_0_count)
print("Class 1 count:", val_class_1_count)

import numpy as np

# Prepare features and labels from the balanced_result list
features = []
labels = []

for window, last_class_value in balanced_result:
    feature_values = window.values
    features.append(feature_values)
    labels.append(last_class_value)

features = np.array(features)
labels = np.array(labels)


# Prepare features and labels from the balanced_result list
val_features = []
val_labels = []

for window, last_class_value in val_balanced_result:
    val_feature_values = window.values
    val_features.append(val_feature_values)
    val_labels.append(last_class_value)

val_features = np.array(val_features)
val_labels = np.array(val_labels)


# Prepare features and labels from the balanced_result list
test_features = []
test_labels = []

for window, last_class_value in test_result:
    test_feature_values = window.values
    test_features.append(test_feature_values)
    test_labels.append(last_class_value)

test_features = np.array(test_features)
test_labels = np.array(test_labels)

import tensorflow as tf
import tensorflow.keras.backend as K

os.environ['TF_DETERMINISTIC_OPS'] = '1'

from tensorflow.keras.layers import Dropout, BatchNormalization, Dense, LSTM


BATCH_SIZE = 1024



global weight_class_1
global weight_class_0

weight_class_1 = (len(class_1_windows) + len(class_0_windows))/len(class_1_windows)
weight_class_0 = (len(class_1_windows) + len(class_0_windows))/len(class_0_windows)

from sklearn.metrics import f1_score
def macro_f1_score(y_true, y_pred):
    def f1_score_func(y_true, y_pred):
        #global weight_class_0
        #global weight_class_1
        
        y_pred_labels = tf.math.round(y_pred)
        y_true_np = y_true.numpy() if isinstance(y_true, tf.Tensor) else y_true
        y_pred_np = y_pred_labels.numpy() if isinstance(y_pred_labels, tf.Tensor) else y_pred_labels
        
        #sample_weights = [weight_class_0 if label == 0 else weight_class_1 for label in y_pred_labels]

        return f1_score(y_true_np, y_pred_np, average='macro') #, sample_weight = sample_weights)
    
    return tf.py_function(f1_score_func, (y_true, y_pred), tf.float32)

import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        # Apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


import tensorflow as tf
from tensorflow.keras import layers
from kerastuner import HyperModel, BayesianOptimization, Objective
from keras.callbacks import EarlyStopping


class MyHyperModel(HyperModel):
    def __init__(self, input_shape, max_len):
        self.input_shape = input_shape
        self.maxlen = max_len
    
    def build(self, hp):
        embed_dim = hp.Int('embed_dim', min_value=16, max_value=128, step=16)
    
        # Ensure `num_heads` is always a divisor of `embed_dim`
        num_heads = hp.Choice('num_heads', values=[i for i in range(1, embed_dim//2 + 1) if embed_dim % i == 0])
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=128, step=32)
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Embedding and Positional Encoding
        x = layers.Dense(embed_dim)(inputs)
        x = PositionalEncoding(self.maxlen, embed_dim)(x)
        
        # Transformer Block
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
        
        # Global Average Pooling layer
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final output layer
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        # Learning Rate Scheduling
        initial_learning_rate = hp.Choice('learning_rate', values=[1e-5, 5e-5, 1e-4, 5e-4,])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[macro_f1_score])
        
        return model

import tensorflow as tf

 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set the CUDA device order (optional)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

 

# Set the GPU to use (0 for GPU 0, 1 for GPU 1, etc.)
gpu_number = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

 

# Initialize TensorFlow session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # Allocate GPU memory as needed
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

 
tf.get_logger().setLevel('ERROR')

# Disable device placement logging
tf.debugging.set_log_device_placement(False)
# Now TensorFlow will only use GPU 0


# Initialize HyperModel
input_shape = (features.shape[1], features.shape[2])# Time steps, 12 features
len_positional_encoding = features.shape[1]
hypermodel = MyHyperModel(input_shape, len_positional_encoding)
objective = Objective("val_macro_f1_score", direction="max")

# Initialize Bayesian Optimization tuner
tuner = BayesianOptimization(
    hypermodel,
    objective=objective,
    max_trials=50,
    overwrite=True,
    directory='bo_tuning_' + folder + "_" + str(RW_minutes),
    project_name='transformer_tuning_new'
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min', restore_best_weights=True)

# Performing hyperparameter tuning
tuner.search(features, labels, 
             validation_data=(val_features, val_labels),
             batch_size=BATCH_SIZE, epochs=500, verbose=1,
             callbacks=[early_stopping])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = tuner.get_best_models(num_models=1)[0]
# Make predictions on the training data
y_train_pred = np.rint(model.predict(features, verbose=0))

# Calculate metrics for the training set
accuracy_train = accuracy_score(labels, y_train_pred)
precision_train = precision_score(labels, y_train_pred)
recall_train = recall_score(labels, y_train_pred)
f1_score_train = f1_score(labels, y_train_pred, average = 'macro')
conf_matrix_train = confusion_matrix(labels, y_train_pred)


# In[42]:


print("Training Set Metrics:")
print("Accuracy:", accuracy_train)
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1 Score:", f1_score_train)
print("Confusion Matrix:")
print(conf_matrix_train)


# In[43]:


# Make predictions on the training data
y_val_pred = np.rint(model.predict(val_features, verbose=0))

# Calculate metrics for the training set
accuracy_train = accuracy_score(val_labels, y_val_pred)
precision_train = precision_score(val_labels, y_val_pred)
recall_train = recall_score(val_labels, y_val_pred)

#sample_weights = np.array([weight_class_0 if label == 0 else weight_class_1 for label in val_labels])

f1_score_train = f1_score(val_labels, y_val_pred, average = 'macro')
conf_matrix_train = confusion_matrix(val_labels, y_val_pred)


# In[44]:


print("Training Set Metrics:")
print("Accuracy:", accuracy_train)
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1 Score:", f1_score_train)
print("Confusion Matrix:")
print(conf_matrix_train)


# In[45]:


# Make predictions on the training data
y_test_pred = np.rint(model.predict(test_features, verbose=0))

# Calculate metrics for the training set
accuracy_train = accuracy_score(test_labels, y_test_pred)
precision_train = precision_score(test_labels, y_test_pred)
recall_train = recall_score(test_labels, y_test_pred)

weight_class_1 = (len(class_1_windows) + len(class_0_windows))/len(class_1_windows)
weight_class_0 = (len(class_1_windows) + len(class_0_windows))/len(class_0_windows)

sample_weights = [weight_class_0 if label == 0 else weight_class_1 for label in test_labels]


f1_score_train = f1_score(test_labels, y_test_pred, average = 'macro', sample_weight=sample_weights)
conf_matrix_train = confusion_matrix(test_labels, y_test_pred)


# In[46]:


print("Training Set Metrics:")
print("Accuracy:", accuracy_train)
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1 Score:", f1_score_train)
print("Confusion Matrix:")
print(conf_matrix_train)





data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Training Set": [accuracy_train, precision_train, recall_train, f1_score_train]
}


# Create a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to CSV
df.to_csv("Transformer_new_" + folder + "_RW_" + str(RW_minutes) + "min.csv", index=False)
