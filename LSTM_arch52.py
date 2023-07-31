import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense, LSTM
import imblearn
import datetime
import os
import gc

import tensorflow as tf
import tensorflow.keras.backend as K

BATCH_SIZE = 1024

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def build_model_LSTM(input_shape):
    model = tf.keras.Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=tf.keras.regularizers.L2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=32, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001)))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    return model



# parmeter support
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_hours', type=float, default=0.25)
parser.add_argument('--read_minutes', type=int, default=20)
parser.add_argument('--sampling_period', type=int, default=5)
parser.add_argument('--path', type=str, default='./')

# parse parameters: if parameters are not provided, the default values are used
args = parser.parse_args()
HOURS = args.pred_hours
SLIDING_WINDOW_LEN = int(args.read_minutes / args.sampling_period * 60)
SAMPLING_FREQ = args.sampling_period
PATH = args.path

if PATH[-1] != '/':
    PATH += '/'

## GLOBAL VARIABLES
EXPERIMENT_ID = f'PRED_{HOURS}h_READ_{SLIDING_WINDOW_LEN}steps_FREQ_{SAMPLING_FREQ}s_LSTM_arch52'
TTF_CAP = HOURS * 3600 * 2
CLASS_TTF = TTF_CAP / 2
RANDOM_HASH_SEED = 1234


# create experiment path
if not os.path.isdir(f'{PATH}{EXPERIMENT_ID}'):
    os.mkdir(f'{PATH}{EXPERIMENT_ID}')

# create model path
if not os.path.isdir(f'{PATH}{EXPERIMENT_ID}/models'):
    os.mkdir(f'{PATH}{EXPERIMENT_ID}/models')


df_machine = pd.read_parquet(f'{PATH}dataset.parquet').query('time_to_failure != -1').reset_index(drop=True)

relevant_alerts = ['alert_11',]

features = [
    'Timestamp',
#    'Completed Pallets [N°]',
#    'Film breaks [N°]',
    'Flag roping',
    'Platform Position [°]',
    'Platform Motor frequency [HZ]',
    'Temperature platform drive [°C]',
    'Temperature slave drive [°C]',
    'Temperature hoist drive [°C]',
    'Tensione totale film [%]',
    'Current speed cart [%]',
    'Platform motor speed [%]',
    'Lifting motor speed [RPM]',
    'Platform rotation speed [RPM]',
    'Slave rotation speed [M/MIN]',
    'Lifting speed rotation [M/MIN]',
    'session_counter',
    'time_to_failure',
    ] + relevant_alerts

df_machine = df_machine[features]

for col in df_machine.columns[df_machine.columns.str.contains('alert')]:
    df_machine[col] = df_machine[col].astype('int8')


N_FEATURES = df_machine.drop(columns=['time_to_failure', 'session_counter', 'Timestamp'] + relevant_alerts).shape[1]
## create class labels
df_machine['time_to_failure'] = df_machine['time_to_failure'].apply(lambda x: x if x < TTF_CAP else TTF_CAP)
df_machine['time_to_failure_class'] = df_machine['time_to_failure'].apply(lambda x: 1 if x <= CLASS_TTF else 0)

# V3: columns to drop during conversion to sliding window
ignore_columns = ['Timestamp', 'time_to_failure', 'time_to_failure_class'] # session_counter and relevant alerts will be removed later!

# ## convert to sliding window
def split_sequence_X(sequence, n_steps):
    X = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
        
    return np.array(X)

# ## generate the data
def generate_sequence(df, x_file_name, y_file_name):
    ## train set
    X_train_list = []
    y_train_list = []
    i = 0
    for session, sdata in df.groupby('session_counter'):
        if i % 10 == 0:
            print('x', end='')
        i += 1
        X = split_sequence_X(sdata.drop(columns=['session_counter', 'time_to_failure_class'] + relevant_alerts + ignore_columns).values, SLIDING_WINDOW_LEN)
        Y = sdata['time_to_failure_class'].values[SLIDING_WINDOW_LEN:].reshape(-1, 1) # final shape: (number_of_labels, 1) it is used for NN outputs...
        
        if X.ndim == 1:
            continue
        
        X_train_list.append(X)
        y_train_list.append(Y)

    X_train = np.concatenate(X_train_list)
    del X_train_list
    y_train = np.concatenate(y_train_list)
    del y_train_list
    
    np.save(x_file_name, X_train)
    np.save(y_file_name, y_train)
    del X_train
    del y_train

GENERATE_FILES = True
if ('y_machine.npy' not in os.listdir(f'{PATH}')) \
    or ('X_machine.npy' not in os.listdir(f'{PATH}')) \
    or GENERATE_FILES:
    print('GENERATING DATA FILES PREPARE YOUR RAM')
    generate_sequence(df_machine, f'{PATH}{EXPERIMENT_ID}/X_machine.npy', f'{PATH}{EXPERIMENT_ID}/y_machine.npy')


# ## load the data
y_machine = np.load(f'{PATH}{EXPERIMENT_ID}/y_machine.npy')
X_machine = np.load(f'{PATH}{EXPERIMENT_ID}/X_machine.npy')

assert not np.any(np.isnan(y_machine))
assert not np.any(np.isnan(X_machine))

print('input shape:', X_machine.shape)
print('output shape:', y_machine.shape)

print('unique values in y:', np.unique(y_machine, return_counts=True))

y_machine = y_machine.astype('float')


def compute_print_result_tables(results, print_flag=False):
    no_failure = []
    failure = []
    macro_avg = []
    weig_avg = []
    acc = []
    for result in results:
        no_failure.append(result['No failure'])
        failure.append(result['Failure'])
        macro_avg.append(result['macro avg'])
        weig_avg.append(result['weighted avg'])
        acc.append(result['accuracy'])

    no_failure = pd.DataFrame(no_failure)
    failure = pd.DataFrame(failure)
    macro_avg = pd.DataFrame(macro_avg)
    weig_avg = pd.DataFrame(weig_avg)

    if print_flag:
        print('No failure class:')
        print(no_failure.mean())
        print('*-----------------------*')
        print('Failure class:')
        print(failure.mean())
        print('*-----------------------*')
        print('macro average:')
        print(macro_avg.mean())
        print('*-----------------------*')
        print('accuracy average:')
        print(np.mean(acc))
    
    return no_failure.mean(), failure.mean(), macro_avg.mean(), np.mean(acc)
    

def print_output(out):
    with open(f'{PATH}{EXPERIMENT_ID}/output_file.txt', 'a') as f:
        f.write(f'{out}' + '\n')
        print(out)


models = [
    'LSTM'
]

print_output(f'Prediction window length (h): {HOURS}; Reading window length (min): {SLIDING_WINDOW_LEN*SAMPLING_FREQ/60}; Sampling period (s): {SAMPLING_FREQ}')

result_dict = {
    'model': [],
    'fold': [],
    'rus': [],
    'tn': [],
    'fp': [],
    'fn': [],
    'tp': [],
}
for model_name in models:
    print_output('------------------------------------------------------------------------')
    print_output('------------------------------------------------------------------------')
    print_output(f'MODEL {model_name}')
    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    X = X_machine
    y = y_machine
    
    results = []
    fold_no = 1

    ## Train-Test split is here
    for train, test in kfold.split(X, y):
        fold_results = []
        
        # fix small leak
        min_test = min(test)
        max_test = max(test)
        
        first_ind_test = min_test
        last_ind_test = max_test
        index_to_delete_train = np.arange(min_test - SLIDING_WINDOW_LEN, max_test + SLIDING_WINDOW_LEN)
        train = np.sort(list(set(train) - set(index_to_delete_train)))

        print_output('------------------------------------------------------------------------')
        print_output(f'Training for fold {fold_no} ...')
        
        for RUS_i in range(10):
            n_samples = X[train].shape[0] # just to explain the reshaping
            n_steps = X[train].shape[1]
            n_features = X[train].shape[2]
            
            n_samples_test = X[test].shape[0] # just to explain the reshaping
            n_steps_test = X[test].shape[1]
            n_features_test = X[test].shape[2]
            
            try:
                ## create undersample each time
                undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority', random_state=RUS_i)

                print_output(f'Fitting RUS {RUS_i+1} ...')

                ## Random UnderSampling for this fold
                X_train_res, y_train_res = undersample.fit_resample(X[train].reshape(n_samples, n_steps * n_features), y[train])
                X_test_res, y_test_res = undersample.fit_resample(X[test].reshape(n_samples_test, n_steps_test * n_features_test), y[test])
                # now the number of samples changes!! correct this
                # reshape back
                X_train_res.reshape((-1, n_steps, n_features))
                n_samples = X_train_res.shape[0]
                X_test_res.reshape((-1, n_steps_test, n_features_test))
                n_samples_test = X_test_res.shape[0]

                ## Standard Scaling
                scaler = StandardScaler()
                scaler.fit(X_train_res.reshape((n_samples * n_steps, n_features)))

                ### DL ONLY: reshape 3 dimensions
                X_train_res_scale = scaler.transform(X_train_res.reshape((n_samples * n_steps, n_features))).reshape(n_samples, n_steps, n_features)
                X_test_res_scale = scaler.transform(X_test_res.reshape((n_samples_test * n_steps_test, n_features_test))).reshape(n_samples_test, n_steps_test, n_features_test)

                ## Fit Fold, RUS
                ### DL ONLY: early stopping and model checkpoint
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min', restore_best_weights=False)
                model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{PATH}{EXPERIMENT_ID}/best_model_tmp.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=0)

                if model_name == 'LSTM':
                    model = build_model_LSTM((SLIDING_WINDOW_LEN, X_train_res_scale.shape[2]))
                    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
                    model.compile(loss=f1_loss, optimizer=optimizer, metrics=['accuracy'])
                    history = model.fit(X_train_res_scale, y_train_res, 
                                validation_data=(X_test_res_scale, y_test_res),
                                batch_size=BATCH_SIZE, epochs=500, verbose=0,
                            callbacks = [early_stopping,
                                model_checkpoint]
                            )
                    model = tf.keras.models.load_model(f'{PATH}{EXPERIMENT_ID}/best_model_tmp.h5', custom_objects={'f1_loss': f1_loss})
                    model.trainable = False
                else:
                    raise NotImplementedError

                ## Evaluate test and append to fold results
                pred_test = np.rint(model.predict(X_test_res_scale, verbose=0))
                fold_results.append(classification_report(y_test_res, pred_test, labels=[0, 1], target_names=['No failure', 'Failure'], digits=4, output_dict=True, zero_division=1))
                tn, fp, fn, tp = confusion_matrix(y_test_res, pred_test).ravel()
                # fill result_dict
                result_dict['model'].append(model_name)
                result_dict['fold'].append(fold_no)
                result_dict['rus'].append(RUS_i)
                result_dict['tn'].append(tn)
                result_dict['fp'].append(fp)
                result_dict['fn'].append(fn)
                result_dict['tp'].append(tp)
                
                ## TENSORFLOW MEMORY LEAK WORK AROUND
                del model
                gc.collect()
                K.clear_session()
                tf.compat.v1.reset_default_graph()
            except:
                continue
    
        results.append(compute_print_result_tables(fold_results, print_flag=False))
        fold_no += 1
    
    ## print averages for Precision, Recall, F1, Accuracy
    all_no_failure = []
    all_failure = []
    all_macro = []
    all_acc = []
    for r in results:
        no_failure_mean = r[0]
        failure_mean = r[1]
        macro_avg_mean = r[2]
        acc_mean = r[3]

        all_no_failure.append(no_failure_mean)
        all_failure.append(failure_mean)
        all_macro.append(macro_avg_mean)
        all_acc.append(acc_mean)

    all_no_failure = pd.DataFrame(all_no_failure)
    all_failure = pd.DataFrame(all_failure)
    all_macro = pd.DataFrame(all_macro)
    all_acc = np.nanmean(all_acc)

    print_output('No failure class:')
    print_output(all_no_failure.mean())
    print_output('*-----------------------*')
    print_output('Failure class:')
    print_output(all_failure.mean())
    print_output('*-----------------------*')
    print_output('macro average:')
    print_output(all_macro.mean())
    print_output('*-----------------------*')
    print_output('accuracy average:')
    print_output(all_acc)
    print_output('\n')
    
print_output('------------------------------------------------------------------------')
print_output('saving results...')
df_res = pd.DataFrame(result_dict)
df_res.to_csv(f'{PATH}{EXPERIMENT_ID}/results.csv', index=False)

print_output('Experiment finished: removing input data')
os.remove(f'{PATH}{EXPERIMENT_ID}/best_model_tmp.h5')
os.remove(f'{PATH}{EXPERIMENT_ID}/y_machine.npy')
os.remove(f'{PATH}{EXPERIMENT_ID}/X_machine.npy')
