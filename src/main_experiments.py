import copy
import warnings
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PowerTransformer
from src.mlp import RegressionMLP, RegressionMLPNS
import os
import time


def preprocess_data(dataset_name, measure):
    def read_input_file(filename):
        d = []
        # Open the file for reading
        with open(filename, 'r') as file:
            for line in file:

                # Split line
                line = line.strip()
                content, mi_value, at_value = line.split(',', 2)

                # Remove the parentheses and leading/trailing whitespace from the content
                content = content.strip()[2:-1]

                # Select value and adapt content
                if measure == 'MI':
                    value = mi_value
                    # assumption: max. number of atoms = 6
                    content = content[:-14]
                else:
                    value = at_value

                # Convert the value to a float
                value = float(value.strip()[0:-1])

                # Append the parsed data as a tuple to the list
                d.append((content, value))
        return d

    # Sample dataset (replace this with your actual dataset)
    # Each row is a set of propositional logic formulas, and the last column is the numerical value.
    data = read_input_file(dataset_name)

    # Split the dataset into features (X) and labels (y)
    X_text = [row[0] for row in data]  # Formulas as text
    y = [row[1] for row in data]  # Labels (numerical values)

    # Define a custom tokenizer that splits based on spaces
    def custom_tokenizer(text):
        return text.split()

    # Use CountVectorizer to convert the text data into numerical features
    vectorizer = CountVectorizer(binary=True,
                                 tokenizer=custom_tokenizer,
                                 token_pattern=None)  # Use binary encoding (1 for presence, 0 for absence)
    X = vectorizer.fit_transform(X_text)
    X = torch.Tensor(X.toarray())
    y = torch.Tensor(y)

    return X, y


def evaluate(X, y, seed, hpos, hpo, num_feat, data_set_name, mode, measure):
    """
    Method starting the evaluation pipeline
    """

    k = 10
    idx = 0
    sk = KFold(n_splits=k, random_state=seed, shuffle=True)
    mse_train, mse_val, mse_test = list(), list(), list()
    mae_train, mae_val, mae_test = list(), list(), list()
    r2_train, r2_val, r2_test = list(), list(), list()
    training_time, inference_time, total_time = list(), list(), list()
    data_set_name = data_set_name[28:]
    target = measure

    for train, test in sk.split(X=X, y=y):

        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1111, random_state=seed)  # 0.1111

        """
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})

        # Plot Histogram on x
        plt.hist(y, bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
        plt.show()
        """

        scaler = PowerTransformer().fit(torch.reshape(y_train, (-1, 1)))
        y_train = scaler.transform(torch.reshape(y_train, (-1, 1)))
        y_val = scaler.transform(torch.reshape(y_val, (-1, 1)))
        y_test = scaler.transform(torch.reshape(y_test, (-1, 1)))

        y_train = torch.tensor(y_train)
        y_val = torch.tensor(y_val)
        y_test = torch.tensor(y_test)

        best_val_score = np.inf
        best_paras = {}
        best_model = ''

        training_start_time = time.time()

        if "mlp" in mode:

            epochs = 1000
            patients = 10

            if hpo:
                for learning_rate in hpos["mlp"]["learning_rate"]:
                    for weight_decay in hpos["mlp"]["weight_decay"]:
                        for hidden_size in hpos["mlp"]["hidden_size"]:

                            if "mlp_flags_model" in mode:
                                model = RegressionMLPNS(X_train.shape[1], hidden_size, 1, target)

                                model.fit(X_train, X_train[:, num_feat:], y_train, X_val, X_val[:, num_feat:], y_val,
                                          patients=patients, epochs=epochs,
                                          learning_rate=learning_rate, weight_decay=weight_decay)
                            else:
                                model = RegressionMLP(X_train.shape[1], hidden_size, 1)
                                model.fit(X_train, y_train, X_val, y_val,
                                          patients=patients, epochs=epochs,
                                          learning_rate=learning_rate, weight_decay=weight_decay)

                            if "mlp_flags_model" in mode:
                                y_pred = model(X_val, X_val[:, num_feat:])
                                y_val = y_val.detach().numpy()
                                y_pred = y_pred.detach().numpy()
                                val_mae = mean_absolute_error(y_true=y_val, y_pred=y_pred)  #mean_squared_error(y_true=y_val, y_pred=y_pred) #
                                y_val = torch.tensor(y_val)
                            else:
                                y_pred = model(X_val)
                                y_val = y_val.detach().numpy()
                                y_pred = y_pred.detach().numpy()
                                val_mae = mean_absolute_error(y_true=y_val, y_pred=y_pred)  # mean_squared_error(y_true=y_val, y_pred=y_pred) #
                                y_val = torch.tensor(y_val)

                            if val_mae < best_val_score:
                                best_val_score = val_mae
                                best_model = model
                                best_paras = {"learning_rate": learning_rate, "weight_decay": weight_decay,
                                              "hidden_size": hidden_size}
                            model = best_model

            else:
                learning_rate = 0.005
                weight_decay = 0.05
                hidden_size = 512

                if "mlp_flags_model" in mode:

                    model = RegressionMLPNS(X_train.shape[1], hidden_size, 1, target)
                    model.fit(X_train, X_train[:, num_feat:], y_train, X_val, X_val[:, num_feat:], y_val,
                              patients=patients, epochs=epochs,
                              learning_rate=learning_rate, weight_decay=weight_decay)
                else:
                    model = RegressionMLP(X_train.shape[1], hidden_size, 1)
                    model.fit(X_train, y_train, X_val, y_val,
                              patients=patients, epochs=epochs,
                              learning_rate=learning_rate, weight_decay=weight_decay)

                best_paras = {"learning_rate": learning_rate, "weight_decay": weight_decay, "hidden_size": hidden_size}

        elif 'lr' in mode:
            if hpo:

                model = LinearRegression().fit(X_train, y_train)
                val_mae =  mean_absolute_error(y_true=y_val, y_pred=model.predict(X_val))  #mean_squared_error(y_true=y_val, y_pred=model.predict(X_val))

                if val_mae < best_val_score:
                    best_val_score = val_mae
                    best_model = model
                    best_paras = {}
                model = best_model
            else:

                model = LinearRegression().fit(X_train, y_train)
                best_paras = {}

        elif 'lasso' in mode:
            if hpo:
                for reg_strength in hpos["lasso"]["reg_strength"]:

                    model = Lasso(alpha=reg_strength, random_state=seed).fit(X_train, y_train)
                    val_mae = mean_absolute_error(y_true=y_val, y_pred=model.predict(X_val))  # mean_squared_error(y_true=y_val, y_pred=model.predict(X_val))  #

                    if val_mae < best_val_score:
                        best_val_score = val_mae
                        best_model = model
                        best_paras = {"alpha": reg_strength}
                    model = best_model
            else:

                model = Lasso(alpha=1, random_state=seed).fit(X_train, y_train)
                best_paras = {"alpha": 1}

        elif 'ridge' in mode:
            if hpo:
                for reg_strength in hpos["ridge"]["reg_strength"]:

                    model = Ridge(alpha=reg_strength, random_state=seed).fit(X_train, y_train)
                    val_mae = mean_absolute_error(y_true=y_val, y_pred=model.predict(X_val))  #mean_squared_error(y_true=y_val, y_pred=model.predict(X_val))  #

                    if val_mae < best_val_score:
                        best_val_score = val_mae
                        best_model = model
                        best_paras = {"alpha": reg_strength}
                    model = best_model
            else:
                model = Ridge(alpha=1, random_state=seed).fit(X_train, y_train)
                best_paras = {"alpha": 1}

        training_time.append(time.time() - training_start_time)

        print(f'Best params: {best_paras}')

        with open(f'../res/hpo/results_{measure}_{data_set_name}_{mode}_hpos.csv', 'a+') as fd:
            fd.write(f'{best_paras}\n')

        pickle.dump(model, open(f'../model/{mode}_{idx}.sav', 'wb'))

        inference_start_time = time.time()
        if "mlp" in mode:
            model.eval()
            with torch.no_grad():

                if "mlp_flags_model" in mode:
                    y_train_pred = [p.item() for p in model(X_train, X_train[:, num_feat:])]
                    y_val_pred = [p.item() for p in model(X_val, X_val[:, num_feat:])]
                    y_test_pred = [p.item() for p in model(X_test, X_test[:, num_feat:])]
                else:
                    y_train_pred = [p.item() for p in model(X_train)]
                    y_val_pred = [p.item() for p in model(X_val)]
                    y_test_pred = [p.item() for p in model(X_test)]
        else:
            y_train_pred = [p for p in model.predict(X_train)]
            y_val_pred = [p for p in model.predict(X_val)]
            y_test_pred = [p for p in model.predict(X_test)]
        inference_time.append(time.time() - inference_start_time)
        total_time.append(training_time[idx] + inference_time[idx])

        idx = idx + 1

        mse_train.append(mean_squared_error(y_train, y_train_pred))
        mse_val.append(mean_squared_error(y_val, y_val_pred))
        mse_test.append(mean_squared_error(y_test, y_test_pred))
        mae_train.append(mean_absolute_error(y_train, y_train_pred))
        mae_val.append(mean_absolute_error(y_val, y_val_pred))
        mae_test.append(mean_absolute_error(y_test, y_test_pred))
        r2_train.append(r2_score(y_train, y_train_pred))
        r2_val.append(r2_score(y_val, y_val_pred))
        r2_test.append(r2_score(y_test, y_test_pred))

    print(f'mse train -- mean: {sum(mse_train) / len(mse_train)} -- sd: {np.std(mse_train)} -- values: {mse_train}')
    print(f'mse val -- mean: {sum(mse_val) / len(mse_val)} -- sd: {np.std(mse_val)} -- values: {mse_val}')
    print(f'mse test -- mean: {sum(mse_test) / len(mse_test)} -- sd: {np.std(mse_test)} -- values: {mse_test}')

    print(f'mae train -- mean: {sum(mae_train) / len(mae_train)} -- sd: {np.std(mae_train)} -- values: {mae_train}')
    print(f'mae val -- mean: {sum(mae_val) / len(mae_val)} -- sd: {np.std(mae_val)} -- values: {mae_train}')
    print(f'mae test -- mean: {sum(mae_test) / len(mae_test)} -- sd: {np.std(mae_test)} -- values: {mae_test}')

    print(f'r2 train -- mean: {sum(r2_train) / len(r2_train)} -- sd: {np.std(r2_train)} -- values: {r2_train}')
    print(f'r2 val -- mean: {sum(r2_val) / len(r2_val)} -- sd: {np.std(r2_val)} -- values: {r2_val}')
    print(f'r2 test -- mean: {sum(r2_test) / len(r2_test)} -- sd: {np.std(r2_test)} -- values: {r2_test}')

    print(f'training time -- mean: {sum(training_time) / len(training_time)} -- sd: {np.std(training_time)} -- values: {training_time}')
    print(f'inference time -- mean: {sum(inference_time) / len(inference_time)} -- sd: {np.std(inference_time)} -- values: {inference_time}')
    print(f'total time -- mean: {sum(total_time) / len(total_time)} -- sd: {np.std(total_time)} -- values: {total_time}')

    if not os.path.exists(f'../res/results_{measure}_{data_set_name}.csv'):
        with open(f'../res/results_{measure}_{data_set_name}.csv', 'w') as fd:
            fd.write('algorithm; seed; tuned; '
                 'mse train avg; mse train std; mse train values; mse val avg; mse val std; mse val values; mse test avg; mse test std; mse test values; '
                 'mae train avg; mae train std; mae train values; mae val avg; mae val std; mae val values; mae test avg; mae test std; mae test values; '
                 'r2 train avg; r2 train std; r2 train values; r2 val avg; r2 val std; r2 val values; r2 test avg; r2 test std; r2 test values; '
                 'training time avg; training time std; training time values;  '
                 'inference time avg; inference time std; inference time values; '
                 'total time avg; total time std; total time values '
                 '\n')

    with open(f'../res/results_{measure}_{data_set_name}.csv', 'a') as fd:
        fd.write(
            f'{mode};{seed};{hpo};'
            f'{sum(mse_train) / len(mse_train)}; {np.std(mse_train)}; {mse_train};'
            f'{sum(mse_val) / len(mse_val)}; {np.std(mse_val)}; {mse_val};'
            f'{sum(mse_test) / len(mse_test)}; {np.std(mse_test)}; {mse_test};'
            f'{sum(mae_train) / len(mae_train)}; {np.std(mae_train)}; {mae_train};'
            f'{sum(mae_val) / len(mae_val)}; {np.std(mae_val)}; {mae_val};'
            f'{sum(mae_test) / len(mae_test)}; {np.std(mae_test)}; {mae_test};'
            f'{sum(r2_train) / len(r2_train)}; {np.std(r2_train)}; {r2_train};'
            f'{sum(r2_val) / len(r2_val)}; {np.std(r2_val)}; {r2_val};'
            f'{sum(r2_test) / len(r2_test)}; {np.std(r2_test)}; {r2_test};'
            f'{sum(training_time) / len(training_time)}; {np.std(training_time)}; {training_time};'
            f'{sum(inference_time) / len(inference_time)}; {np.std(inference_time)}; {inference_time};'
            f'{sum(total_time) / len(total_time)}; {np.std(total_time)}; {total_time};'
            f'\n')


if __name__ == "__main__":

    hpo = True
    seed = 42
    # "lr_flags_data", "lr_no_flags", "ridge_flags_data", "ridge_no_flags", "lasso_flags_data", "lasso_no_flags"
    # "mlp_no_flags", "mlp_flags_data", "mlp_flags_model"
    # "lr_flags_data", "lr_no_flags", "ridge_flags_data", "ridge_no_flags", "lasso_flags_data", "lasso_no_flags"
    modes = ["lr_flags_data", "lr_no_flags", "ridge_flags_data", "ridge_no_flags", "lasso_flags_data", "lasso_no_flags", "mlp_no_flags", "mlp_flags_data", "mlp_flags_model"]
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    data_set_names = [
        "../datasets/datasets_2025_3/1T-kbs__max-9-atoms__max-15-elements__with-flags_V2.txt",
    ]

    """
    data_set_names = [
        "../datasets/datasets_2025_2/1T-kbs__max-3-atoms__max-5-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-3-atoms__max-10-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-3-atoms__max-15-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-6-atoms__max-5-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-6-atoms__max-10-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-6-atoms__max-15-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-9-atoms__max-5-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-9-atoms__max-10-elements__with-flags.txt",
        "../datasets/datasets_2025_2/1T-kbs__max-9-atoms__max-15-elements__with-flags.txt",
    ]
    """

    """
        data_set_names = [
        "../datasets/datasets_2025/1T-kbs__max-3-atoms__max-5-elements__with-flags.txt",
        "../datasets/datasets_2025/1T-kbs__max-3-atoms__max-5-elements__with-flags_HIGH_CONJUNCTION.txt",
        "../datasets/datasets_2025/1T-kbs__max-3-atoms__max-10-elements__with-flags.txt",
        "../datasets/datasets_2025/1T-kbs__max-3-atoms__max-10-elements__with-flags_HIGH_CONJUNCTION.txt",
        "../datasets/datasets_2025/1T-kbs__max-6-atoms__max-5-elements__with-flags.txt",
        "../datasets/datasets_2025/1T-kbs__max-6-atoms__max-5-elements__with-flags_HIGH_CONJUNCTION.txt"
    ]
    """

    """
    data_set_names = [
        "../datasets/AT-measure/1T-kbs__AT-measure__max-3-atoms__max-5-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-3-atoms__max-10-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-3-atoms__max-15-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-6-atoms__max-5-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-6-atoms__max-10-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-6-atoms__max-15-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-9-atoms__max-5-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-9-atoms__max-10-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/AT-measure/1T-kbs__AT-measure__max-9-atoms__max-15-elements_with-Consistency-and-UB-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-3-atoms__max-5-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-3-atoms__max-10-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-3-atoms__max-15-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-6-atoms__max-5-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-6-atoms__max-10-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-6-atoms__max-15-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-9-atoms__max-5-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-9-atoms__max-10-elements_with-Consistency-Flag.txt",
        "../datasets/MI-measure/1T-kbs__MI-measure__max-9-atoms__max-15-elements_with-Consistency-Flag.txt",
    ]
    """

    hpos = {
        "mlp": {"learning_rate": [0.001, 0.002, 0.003], "weight_decay": [0.01, 0.03, 0.05], "hidden_size": [32, 64, 128]},
        #"mlp": {"learning_rate": [0.002], "weight_decay": [0.05], "hidden_size": [128]},
        #"mlp": {"learning_rate": [0.005], "weight_decay": [0.05], "hidden_size": [32]},
        #"mlp": {"learning_rate": [0.003, 0.005], "weight_decay": [0.01, 0.05], "hidden_size": [256, 512]},
        #"mlp": {"learning_rate": [0.005], "weight_decay": [0.05], "hidden_size": [512]},
        #"mlp": {"learning_rate": [0.005], "weight_decay": [0.05], "hidden_size": [512]},
        #"mlp": {"learning_rate": [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        #        "weight_decay": [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        #        "hidden_size": [2, 4, 8, 16, 32, 64, 128, 256]},
        "lr": {},
        "ridge": {"reg_strength": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4]},  # default 1.0
        "lasso": {"reg_strength": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4]}  # default 1.0
    }

    for data_set_name in data_set_names:
        for mode in modes:
            for measure in ["MI", "AT"]:

                X, y = preprocess_data(data_set_name, measure)

                print(f'+++ Algorithm: {mode} --- Tuned: {hpo} --- Seed: {seed} +++')


                num_feat = 0
                if "no_flags" in mode or "flags_model" in mode:
                    if measure == "AT":
                        # max. number of upper-bound flags = max. number of atoms
                        num_atoms = int(data_set_name[data_set_name.index("-atoms") - 1])
                        # max. number of features - consistency flag - upper-bound flags
                        num_feat = X.shape[1] - 1 - num_atoms
                    else:
                        # max. number of features - consistency flag
                        num_feat = X.shape[1] - 1

                if "no_flags" in mode:
                    X = X[:, 0:num_feat]

                evaluate(X, y, seed, hpos, hpo, num_feat, data_set_name, mode, measure)
