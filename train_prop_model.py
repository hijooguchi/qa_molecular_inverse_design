import numpy as np
from os.path import join, isdir
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from get_props import fix_seed, parse_arguments


def yyplot(y_obs: np.ndarray, y_pred: np.ndarray, save_name: str) -> None:
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_obs, y_pred, s=5)
    plt.plot(
        [ymin - yrange * 0.01, ymax + yrange * 0.01],
        [ymin - yrange * 0.01, ymax + yrange * 0.01],
    )
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y observed', fontsize=24)
    plt.ylabel('y predicted', fontsize=24)
    plt.title('Observed-Predicted Plot', fontsize=24)
    plt.tick_params(labelsize=16)

    plt.legend(
        [
            f'RMSE : {mean_squared_error(y_obs, y_pred, squared=False):.3f}\n'
            f'MAE : {mean_absolute_error(y_obs, y_pred):.3f}\n'
            f'R2 : {r2_score(y_obs, y_pred):.3f}'
        ],
        fontsize=14,
        loc='upper left',
    )
    plt.savefig(save_name)
    plt.close()


def run() -> None:
    args = parse_arguments()
    save_dir = args.save_dir
    log_filename = join(save_dir, 'log.txt')

    if not isdir(save_dir):
        raise FileNotFoundError(f'{save_dir} is not available')

    with open(join(save_dir, 'config.json'), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    fix_seed(args.seed)

    x_train = np.load(join(save_dir, f'train_cddd.npy'))
    x_test = np.load(join(save_dir, f'test_cddd.npy'))

    props = [
        'QED',
        'MW',
        'logP',
        'TPSA',
        'nHBD',
        'nHBA',
        'nRB',
        'nAR',
        'SAS',
    ]

    for prop in tqdm(props):
        y_train = np.load(
            join(save_dir, f'train_{prop}.npy')
        ).reshape(-1, 1)
        y_test = np.load(
            join(save_dir, f'test_{prop}.npy')
        ).reshape(-1, 1)

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        log_message = (
            f'Liner regression of {prop}\n'
            f'  RMSE:'
            f' {mean_squared_error(y_test, y_pred, squared=False):.8f}\n'
            f'  MAE: {mean_absolute_error(y_test, y_pred):.8f}\n'
            f'  R2: {r2_score(y_test, y_pred):.8f}\n'
        )
        print(log_message)

        with open(log_filename, 'a') as f:
            f.write(log_message)

        pickle.dump(
            model,
            open(join(save_dir, f'model_{prop}.pkl'), 'wb'),
        )
        yyplot(
            y_obs=y_test,
            y_pred=y_pred,
            save_name=join(save_dir, f'yyplot_{prop}.png'),
        )

        ss = StandardScaler()
        y_train_ss = ss.fit_transform(y_train)
        y_test_ss = ss.transform(y_test)

        model_ss = LinearRegression()
        model_ss.fit(x_train, y_train_ss)
        y_pred_ss = model_ss.predict(x_test)

        log_message = (
            f'Liner regression of {prop} (standard scaler)\n'
            f'  RMSE:'
            f' {mean_squared_error(y_test_ss, y_pred_ss, squared=False):.8f}\n'
            f'  MAE: {mean_absolute_error(y_test_ss, y_pred_ss):.8f}\n'
            f'  R2: {r2_score(y_test_ss, y_pred_ss):.8f}\n'
        )
        print(log_message)

        with open(log_filename, 'a') as f:
            f.write(log_message)

        pickle.dump(
            model_ss,
            open(join(save_dir, f'model_{prop}_ss.pkl'), 'wb'),
        )
        pickle.dump(
            ss,
            open(join(save_dir, f'standard_scaler_{prop}.pkl'), 'wb'),
        )
        yyplot(
            y_obs=y_test_ss,
            y_pred=y_pred_ss,
            save_name=join(save_dir, f'yyplot_{prop}_ss.png'),
        )


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt as e:
        print('[STOP]', e)
