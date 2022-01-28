import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import fun_preprocessing as fp

def manual_arima(train_df, p, d, q):
    model = ARIMA(train_df, order=(p, d, q))
    return model.fit()


def set_hparams(train, val, max_search=5, f_period=4):
    cross_val = list()
    for p in range(max_search):
        for d in range(2):
            for q in range(max_search):
                predictions = manual_arima(train_df=train, p=p, d=d, q=q).get_forecast(f_period).predicted_mean
                rmse = np.sqrt(sum((val - predictions) ** 2) / f_period)
                cross_val.append([p, d, q, rmse])
    results = pd.DataFrame(cross_val)
    results.columns = ['p', 'd', 'q', 'rmse']
    arg_min = results['rmse'].idxmin()
    minimum_results = results.iloc[arg_min]

    return minimum_results['p'], minimum_results['d'], minimum_results['q']


def last_fit(train, val, p, d, q):
    refit_data = pd.concat([train, val], axis=0)
    last_fit = ARIMA(refit_data, order=(p, d, q))
    return last_fit.fit()


def run_arima_routine(read_path='./input/inflows-20_21.xlsx',
                      grouping='W-SUN',
                      forecast_periods=4,
                      confidence_int=0.5,
                      periods_search=4,
                      log_bool=True,
                      live=False):

    if live:
        # Read, group and split to train test and validation sets.
        w_df_train, w_df_test = fp.train_val_test_split(
            data=fp.group_data(
                data=fp.read_clean_data(path=read_path),
                grouping=grouping,
                log_bool=log_bool),
            live=live)

        # Hyperparameter tuning
        best_p, best_d, best_q = set_hparams(train=w_df_train, val=w_df_test, max_search=periods_search+1, f_period=4)
        print(best_p, best_d, best_q)
        # Last fit and forecast
        refit_model = last_fit(w_df_train, w_df_test, best_p, best_d, best_q)
        fc = refit_model.get_forecast(forecast_periods)

        # Write results
        fp.write_results(prediction=fc.predicted_mean, confidence=fc.conf_int(alpha=confidence_int))
        return w_df_train, w_df_test, refit_model, fc.predicted_mean

    else:
        w_df_train, w_df_val, w_df_test = fp.train_val_test_split(
            data=fp.group_data(
                data=fp.read_clean_data(path=read_path),
                grouping=grouping,
                log_bool=log_bool),
            live=live)

        # Hyperparameter tuning
        best_p, best_d, best_q = set_hparams(train=w_df_train, val=w_df_val, max_search=periods_search+1, f_period=4)
        print(best_p, best_d, best_q)
        # Last fit and forecast
        refit_model = last_fit(w_df_train, w_df_val, best_p, best_d, best_q)
        fc = refit_model.get_forecast(forecast_periods)

        # Write results
        fp.write_results(prediction=fc.predicted_mean, confidence=fc.conf_int(alpha=confidence_int), log_bool=log_bool)
        return w_df_train, w_df_val, refit_model, fc.predicted_mean


def run_arima_routine_st(data,
                         grouping='W-SUN',
                         forecast_periods=4,
                         confidence_int=0.5,
                         periods_search=4,
                         log_bool=True,
                         live=False):

    if live:
        # Read, group and split to train test and validation sets.
        w_df_train, w_df_test = fp.train_val_test_split(data=data, live=live)

        # Hyperparameter tuning
        best_p, best_d, best_q = set_hparams(train=w_df_train, val=w_df_test, max_search=periods_search+1, f_period=4)
        print(best_p, best_d, best_q)
        # Last fit and forecast
        refit_model = last_fit(w_df_train, w_df_test, best_p, best_d, best_q)
        fc = refit_model.get_forecast(forecast_periods)

        # Write results
        results = fp.write_results_st(prediction=fc.predicted_mean, confidence=fc.conf_int(alpha=confidence_int))
        return w_df_train, w_df_test, refit_model, fc.predicted_mean, results

    else:
        w_df_train, w_df_val, w_df_test = fp.train_val_test_split(data=data, live=live)

        # Hyperparameter tuning
        best_p, best_d, best_q = set_hparams(train=w_df_train, val=w_df_val, max_search=periods_search+1, f_period=4)
        print(best_p, best_d, best_q)
        # Last fit and forecast
        refit_model = last_fit(w_df_train, w_df_val, best_p, best_d, best_q)
        fc = refit_model.get_forecast(forecast_periods)

        # Write results
        results = fp.write_results_st(prediction=fc.predicted_mean, confidence=fc.conf_int(alpha=confidence_int), log_bool=log_bool)
        return w_df_train, w_df_val, refit_model, fc.predicted_mean, results