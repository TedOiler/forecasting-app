import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------
def group_data(data, grouping, log_bool=True):
    data['document_date'] = pd.to_datetime(data['document_date'], format="%Y-%m-%d")
    group = data.groupby(pd.Grouper(key='document_date', freq=grouping))
    group_df = pd.DataFrame(group['total_amount'].sum())  # convert to dataframe for ease of filtering
    filt = group_df.index >= '2020-01-01'
    group_df = group_df[filt]

    if log_bool:
        # convert to log-total amount for always positive results.
        group_df['log_total_amount'] = np.log(group_df['total_amount'])
        group_df = group_df['log_total_amount']
    else:
        group_df = group_df['total_amount']

    return group_df


def train_val_test_split(data, f_period):
    train_df = data.iloc[:-f_period]
    test_df = data.iloc[-f_period:]
    return train_df, test_df


'''
Seems to not be needed for the time being

def write_results_st(prediction, confidence, path='./output/', log_bool=True):
    if log_bool:
        data = pd.concat([round(np.exp(prediction), 2), round(np.exp(confidence), 2)], axis=1)
    else:
        data = pd.concat([round(prediction, 2), round(confidence, 2)], axis=1)

    return data
'''


# -------------------------------------------


def fit_arima_model(train_df, val_df, max_search, model_metric, grouping=None):
    def manual_arima(df, p, d, q):
        model = ARIMA(df, order=(p, d, q))
        return model.fit(method_kwargs={"warn_convergence": False})  # supress warnings

    def set_hparams_arima(train, val, max_search=max_search, metric=model_metric):
        cross_val = list()
        for p in tqdm(range(max_search)):
            for d in range(2):
                for q in range(max_search):
                    try:
                        model = manual_arima(df=train, p=p, d=d, q=q)
                        predictions = model.get_forecast(val.shape[0]).predicted_mean
                        aic = model.aic
                        bic = model.bic
                        rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                        mae = sum(np.abs((val - predictions))) / val.shape[0]
                        mape = sum(np.abs((val - predictions))/np.abs(val)) * 100 / val.shape[0]
                        logcosh = sum(np.log(np.cosh(predictions - val))) / val.shape[0]
                    except:
                        aic = np.infty
                        bic = np.infty
                        rmse = np.infty
                        mae = np.infty
                        mape = np.infty
                        logcosh = np.infty
                    cross_val.append([p, d, q, rmse, mae, mape, logcosh, aic, bic])
        results = pd.DataFrame(cross_val)
        results.columns = ['p', 'd', 'q', 'rmse', 'mae', 'mape', 'logcosh', 'aic', 'bic']
        arg_min = results[metric].idxmin()
        minimum_results = results.iloc[arg_min]
        order = (minimum_results['p'], minimum_results['d'], minimum_results['q'])
        return order, results

    def last_fit_arima(train_data, val_data, best_p, best_d, best_q):
        refit_data = pd.concat([train_data, val_data], axis=0)
        last_fit = ARIMA(refit_data, order=(best_p, best_d, best_q))
        return last_fit.fit()

    hparams = set_hparams_arima(train=train_df, val=val_df, max_search=max_search)
    refit_model = last_fit_arima(train_data=train_df, val_data=val_df,
                                 best_p=hparams[0][0], best_d=hparams[0][1], best_q=hparams[0][2])
    return refit_model, (hparams[0][0], hparams[0][1], hparams[0][2]), hparams[1]


def fit_ari_model(train_df, val_df, max_search, model_metric, grouping=None):
    def manual_arima(df, p, d, q):
        model = ARIMA(df, order=(p, d, q))
        return model.fit(method_kwargs={"warn_convergence": False})  # supress warnings

    def set_hparams_arima(train, val, max_search=max_search, metric=model_metric):
        cross_val = list()
        for p in tqdm(range(max_search)):
            for d in range(2):
                model = manual_arima(df=train, p=p, d=d, q=0)
                predictions = model.get_forecast(val.shape[0]).predicted_mean
                aic = model.aic
                bic = model.bic
                rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                mae = sum(np.abs((val - predictions))) / val.shape[0]
                mape = sum(np.abs((val - predictions))/np.abs(val)) * 100 / val.shape[0]
                logcosh = sum(np.log(np.cosh(predictions - val))) / val.shape[0]
                cross_val.append([p, d, rmse, mae, mape, logcosh, aic, bic])
        results = pd.DataFrame(cross_val)
        results.columns = ['p', 'd', 'rmse', 'mae', 'mape', 'logcosh', 'aic', 'bic']
        arg_min = results[metric].idxmin()
        minimum_results = results.iloc[arg_min]
        order = (minimum_results['p'], minimum_results['d'])
        return order, results

    def last_fit_arima(train_data, val_data, best_p, best_d, best_q=0):
        refit_data = pd.concat([train_data, val_data], axis=0)
        last_fit = ARIMA(refit_data, order=(best_p, best_d, best_q))
        return last_fit.fit()

    hparams = set_hparams_arima(train=train_df, val=val_df, max_search=max_search)
    refit_model = last_fit_arima(train_data=train_df, val_data=val_df,
                                 best_p=hparams[0][0], best_d=hparams[0][1], best_q=0)
    return refit_model, (hparams[0][0], hparams[0][1]), hparams[1]


def fit_ima_model(train_df, val_df, max_search, model_metric, grouping=None):
    def manual_arima(df, p, d, q):
        model = ARIMA(df, order=(p, d, q))
        return model.fit(method_kwargs={"warn_convergence": False})  # supress warnings

    def set_hparams_arima(train, val, max_search=max_search, metric=model_metric):
        cross_val = list()
        for q in tqdm(range(max_search)):
            for d in range(2):
                model = manual_arima(df=train, p=0, d=d, q=q)
                predictions = model.get_forecast(val.shape[0]).predicted_mean
                aic = model.aic
                bic = model.bic
                rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                mae = sum(np.abs((val - predictions))) / val.shape[0]
                mape = sum(np.abs((val - predictions))/np.abs(val)) * 100 / val.shape[0]
                logcosh = sum(np.log(np.cosh(predictions - val))) / val.shape[0]
                cross_val.append([d, q, rmse, mae, mape, logcosh, aic, bic])
        results = pd.DataFrame(cross_val)
        results.columns = ['d', 'q', 'rmse', 'mae', 'mape', 'logcosh', 'aic', 'bic']
        arg_min = results[metric].idxmin()
        minimum_results = results.iloc[arg_min]
        order = (minimum_results['d'], minimum_results['q'])
        return order, results

    def last_fit_arima(train_data, val_data, best_d, best_q, best_p=0):
        refit_data = pd.concat([train_data, val_data], axis=0)
        last_fit = ARIMA(refit_data, order=(best_p, best_d, best_q))
        return last_fit.fit()

    hparams = set_hparams_arima(train=train_df, val=val_df, max_search=max_search)
    refit_model = last_fit_arima(train_data=train_df, val_data=val_df,
                                 best_p=0, best_d=hparams[0][0], best_q=hparams[0][1])
    return refit_model, (hparams[0][0], hparams[0][1]), hparams[1]


def fit_sarima_model(train_df, val_df, max_search, model_metric, grouping):
    max_search_ = max_search/5  # To prevent extremely long training times

    def manual_sarima(df, order, s_order):
        model = SARIMAX(df, order=order, seasonal_order=s_order)
        return model.fit(disp=-1)  # supress warnings

    def set_hparams_sarima(train, val, max_search, grouping=grouping, metric=model_metric):
        # Seasonality every 3 months or every month in various granularities.
        seasonality = {'D': [90, 30],
                       'W-SUN': [12, 4],
                       'W-MON': [12, 4],
                       'M': [3, 1]}

        cross_val = list()
        for p in tqdm(range(max_search)):
            for d in range(2):  # Prevent overfitting
                for q in range(2):  # Not many MA terms are needed. Prevent overfitting
                    for P in range(max_search):
                        for D in range(2):  # Prevent overfitting
                            for Q in range(2):  # Not many MA terms are needed. Prevent overfitting
                                for S in seasonality[grouping]:
                                    try:
                                        model = manual_sarima(df=train, order=(p, d, q), s_order=(P, D, Q, S))
                                        predictions = model.get_forecast(val.shape[0]).predicted_mean
                                        rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                                        mae = sum(np.abs((val - predictions))) / val.shape[0]
                                        mape = sum(np.abs((val - predictions))/np.abs(val)) * 100 / val.shape[0]
                                        logcosh = sum(np.log(np.cosh(predictions - val))) / val.shape[0]
                                        aic = model.aic
                                        bic = model.bic
                                    except:
                                        rmse = np.infty
                                        mae = np.infty
                                        mape = np.infty
                                        logcosh = np.infty
                                        aic = np.infty
                                        bic = np.infty
                                        continue
                                    cross_val.append([p, d, q, P, D, Q, S, rmse, mae, mape, logcosh, aic, bic])
        results = pd.DataFrame(cross_val)
        results.columns = ['p', 'd', 'q', 'P', 'D', 'Q', 'S', 'rmse', 'mae', 'mape', 'logcosh', 'aic', 'bic']
        arg_min = results[metric].idxmin()
        minimum_results = results.iloc[arg_min]
        order = (minimum_results['p'], minimum_results['d'], minimum_results['q'])
        seasonal_order = (minimum_results['P'], minimum_results['D'], minimum_results['Q'], minimum_results['S'])
        return order, seasonal_order, results

    def last_fit_sarima(train_data, val_data, best_order, best_s_order):
        refit_data = pd.concat([train_data, val_data], axis=0)
        last_fit = SARIMAX(refit_data, order=best_order, seasonal_order=best_s_order)
        return last_fit.fit(disp=-1)

    hparams = set_hparams_sarima(train=train_df, val=val_df, max_search=max_search_)
    refit_model = last_fit_sarima(train_data=train_df, val_data=val_df, best_order=hparams[0], best_s_order=hparams[1])
    return refit_model, (hparams[0], hparams[1]), hparams[2]


def fit_ets_model(train_df, val_df, max_search, model_metric, grouping=None):
    def manual_ets(df, t, d, s, p, b, r):
        model = ExponentialSmoothing(df, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
        return model.fit(optimized=True, use_boxcox=b, remove_bias=r)

    def set_hparams_ets(train, val, max_search=max_search, metric=model_metric):
        cross_val = list()
        for t in tqdm(['add', 'mul', None]):
            for d in [True, False]:
                for s in ['add', 'mul', None]:
                    for p in range(max_search):
                        for b in [True, False]:
                            for r in [True, False]:
                                try:
                                    model = manual_ets(df=train, t=t, d=d, s=s, p=p, b=b, r=r)
                                    predictions = model.forecast(val.shape[0])
                                    aic = model.aic
                                    bic = model.bic
                                    rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                                    mae = sum(np.abs((val - predictions))) / val.shape[0]
                                    mape = sum(np.abs((val - predictions))/np.abs(val)) * 100 / val.shape[0]
                                    logcosh = sum(np.log(np.cosh(predictions - val))) / val.shape[0]
                                except:
                                    aic = np.infty
                                    bic = np.infty
                                    rmse = np.infty
                                    mae = np.infty
                                    mape = np.infty
                                    logcosh = np.infty
                                cross_val.append([t, d, s, p, b, r, rmse, mae, mape, logcosh, aic, bic])
        results = pd.DataFrame(cross_val)
        results.columns = ['t', 'd', 's', 'p', 'b', 'r', 'rmse', 'mae', 'mape', 'logcosh', 'aic', 'bic']
        arg_min = results[metric].idxmin()
        minimum_results = results.iloc[arg_min]
        params = (minimum_results['t'], minimum_results['d'], minimum_results['s'],
                  minimum_results['p'], minimum_results['b'], minimum_results['r'])
        return params, results

    def last_fit_ets(train_data, val_data, best_t, best_d, best_s, best_p, best_b, best_r):
        refit_data = pd.concat([train_data, val_data], axis=0)
        last_fit = ExponentialSmoothing(np.asarray(refit_data), trend=best_t, damped_trend=best_d, seasonal=best_s,
                                        seasonal_periods=best_p)
        return last_fit.fit(optimized=True, use_boxcox=best_b, remove_bias=best_r)

    hparams = set_hparams_ets(train=train_df, val=val_df, max_search=max_search)
    refit_model = last_fit_ets(train_data=train_df, val_data=val_df,
                                best_t=hparams[0][0], best_d=hparams[0][1], best_s=hparams[0][2],
                                best_p=hparams[0][3], best_b=hparams[0][4], best_r=hparams[0][5])
    return refit_model, (hparams[0][0], hparams[0][1], hparams[0][2],
                         hparams[0][3], hparams[0][4], hparams[0][5]), hparams[1]


# Discontinued methods  ---------------------------------------------------------------------------

def fit_garch_model(train_df, val_df, max_search, f_period, model_metric, grouping=None):
    # discontinued
    def manual_garch(df, p, o, q):
        model = arch_model(df, vol='GARCH', p=p, o=o, q=q)
        return model.fit(disp='off')

    def set_hparams_garch(train, val, max_search=max_search, f_period=f_period, metric=model_metric):
        cross_val = list()
        for p in tqdm(range(max_search)):
            for o in range(max_search):
                for q in range(max_search):
                    try:
                        model = manual_garch(df=train, p=p, o=o, q=q)
                        predictions = np.sqrt(model.forecast(horizon=f_period).variance.values[-1, :])
                        aic = model.aic
                        bic = model.bic
                        rmse = np.sqrt(sum((val - predictions) ** 2) / f_period)
                    except:
                        aic = np.infty
                        bic = np.infty
                        rmse = np.infty
                    cross_val.append([p, o, q, rmse, aic, bic])
        results = pd.DataFrame(cross_val)
        results.columns = ['p', 'o', 'q', 'rmse', 'aic', 'bic']
        arg_min = results[metric].idxmin()
        minimum_results = results.iloc[arg_min]
        order = (minimum_results['p'], minimum_results['o'], minimum_results['q'])
        return order, results

    def last_fit_garch(train_data, val_data, best_p, best_o, best_q):
        refit_data = pd.concat([train_data, val_data], axis=0)
        last_fit = arch_model(refit_data, vol='GARCH', p=int(best_p), o=int(best_o), q=int(best_q))
        return last_fit.fit(disp='off')

    hparams = set_hparams_garch(train=train_df, val=val_df, max_search=max_search, f_period=f_period)
    refit_model = last_fit_garch(train_data=train_df, val_data=val_df, best_p=hparams[0][0], best_o=hparams[0][1],
                                 best_q=hparams[0][2])
    return refit_model, [hparams[0][0], hparams[0][1], hparams[0][2]], hparams[1]


def fit_arima_garch_model(train_df, val_df, max_search, f_period, model_metric, grouping=None):
    # discontinued
    arima_model, _, _ = fit_arima_model(train_df=train_df, val_df=val_df, max_search=max_search,
                                        f_period=f_period, model_metric=model_metric, grouping=grouping)
    garch_model, _, _ = fit_garch_model(train_df=train_df, val_df=val_df, max_search=max_search,
                                        f_period=f_period, model_metric=model_metric, grouping=grouping)

    predicted_mu = arima_model.predict(n_periods=f_period)
    garch_forecast = garch_model.forecast(horizon=f_period)
    predicted_et = np.sqrt(garch_forecast.mean['h.1'].iloc[-1])
    prediction = predicted_mu + predicted_et
    pass


def fit_sarima_garch_model(train_df, val_df, max_search, f_period, model_metric, grouping=None):
    # https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff
    # https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/
    # https://medium.com/analytics-vidhya/a-step-by-step-implementation-of-a-trading-strategy-in-python-using-arima-garch-models-b622e5b3aa39
    # discontinued
    pass

# ------------------------------------------------------------------------------------------------
