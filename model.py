import scipy.stats as stats
import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from stqdm import stqdm
import warnings

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")


class Model:
    def __init__(self, algorithm, train_set, test_set, grouping='W-SUN', depth=3, forecast_period=4,
                 model_metric='rmse',
                 log_bool=True, conf=0.8):
        self.algorithm = algorithm
        self.train_set = train_set
        self.test_set = test_set
        self.grouping = grouping
        self.grouped_data = None
        self.forecast_period = forecast_period
        self.depth = depth
        self.model_metric = model_metric
        self.log_bool = log_bool
        self.conf = conf

        self.scores = None
        self.prediction = None
        self.prediction_past = None
        self.conf_int = None
        self.metric_history = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.model = None
        self.model_summary = None
        self.best_hparams = None
        self.diagnostics = None
        self.pred_scaler = None
        self.models_dict = {'AR': fit_ari_model,
                            'MA': fit_ima_model,
                            'ARIMA': fit_arima_model,
                            'SARIMA': fit_sarima_model,
                            'ETS': fit_ets_model}

    def fit(self):
        refit_model, best_hparams, metric_history = self.models_dict[self.algorithm](train_df=self.train_set,
                                                                                     val_df=self.test_set,
                                                                                     max_search=self.depth,
                                                                                     model_metric=self.model_metric,
                                                                                     grouping=self.grouping)

        # Update the object values
        if self.algorithm in ['AR', 'MA', 'ARIMA', 'SARIMA']:
            fc = refit_model.get_forecast(self.forecast_period)
            self.prediction = fc.predicted_mean
            self.conf_int = fc.conf_int(alpha=self.conf)
            self.diagnostics = refit_model.plot_diagnostics(figsize=(15, 15))
            self.prediction_past = refit_model.predict()
        elif self.algorithm == 'ETS':
            fc = refit_model.forecast(self.forecast_period)
            try:
                self.prediction = pd.DataFrame(fc).set_index(
                    self.test_set.index + datetime.timedelta(weeks=self.forecast_period)).squeeze()
            except:
                self.prediction = pd.DataFrame(fc)
            self.conf_int = None  # need to fix this
            self.diagnostics = None  # need to fix this
            self.prediction_past = pd.DataFrame(refit_model.fittedvalues)[:-self.forecast_period].set_index(
                self.test_set.index)
            self.prediction_past.columns = ['predicted_mean']
        elif self.algorithm == 'GARCH':
            fc = refit_model.forecast(horizon=self.forecast_period).variance.values[-1, :]
            self.prediction = fc
            self.conf_int = None  # need to fix this
            self.diagnostics = None  # need to fix this

        self.metric_history = metric_history
        self.scores = {'aic': refit_model.aic, 'bic': refit_model.bic,
                       'rmse': self.metric_history['rmse'].iloc[self.metric_history['rmse'].idxmin()],
                       'mae': self.metric_history['mae'].iloc[self.metric_history['rmse'].idxmin()],
                       'mape': self.metric_history['mape'].iloc[self.metric_history['rmse'].idxmin()]}
        self.df_train = self.test_set
        self.df_test = self.test_set
        self.model = refit_model
        self.model_summary = refit_model.summary()
        self.best_hparams = best_hparams
        self.pred_scaler = stats.norm.ppf(1 - (1 - self.conf) / 2) * np.sqrt(self.model.sse / self.test_set.shape[0])
        # self.diagnostics = (plt.plot(self.model.resid), ts.plot_acf(self.model.resid))
        # results = write_results_st(prediction=self.pred, confidence=self.conf_int)


def evaluate(input_model, conf=0.8):
    history = [x for x in input_model.df_train]
    predictions = list()
    lower_int = list()
    upper_int = list()
    # walk-forward validation

    if input_model.algorithm == 'AR':
        for t in range(len(input_model.df_test)):
            model = ARIMA(history, order=(input_model.best_hparams[0], input_model.best_hparams[1], 0)).fit()
            output = model.get_forecast()
            lower = [output.conf_int(1 - conf)[0][0]]
            yhat = output.predicted_mean
            upper = [output.conf_int(1 - conf)[0][1]]
            predictions.append(yhat)
            lower_int.append(lower)
            upper_int.append(upper)
            obs = input_model.df_test[t]
            history.append(obs)
        # plot forecasts against actual outcomes
        plt.title("1-step Rolling Forecast\nAR Model")
        plt.plot(input_model.df_test.index, input_model.df_test)
        plt.plot(input_model.df_test.index, predictions, color='red')
        plt.plot(input_model.df_test.index, lower_int, color='grey')
        plt.plot(input_model.df_test.index, upper_int, color='grey')
        plt.xticks(rotation=45)
        plt.show()

    elif input_model.algorithm == 'MA':
        for t in range(len(input_model.df_test)):
            model = ARIMA(history, order=(0, input_model.best_hparams[0], input_model.best_hparams[1])).fit()
            output = model.get_forecast()
            lower = [output.conf_int(1 - conf)[0][0]]
            yhat = output.predicted_mean
            upper = [output.conf_int(1 - conf)[0][1]]
            predictions.append(yhat)
            lower_int.append(lower)
            upper_int.append(upper)
            obs = input_model.df_test[t]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))
        # plot forecasts against actual outcomes
        plt.title("1-step Rolling Forecast\nMA Model")
        plt.plot(input_model.df_test.index, input_model.df_test)
        plt.plot(input_model.df_test.index, predictions, color='red')
        plt.plot(input_model.df_test.index, lower_int, color='grey')
        plt.plot(input_model.df_test.index, upper_int, color='grey')
        plt.xticks(rotation=45)
        plt.show()

    elif input_model.algorithm == 'ARIMA':
        for t in range(len(input_model.df_test)):
            model = ARIMA(history,
                          order=(input_model.best_hparams[0], input_model.best_hparams[1], input_model.best_hparams[2]))
            model_fit = model.fit()
            output = model_fit.get_forecast()
            lower = [output.conf_int(1 - conf)[0][0]]
            yhat = output.predicted_mean
            upper = [output.conf_int(1 - conf)[0][1]]
            predictions.append(yhat)
            lower_int.append(lower)
            upper_int.append(upper)
            obs = input_model.df_test[t]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))
        # plot forecasts against actual outcomes
        plt.title("1-step Rolling Forecast\nARIMA Model")
        plt.plot(input_model.df_test.index, input_model.df_test)
        plt.plot(input_model.df_test.index, predictions, color='red')
        plt.plot(input_model.df_test.index, lower_int, color='grey')
        plt.plot(input_model.df_test.index, upper_int, color='grey')
        plt.xticks(rotation=45)
        plt.show()

    elif input_model.algorithm == 'SARIMA':
        for t in range(len(input_model.df_test)):
            model = SARIMAX(history, order=input_model.best_hparams[0], seasonal_order=input_model.best_hparams[1]).fit(
                disp=-1)
            output = model.get_forecast()
            lower = [output.conf_int(1 - conf)[0][0]]
            yhat = output.predicted_mean
            upper = [output.conf_int(1 - conf)[0][1]]
            predictions.append(yhat)
            lower_int.append(lower)
            upper_int.append(upper)
            obs = input_model.df_test[t]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))
        # plot forecasts against actual outcomes
        plt.title("1-step Rolling Forecast\nSARIMA Model")
        plt.plot(input_model.df_test.index, input_model.df_test)
        plt.plot(input_model.df_test.index, predictions, color='red')
        plt.plot(input_model.df_test.index, lower_int, color='grey')
        plt.plot(input_model.df_test.index, upper_int, color='grey')
        plt.xticks(rotation=45)
        plt.show()

    elif input_model.algorithm == 'ETS':
        for t in range(len(input_model.df_test)):
            model = ExponentialSmoothing(np.array(history), trend=input_model.best_hparams[0],
                                         damped_trend=input_model.best_hparams[1],
                                         seasonal=input_model.best_hparams[2],
                                         seasonal_periods=input_model.best_hparams[3]).fit(optimized=True,
                                                                                           use_boxcox=
                                                                                           input_model.best_hparams[4],
                                                                                           remove_bias=
                                                                                           input_model.best_hparams[5])
            yhat = model.forecast()
            predictions.append(yhat)
            obs = input_model.df_test[t]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))
        # plot forecasts against actual outcomes
        plt.title("1-step Rolling Forecast\nETS Model")
        plt.plot(input_model.df_test.index, input_model.df_test)
        plt.plot(input_model.df_test.index, predictions, color='red')
        plt.xticks(rotation=45)
        plt.show()


def fitted_values_forecast(input_model, conf=0.8):
    # https://mins.space/blog/2020-09-14-tuning-forecasting-sarima-models/
    pass


def pool(*args, metric='rmse'):
    models = [args[_] for _ in range(len(args))]
    pred_mean = [models[_].prediction for _ in range(len(args))]
    pred_lower = [models[_].conf_int.iloc[:, 0] for _ in range(len(args))]
    pred_upper = [models[_].conf_int.iloc[:, 1] for _ in range(len(args))]

    scores = [models[_].scores[metric] for _ in range(len(args))]
    if metric == 'rmse':
        deltas = [1 / scores[_] for _ in range(len(args))]
    else:
        scores -= min(scores)
        deltas = [np.exp(-1 / 2 * scores[_]) for _ in range(len(args))]
    total = sum(deltas)
    weights = [deltas[_] / total for _ in range(len(args))]

    pred_mean = np.array(pred_mean).reshape((-1, len(args)))
    pred_lower = np.array(pred_lower).reshape((-1, len(args)))
    pred_upper = np.array(pred_upper).reshape((-1, len(args)))
    weights = np.array(weights).reshape((-1, 1))
    model_w = sorted(list(zip([models[_].algorithm for _ in range(len(args))],
                              [round(weights[_][0], 4) for _ in range(len(args))])),
                     key=lambda x: x[1], reverse=True)

    w_pred_lower = np.sum((pred_lower.T * weights).T, axis=1)
    w_pred_mean = np.sum((pred_mean.T * weights).T, axis=1)
    w_pred_upper = np.sum((pred_upper.T * weights).T, axis=1)
    prediction_df = pd.DataFrame({'predicted_lower': w_pred_lower,
                                  'predicted_mean': w_pred_mean,
                                  'predicted_upper': w_pred_upper},
                                 index=models[0].prediction.index)
    # prediction_df = pd.DataFrame(np.sum((predictions.T * weights).T, axis=1))

    return round(prediction_df, 4), model_w


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


def clean_data(data):
    data_frame = data[['Document Date', 'Total amount']]
    # fix column naming
    data_frame.columns = data_frame.columns.str.replace(' ', '_')
    data_frame.columns = [x.lower() for x in data_frame.columns]
    data_frame['document_date'] = pd.to_datetime(data_frame['document_date'], format="%Y-%m-%d")

    return data_frame


def train_val_test_split(data, f_period):
    train_df = data.iloc[:-f_period]
    test_df = data.iloc[-f_period:]
    return train_df, test_df

# -------------------------------------------


def fit_arima_model(train_df, val_df, max_search, model_metric, grouping=None):
    def manual_arima(df, p, d, q):
        model = ARIMA(df, order=(p, d, q))
        return model.fit(method_kwargs={"warn_convergence": False})  # supress warnings

    def set_hparams_arima(train, val, max_search=max_search, metric=model_metric):
        cross_val = list()
        for p in stqdm(range(max_search), desc="ARIMA"):
            for d in range(2):
                for q in range(max_search):
                    try:
                        model = manual_arima(df=train, p=p, d=d, q=q)
                        predictions = model.get_forecast(val.shape[0]).predicted_mean
                        aic = model.aic
                        bic = model.bic
                        rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                        mae = sum(np.abs((val - predictions))) / val.shape[0]
                        mape = sum(np.abs((val - predictions)) / np.abs(val)) * 100 / val.shape[0]
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
        for p in stqdm(range(max_search), desc="AR"):
            for d in range(2):
                model = manual_arima(df=train, p=p, d=d, q=0)
                predictions = model.get_forecast(val.shape[0]).predicted_mean
                aic = model.aic
                bic = model.bic
                rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                mae = sum(np.abs((val - predictions))) / val.shape[0]
                mape = sum(np.abs((val - predictions)) / np.abs(val)) * 100 / val.shape[0]
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
    def manual_ima(df, p, d, q):
        model = ARIMA(df, order=(p, d, q))
        return model.fit(method_kwargs={"warn_convergence": False})  # supress warnings

    def set_hparams_arima(train, val, max_search=max_search, metric=model_metric):
        cross_val = list()
        for q in stqdm(range(max_search), desc="MA"):
            for d in range(2):
                model = manual_ima(df=train, p=0, d=d, q=q)
                predictions = model.get_forecast(val.shape[0]).predicted_mean
                aic = model.aic
                bic = model.bic
                rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                mae = sum(np.abs((val - predictions))) / val.shape[0]
                mape = sum(np.abs((val - predictions)) / np.abs(val)) * 100 / val.shape[0]
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
        for p in stqdm(range(max_search), desc="SARIMA"):
            for d in stqdm(range(2)):  # Prevent overfitting
                for q in stqdm(range(2)):  # Not many MA terms are needed. Prevent overfitting
                    for P in stqdm(range(max_search)):
                        for D in stqdm(range(2)):  # Prevent overfitting
                            for Q in stqdm(range(2)):  # Not many MA terms are needed. Prevent overfitting
                                for S in stqdm(seasonality[grouping]):
                                    try:
                                        model = manual_sarima(df=train, order=(p, d, q), s_order=(P, D, Q, S))
                                        predictions = model.get_forecast(val.shape[0]).predicted_mean
                                        rmse = np.sqrt(sum((val - predictions) ** 2) / val.shape[0])
                                        mae = sum(np.abs((val - predictions))) / val.shape[0]
                                        mape = sum(np.abs((val - predictions)) / np.abs(val)) * 100 / val.shape[0]
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

    hparams = set_hparams_sarima(train=train_df, val=val_df, max_search=max_search)
    refit_model = last_fit_sarima(train_data=train_df, val_data=val_df, best_order=hparams[0], best_s_order=hparams[1])
    return refit_model, (hparams[0], hparams[1]), hparams[2]


def fit_ets_model(train_df, val_df, max_search, model_metric, grouping=None):
    def manual_ets(df, t, d, s, p, b, r):
        model = ExponentialSmoothing(df, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
        return model.fit(optimized=True, use_boxcox=b, remove_bias=r)

    def set_hparams_ets(train, val, max_search=max_search, metric=model_metric):
        cross_val = list()
        for t in stqdm(['add', 'mul', None], desc="ETS"):
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
                                    mape = sum(np.abs((val - predictions)) / np.abs(val)) * 100 / val.shape[0]
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
