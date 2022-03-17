import streamlit as st
import pandas as pd
import model as m
import plotly.express as px


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title='Forecaster', page_icon="📈")
st.sidebar.title("Sidebar")

# Input Form --------------------------------------------
st.sidebar.header("1. Input")
input_form = st.sidebar.form(key="input")
input_expander = input_form.expander("File Upload")
# Input Data Upload Form
with input_expander:
    uploaded_file = input_expander.file_uploader("Choose a file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, header=[0])
        except:
            df = pd.read_csv(uploaded_file, header=[0])
        try:
            data = m.clean_data(data=df)
            data['document_date'].dt.strftime("%Y-%m-%d")

        except:
            data = None
            input_expander.error(
                "File Upload Unsuccessful")
    else:
        data = None

params_expander = input_form.expander("Model Parameters")
with params_expander:
    col1, col2 = params_expander.columns(2)
    grouping = col1.selectbox("Cohort", ["W-SUN", "W-MON", "D", "M"])
    model_metric = col2.selectbox("Loss", ["rmse", "mae", "aic", "bic"])
    forecast_period = params_expander.number_input('Forecast Periods', min_value=1, max_value=90, value=4, step=1)
    test_period = params_expander.number_input("Test Periods", 1, 200, value=15, step=1)
    conf = 1 - params_expander.slider("Confidence Int", 0., 1., value=0.2, step=0.05)
    # log_bool = params_expander.checkbox("Log")
    log_bool = False

col1, col2 = input_form.columns(2)
submit_data_btn = col1.form_submit_button("Submit")
# Visualize Input Data
if submit_data_btn:
    col2.success("Success")

# Documentation --------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Documentation")
document_expander = st.sidebar.expander("Search Documentation")
with document_expander:
    code_link = '[Source code](https://github.com/TedOiler/forecasting-app)'
    document_expander.markdown(code_link, unsafe_allow_html=True)
    documentation_link = '[Documentation](https://github.com/TedOiler/forecasting-app)'
    document_expander.markdown(documentation_link, unsafe_allow_html=True)

#  Main-Page --------------------------------------------
st.title("Cashflow Forecasting")
if data is not None:
    st.session_state.grouped_data = m.group_data(data=data, grouping=grouping, log_bool=log_bool).iloc[:-1]
    df_train, df_test = m.train_val_test_split(data=st.session_state.grouped_data, f_period=test_period)
else:
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    st.session_state.grouped_data = pd.DataFrame([1])
train_prc_length = df_train.shape[0] / st.session_state.grouped_data.shape[0]
data_expander = st.expander(f"Train-Test: {round(train_prc_length * 100)}% - {round((1 - train_prc_length) * 100)}%")
with data_expander:
    col1, col2, col3, col4 = data_expander.columns(4)
    col1.write(f"Train length: {df_train.shape[0]}")
    col1.dataframe(df_train)
    col1.download_button(label='Export Train as CSV',
                         data=convert_df(df_train),
                         file_name='train_data.csv',
                         mime='text/csv')
    col2.write(f"Test length: {df_test.shape[0]}")
    col2.dataframe(df_test)
    col2.download_button(label='Export Test as CSV',
                         data=convert_df(df_test),
                         file_name='test_data.csv',
                         mime='text/csv')
    col3.empty()
    col4.empty()

models_form = st.form(key="models")
model_expander = models_form.expander("Models")
with model_expander:
    col1, col2 = model_expander.columns(2)
    algo_1 = col1.selectbox("Model #1", ['MA', None], 0)
    strength_1 = col1.slider("Model #1 Depth", 1, test_period, value=int(test_period / 1.2), step=1)
    algo_2 = col2.selectbox("Model #2", ['AR', None], 0)
    strength_2 = col2.slider("Model #2 Depth", 1, test_period, value=int(test_period / 1.2), step=1)
    col1.markdown('---')
    col2.markdown('---')
    algo_3 = col1.selectbox("Model #3", ['ARIMA', None], 0)
    strength_3 = col1.slider("Model #3 Depth", 1, test_period, value=int(test_period / 2), step=1)
    algo_4 = col2.selectbox("Model #4", ['SARIMA', None], 0)
    strength_4 = col2.slider("Model #4 Depth", 1, 8, value=4, step=1)
    col1, col2, col3, col4 = models_form.columns(4)
    train_models_btn = col1.form_submit_button("Train")

    # State initialization to avoid errors down the stream
    if 'model_1' not in st.session_state:
        st.session_state.model_1 = m.Model(algorithm=algo_1, train_set=df_train, test_set=df_test)
    if 'model_2' not in st.session_state:
        st.session_state.model_2 = m.Model(algorithm=algo_2, train_set=df_train, test_set=df_test)
    if 'model_3' not in st.session_state:
        st.session_state.model_3 = m.Model(algorithm=algo_3, train_set=df_train, test_set=df_test)
    if 'model_4' not in st.session_state:
        st.session_state.model_4 = m.Model(algorithm=algo_4, train_set=df_train, test_set=df_test)
    if 'all_data' not in st.session_state:
        st.session_state.all_data = pd.DataFrame()

    if train_models_btn:
        st.session_state.model_1 = m.Model(algorithm=algo_1, train_set=df_train, test_set=df_test,
                                           grouping=grouping, depth=strength_1,
                                           model_metric=model_metric, forecast_period=forecast_period,
                                           log_bool=log_bool, conf=conf)
        st.session_state.model_1.fit()

        st.session_state.model_2 = m.Model(algorithm=algo_2, train_set=df_train, test_set=df_test,
                                           grouping=grouping, depth=strength_2,
                                           model_metric=model_metric, forecast_period=forecast_period,
                                           log_bool=log_bool, conf=conf)
        st.session_state.model_2.fit()

        st.session_state.model_3 = m.Model(algorithm=algo_3, train_set=df_train, test_set=df_test,
                                           grouping=grouping, depth=strength_3,
                                           model_metric=model_metric, forecast_period=forecast_period,
                                           log_bool=log_bool, conf=conf)
        st.session_state.model_3.fit()

        st.session_state.model_4 = m.Model(algorithm=algo_4, train_set=df_train, test_set=df_test,
                                           grouping=grouping, depth=strength_4,
                                           model_metric=model_metric, forecast_period=forecast_period,
                                           log_bool=log_bool, conf=conf)
        st.session_state.model_4.fit()

        col4.success("Success")

residuals_expander = st.expander("Model Evaluation")
if st.session_state.model_1 is not None:
    col1, col2 = residuals_expander.columns(2)
    col1.write(f"{algo_1} - Residuals")
    col1.pyplot(st.session_state.model_1.diagnostics)
    col2.write(f"{algo_2} - Residuals")
    col2.pyplot(st.session_state.model_2.diagnostics)
    col1.write(f"{algo_3} - Residuals")
    col1.pyplot(st.session_state.model_3.diagnostics)
    col2.write(f"{algo_4} - Residuals")
    col2.pyplot(st.session_state.model_4.diagnostics)

# -----------------------------------------------------------

fitted_values_expander = st.expander("Visual")
if st.session_state.model_1 is not None:
    col1, col2 = fitted_values_expander.columns(2)
    true = pd.concat([df_train, df_test], axis=0)

    st.session_state.all_data_m1 = pd.concat([true, st.session_state.model_1.prediction_past], axis=1)
    st.session_state.all_data_m2 = pd.concat([true, st.session_state.model_2.prediction_past], axis=1)
    st.session_state.all_data_m3 = pd.concat([true, st.session_state.model_3.prediction_past], axis=1)
    st.session_state.all_data_m4 = pd.concat([true, st.session_state.model_4.prediction_past], axis=1)
    fig1 = px.line(st.session_state.all_data_m1, template="seaborn")
    fig1.update_layout(
        title=f"MA Prediction: {st.session_state.model_1.best_hparams}",
        xaxis_title="Date",
        yaxis_title="True Value",
        legend_title="Legend",
        showlegend=True)
    fig2 = px.line(st.session_state.all_data_m2, template="seaborn")
    fig2.update_layout(
        title=f"AR Prediction: {st.session_state.model_2.best_hparams}",
        xaxis_title="Date",
        yaxis_title="True Value",
        legend_title="Legend",
        showlegend=True)
    fig3 = px.line(st.session_state.all_data_m3, template="seaborn")
    fig3.update_layout(
        title=f"ARIMA Prediction: {st.session_state.model_3.best_hparams}",
        xaxis_title="Date",
        yaxis_title="True Value",
        legend_title="Legend",
        showlegend=True)
    fig4 = px.line(st.session_state.all_data_m4, template="seaborn")
    fig4.update_layout(
        title=f"SARIMA Prediction: {st.session_state.model_4.best_hparams}",
        xaxis_title="Date",
        yaxis_title="True Value",
        legend_title="Legend",
        showlegend=True)

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    col1.plotly_chart(fig3, use_container_width=True)
    col2.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------

pool_form = st.form(key="pool")
pool_expander = pool_form.expander("Pool Predictions")
col1, col2 = pool_expander.columns(2)
selected_models = col1.multiselect("Select Models", ['MA', 'AR', 'ARIMA', 'SARIMA'])
weight = col2.selectbox("Weight Metric", ['rmse', 'mae', 'aic', 'bic'])
col1, col2 = pool_form.columns(2)
pool_btn = col1.form_submit_button("Pool")
model_dict = {'MA': st.session_state.model_1,
              'AR': st.session_state.model_2,
              'ARIMA': st.session_state.model_3,
              'SARIMA': st.session_state.model_4}
if pool_btn:
    models = [model_dict[selected_models[_]] for _ in range(len(selected_models))]
    pool_forecast, model_list, pool_predicted = m.pool(*models, metric=weight)
    true = pd.concat([df_train, df_test], axis=0)
    st.session_state.all_data = pd.concat([true, pool_predicted, pool_forecast], axis=1)
    # st.write(st.session_state.all_data)
    st.session_state.all_data.columns = ['true', 'prediction', 'forecast_lower', 'forecast_mean', 'forecast_upper']
    fig = px.line(st.session_state.all_data, template="seaborn")
    fig.update_layout(
        title=f"Forecast \n {model_list}",
        xaxis_title="Date",
        yaxis_title="True Value",
        legend_title="Legend")
    pool_expander.plotly_chart(fig, use_container_width=True)

    col2.success("Success")

else:
    all_data = None

export_expander = st.expander("Export Results")
if st.session_state.all_data is not None:
    export_expander.dataframe(st.session_state.all_data)
    export_expander.download_button(label='Export Results as CSV',
                                    data=convert_df(st.session_state.all_data),
                                    file_name='forecast.csv',
                                    mime='text/csv')
st.markdown("---")
model_dicts = st.expander("Data Dump")
if st.session_state.all_data is not None:
    col1, col2, col3, col4 = model_dicts.columns(4)
    col1.write("MA Model Data")
    col1.write(st.session_state.model_1.__dict__)
    col2.write("AR Model Data")
    col2.write(st.session_state.model_2.__dict__)
    col3.write("ARIMA Model Data")
    col3.write(st.session_state.model_3.__dict__)
    col4.write("SARIMA Model Data")
    col4.write(st.session_state.model_4.__dict__)
