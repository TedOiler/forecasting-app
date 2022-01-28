import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import fun_preprocessing as fp
import fun_models as fm

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.set_option('deprecation.showPyplotGlobalUse', False) #Temporary
st.title("Monitoring")
st.sidebar.title("Cashflow Forecasting")

# Input Form --------------------------------------------
st.sidebar.header("Input File")
input_form = st.sidebar.form(key="input")
input_expander = input_form.expander("File Upload")
with input_expander:
    uploaded_file = input_expander.file_uploader("Choose a file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=[0])
        except Exception:
            df = pd.read_excel(uploaded_file, header=[0])
        try:
            data = fp.clean_data(data=df)
        except Exception:
            data = pd.dataFrame()
            input_expander.error(
                "File was expected to have the following columns: [\'Document Date\', \'Total amount\', \'DEAL ID\', \'Apartment\']")
    else:
        data = pd.DataFrame()

col1, col2 = input_form.columns(2)
submit_data_btn = col1.form_submit_button("Submit")
if submit_data_btn:
    col2.success("Success")
input_data_expander = input_form.expander("Raw data preview")
with input_data_expander:
    input_data_expander.dataframe(data)

# Parameters Form --------------------------------------------
st.sidebar.header("Parameters")
parameter_form = st.sidebar.form(key="parameters")
params_expander = parameter_form.expander("Set Model Parameters")
with params_expander:
    # set Flags
    grouping = params_expander.selectbox("Granularity", ["W-SUN", "D", "W-MON", "M"])
    col1, col2 = params_expander.columns(2)
    log_transform = col1.checkbox("Log")
    live = col2.checkbox("Live")
    f_period = col1.slider("Forecast periods", 1, 12, 1)
    confidence_int = col1.slider("Confidence Int", 0., 1., 0.05)
    search_strength = col2.slider("Search Strength", 1, 10, 1)
    dummy = col2.slider("Dummy", 1, 7, 1)

col1, col2 = parameter_form.columns(2)
submit_params_btn = col1.form_submit_button("Submit")
try:
    group = fp.group_data(data=data, grouping=grouping, log_bool=log_transform)
    if submit_params_btn:
        col2.success("Success")
except Exception:
    group = pd.DataFrame()
    if submit_data_btn:
        parameter_form.error("Grouping unsuccessful. Check data input file.")
group_expander = parameter_form.expander("Grouped data preview")

with group_expander:
    group_expander.dataframe(group)

# Model ---------------------------------------------------
model_expader = st.expander("Model")
if submit_params_btn:
    with st.spinner('Training Model...'):
        train, val, model, pred, results = fm.run_arima_routine_st(data=group,
                                                                   grouping=grouping,
                                                                   forecast_periods=f_period,
                                                                   confidence_int=confidence_int,
                                                                   periods_search=search_strength,
                                                                   log_bool=log_transform,
                                                                   live=live)

    model_expader.success('Model Trained')
    csv = convert_df(results)
    model_expader.write(results)
    model_expader.download_button(label="Download data as CSV", data=csv, file_name='output.csv', mime='text/csv',)

    # Plotting results
    true = pd.concat([round(np.exp(train),2), round(np.exp(val),2)], axis=0)
    true_mod = pd.concat([true, round(np.exp(model.predict()),2)], axis=1)
    all = pd.concat([true_mod, round(np.exp(pred),2)], axis=0)
    all.columns = ['true', 'predicted', 'forecast']
    all.plot()
    model_expader.pyplot()
else:
    model_expader.info("Input from sidebar expected")


# Documentation --------------------------------------------
st.sidebar.header("Documentation")
document_expander = st.sidebar.expander("Search Documentation")
with document_expander:
    code_link = '[Source code](https://github.com/TedOiler/forecasting-app)'
    document_expander.markdown(code_link, unsafe_allow_html=True)
    documentation_link = '[Documentation](https://github.com/TedOiler/forecasting-app)'
    document_expander.markdown(documentation_link, unsafe_allow_html=True)