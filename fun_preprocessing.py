import numpy as np
import pandas as pd
import streamlit as st

@st.cache()
def read_clean_data(path='./input/inflows-20_21.xlsx'):
    # Read dataset according to path
    data = pd.read_excel(path)

    # Select specific columns
    data_frame = data[['Document Date', 'Total amount', 'DEAL ID', 'Apartment']]

    # fix column naming
    data_frame.columns = data_frame.columns.str.replace(' ', '_')
    data_frame.columns = [x.lower() for x in data_frame.columns]
    data_frame['document_date'] = pd.to_datetime(data_frame['document_date'], format="%Y-%m-%d")

    return data_frame

@st.cache()
def clean_data(data):
    data_frame = data[['Document Date', 'Total amount', 'DEAL ID', 'Apartment']]
    # fix column naming
    data_frame.columns = data_frame.columns.str.replace(' ', '_')
    data_frame.columns = [x.lower() for x in data_frame.columns]
    data_frame['document_date'] = pd.to_datetime(data_frame['document_date'], format="%Y-%m-%d")

    return data_frame

@st.cache()
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

@st.experimental_memo()
def train_val_test_split(data, live=True):
    if live:
        train_df = data.iloc[:-4]
        test_df = data.iloc[-4:]
        return train_df, test_df
    else:
        train_df = data.iloc[:-8]
        val_df = data.iloc[-8:-4]
        test_df = data.iloc[-4:]
        return train_df, val_df, test_df

@st.cache()
def write_results(prediction, confidence, path='./output/', log_bool=True):
    if log_bool:
        data = pd.concat([round(np.exp(prediction),2), round(np.exp(confidence),2)], axis=1)
    else:
        data = pd.concat([round(prediction,2), round(confidence,2)], axis=1)

    data.to_csv(path+'results.csv', sep='\t', encoding='utf-8')
    return 0

@st.cache()
def write_results_st(prediction, confidence, path='./output/', log_bool=True):
    if log_bool:
        data = pd.concat([round(np.exp(prediction),2), round(np.exp(confidence),2)], axis=1)
    else:
        data = pd.concat([round(prediction,2), round(confidence,2)], axis=1)

    return data