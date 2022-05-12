import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy

from datetime import datetime

winter_fl='./data/ala1922-2022decapr.csv'
@st.cache
def get_winter_data(fl):

    df_winter=pd.read_csv(fl,parse_dates=['date'])
    df_winter['fd']=np.abs(np.minimum(0,df_winter['at']))
    df_fdd=df_winter.groupby(['seas'])['fd'].sum().reset_index(name='fdd')

    return df_winter,df_fdd

df_winter,df_fdd=get_winter_data(winter_fl)
fig_fdd=px.line(x=df_fdd['seas'],y=df_fdd['fdd'])
st.write(fig_fdd)
