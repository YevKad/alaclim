import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy

from datetime import datetime
st.header('Winter Climate Analysis in Almaty, Kazakhstan')
st.markdown('''
Winter seasons **1st November to 30th April** from **1922 to 2022**.

Data Courtesy: 1922-2005 [NOAA Climate Data](https://www.ncdc.noaa.gov/cdo-web/), 2005-2022 [RP5](https://rp5.ru/)
''')
st.map(pd.DataFrame({'city':['Almaty'],'latitude':[43.2331],'longitude':[76.9331]}),
zoom=6)
winter_fl='./data/ala1922-2022decapr.csv'
@st.cache
def get_winter_data(fl):

    df_winter=pd.read_csv(fl,parse_dates=['date'])
    df_winter['fd']=np.abs(np.minimum(0,df_winter['at']))
    df_winter['date_rel']=pd.to_datetime(df_winter['date'].dt.strftime('2016-%m-%d'),format='%Y-%m-%d')
    df_winter.loc[df_winter['mon']>8,'date_rel']=df_winter['date_rel'] - pd.DateOffset(years=1)

    df_winter.loc[df_winter['date']<datetime(1972,6,1), 'period']='1922-1972'
    df_winter.loc[df_winter['date']>datetime(1972,6,1), 'period']='1972-2022'

    df_fdd=df_winter.groupby(['seas'])['fd'].sum().reset_index(name='fdd')

    return df_winter,df_fdd
df_winter,df_fdd=get_winter_data(winter_fl)
seas_lst=df_fdd['seas'].unique().tolist()
# st_ms = st.sidebar.multiselect("Event Seasons", seas_lst, default=seas_lst)

fig_lin=px.line(df_winter, x="date_rel", y="at",color='seas',
                labels={'seas':'Winter Season (Nov-Apr)','at':'Air Temperature (\u00B0C)',
                        'date_rel':'Day'})

fig_lin.update_xaxes(
    tickformat="%b-%d")
st.write(fig_lin)

fig_hist=px.histogram(df_winter, x="at", color="period", barmode="overlay",
labels={'seas':'Winter Season (Nov-Apr)','at':'Air Temperature (\u00B0C)'})
fig_hist.update_xaxes(range = [-35,35])
st.write(fig_hist)


fig_histan=px.histogram(df_winter, x="at", animation_frame="seas",
           range_x=[-35,35], range_y=[0,60],
           labels={'seas':'Winter Season (Nov-Apr)','at':'Air Temperature (\u00B0C)'})
st.write(fig_histan)
fig_box = px.box(df_winter, x="seas", y="at",
                labels={'seas':'Winter Season (Nov-Apr)','at':'Air Temperature (\u00B0C)'},)
fig_box.update_layout(width=800,height=600)
st.write(fig_box)

st.subheader('Freezing Degree Days')
st.markdown('''
**Freezing Degree Days (FDD)** - is an absolute sum of negative daily mean air temperature
and is commonly used as an indicator of winter severity: colder winters have higher FDD.

FDD is estimated with the following equasion:
''')
st.latex(r'''
FDD=\sum{|min(0,at_i)|}
''')
st.markdown('''Where *at* is daily mean air temperature

In this analysis, FDD is estimated for date range from 1st November to 30th April.

Plot below shows that FDD Trend is descending, meaning that winters in Almaty are generally warming.
.''')


poly_deg=st.slider('Degree of a Polynomial Trend', min_value=1, max_value=20, value=1, step=1)
fig_fdd=go.Figure()
trend_x = np.arange(df_fdd.shape[0]) # = array([0, 1, 2, ..., 3598, 3599, 3600])
trend_fit = np.polyfit(trend_x, df_fdd['fdd'], poly_deg)
trend_fit_fn = np.poly1d(trend_fit)

fig_fdd.add_trace(px.line(x=df_fdd['seas'],y=df_fdd['fdd']).data[0])
fig_fdd.add_trace(px.line(x=df_fdd['seas'],y=trend_fit_fn(trend_x)).data[0])
fig_fdd['data'][1]['line']['color']="black"
fig_fdd.update_yaxes(title='FDD (\u00B0C)')
fig_fdd.update_xaxes(title='Winter Season (Nov-Apr)')
st.write(fig_fdd)
