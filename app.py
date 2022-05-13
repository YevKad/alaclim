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
st.markdown('by Yev Kadranov')
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
    df_winter['month']=df_winter['date'].dt.strftime('%b')
    df_winter['date_rel']=pd.to_datetime(df_winter['date'].dt.strftime('2016-%m-%d'),format='%Y-%m-%d')
    df_winter.loc[df_winter['mon']>8,'date_rel']=df_winter['date_rel'] - pd.DateOffset(years=1)

    df_winter.loc[df_winter['date']<datetime(1972,6,1), 'period']='1922-1972'
    df_winter.loc[df_winter['date']>datetime(1972,6,1), 'period']='1972-2022'

    df_fdd=df_winter.groupby(['seas'])['fd'].sum().reset_index(name='fdd')
    df_winter['dt']=df_winter['date'].dt.strftime('%Y%m').astype(int)
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


dist_type = st.selectbox(
    'Distribution Type',
     ['Normal','Gamma'])
dist_string=f'Distribution Type: {dist_type}'
st.write(dist_string)

df_dist_lst=[]
for peri in ['1922-1972','1972-2022']:
    df_per=df_winter[df_winter["period"]==peri]
    x_pdf=np.linspace(df_per['at'].min(),df_per['at'].max())
    if dist_type=='Gamma':

        [a_fit,loc_fit,scale_fit]=scipy.stats.gamma.fit(df_per["at"])
        y_pdf=scipy.stats.gamma.pdf(x_pdf,a_fit,loc=loc_fit,scale=scale_fit)

        dist_params_s=f'**a={a_fit:.3f}, loc={loc_fit:.3f}, scale={scale_fit:.3f}**'

    elif dist_type=='Normal':
        [mean_fit,std_fit]=scipy.stats.norm.fit(df_per["at"])
        y_pdf=scipy.stats.norm.pdf(x_pdf,mean_fit,std_fit)
        dist_params_s=f'**Mean={mean_fit:.3f}, SD={std_fit:.3f}**'
    st.markdown(f'{peri}. Distribution Parameters: '+dist_params_s)
    df_dist=pd.DataFrame({'x':x_pdf,'y':y_pdf})
    df_dist['period']=peri
    df_dist_lst.append(df_dist)
df_dist=pd.concat(df_dist_lst)

fig_hist=go.Figure()
fig_hist.add_trace(px.histogram(df_winter, x="at", color="period", barmode="overlay",
labels={'seas':'Winter Season (Nov-Apr)','at':'Air Temperature (\u00B0C)'},range_x=[-35,35],histnorm='probability density').data[0])
# fig_hist.update_xaxes(range = [-35,35])
fig_hist.add_trace(px.histogram(df_winter, x="at", color="period", barmode="overlay",
labels={'seas':'Winter Season (Nov-Apr)','at':'Air Temperature (\u00B0C)'},range_x=[-35,35],histnorm='probability density').data[1])
fig_hist.add_trace(px.line(df_dist,x='x',y='y', color="period").data[0])
fig_hist.add_trace(px.line(df_dist,x='x',y='y', color="period").data[1])
fig_hist.update_yaxes(title='Probability Density')
fig_hist.update_xaxes(title='Air Temperature (\u00B0C)')
st.write(fig_hist)
st.markdown('''Above :arrow_up: comparison of Distributions and fitted PDF for periods from 1922-1972 and 1972-2022 Show that there are less extreme cold observations for last 50 years (1972-2022).

Mean Air Temperature for the period 1972-2022 is  0.6 degree warmer than for the period 1922-1972
''')
fig_histan=px.histogram(df_winter, x="at", animation_frame="seas",
           range_x=[-35,35], range_y=[0,60],
           labels={'seas':'Winter Season (Nov-Apr)','at':'Air Temperature (\u00B0C)'})
st.write(fig_histan)
mon_lst=df_winter['month'].unique().tolist()

st_mon_lst = st.multiselect("Months Used", mon_lst, default=mon_lst)

step=st.slider('Group years ', min_value=1, max_value=50, value=10, step=1)
date_range=np.array([datetime(1922+i,6,1,0,0).strftime('%Y%m') for i in range(step,101,step)]).astype(int)
date_bin_n=np.array([f'{1922+i}-{1922+step+i}' for i in range(0,100,step)])
df_winter_g=df_winter.copy()
df_winter_g['date_bin']=date_bin_n[np.digitize(df_winter_g['dt'], date_range)]

df_mon=df_winter_g[df_winter_g['month'].isin(st_mon_lst)]
fig_box = px.box(df_mon, x="date_bin", y="at",
                labels={'seas':'Winter Season','at':'Air Temperature (\u00B0C)','date_bin':'Date Range'},)
fig_box.update_layout(width=800,height=600)
st.write(fig_box)

st.markdown('''Above :arrow_up: Box Plot represents Air Temperature distribution in a given year group. If winters are grouped every 10 years, median air temperature steadily increases from 1972-1982.
''')

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
st.markdown('**Top 5 Warmest Winter for the past 100 years by FDD:**')
st.dataframe(df_fdd.sort_values(by=['fdd']).head(5).reset_index(drop=True))
st.markdown('**Top 5 Coldest Winter for the past 100 years by FDD:**')
st.dataframe(df_fdd.sort_values(by=['fdd'],ascending=[False]).head(5).reset_index(drop=True))
