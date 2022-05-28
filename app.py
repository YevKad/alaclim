import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy
import pymannkendall as mk

from datetime import datetime

st.header('Winter Climate Analysis in Almaty, Kazakhstan')
st.markdown('''by [Yevgeniy Kadranov](https://www.linkedin.com/in/yevkad/)
from [ICEMAN.KZ](https://iceman.kz/)\n\n-----
Winter seasons **from 1st November to 30th April**, **1922 to 2022**.\n
Data Courtesy:
1922-2005 [NOAA Climate Data](https://www.ncdc.noaa.gov/cdo-web/),
2005-2022 [RP5](https://rp5.ru/)
''')

st.map(pd.DataFrame({'city':['Almaty'],
                    'latitude':[43.2331],'longitude':[76.9331]
                    }),
        zoom=6)
st.markdown("-----")
winter_fl='./data/ala1922-2022decapr.csv'
@st.cache
def get_winter_data(fl):

    df_winter=pd.read_csv(fl,parse_dates=['date'])
    df_winter['fd']=np.abs(np.minimum(0,df_winter['at']))
    df_winter['month']=df_winter['date'].dt.strftime('%b')
    #creating field with "relative" date to aggregate given time range for each day
    df_winter['date_rel']=pd.to_datetime(df_winter['date'].dt.strftime('2016-%m-%d'),format='%Y-%m-%d')
    df_winter.loc[df_winter['mon']>8,'date_rel']=df_winter['date_rel'] - pd.DateOffset(years=1)

    df_winter.loc[df_winter['date']<datetime(1972,6,1), 'period']='1922-1972'
    df_winter.loc[df_winter['date']>datetime(1972,6,1), 'period']='1972-2022'

    df_fdd=df_winter.groupby(['seas'])['fd'].sum().reset_index(name='fdd')
    df_fdd['fdd']=df_fdd['fdd'].astype(int)
    df_winter['dt']=df_winter['date'].dt.strftime('%Y%m').astype(int)

    return df_winter,df_fdd

df_winter,df_fdd=get_winter_data(winter_fl)

df_winter_agg=df_winter.groupby(['date_rel']).agg(
                                                {'at':['min','max','mean']}
                                                ).reset_index()

df_winter_agg.columns=['date','min','max','mean']

seas_lst=df_fdd['seas'].unique().tolist()
# st_ms = st.sidebar.multiselect("Event Seasons", seas_lst, default=seas_lst)

yr_s,yr_e=st.slider('Year Range ',
                    min_value=1922, max_value=2021, value=(2011,2021), step=1)
yr_e=yr_e+1
per_rng=f'{yr_s}-{yr_e}'

dt_s=datetime(yr_s,6,1)
dt_e=datetime(yr_e,6,1)

df_seas_sel=df_winter[(df_winter['date']>dt_s) & (df_winter['date']<dt_e)]

df_seas_sel_agg=df_seas_sel.groupby(['date_rel']).agg(
                                                {'at':['min','max','mean']}
                                                ).reset_index()

df_seas_sel_agg.columns=['date','min','max','mean']

fig_agg=px.line(df_seas_sel,x='date_rel',y='at', color="seas")
fig_agg.update_traces(opacity=min(0.8,10/(yr_e-yr_s)),line=dict(width=0.5))
fig_agg.add_trace(go.Scatter(x=df_winter_agg['date'], y=df_winter_agg['max'],
                                name='100 years Max',
                                legendrank=3,
                                line = dict(color='firebrick', width=4)))
fig_agg.add_trace(go.Scatter(x=df_winter_agg['date'], y=df_winter_agg['min'],
                                name='100 years Min',
                                legendrank=1,
                                line=dict(color='royalblue', width=4)))
fig_agg.add_trace(go.Scatter(x=df_winter_agg['date'], y=df_winter_agg['mean'],
                                name='100 years Avg',
                                legendrank=2,
                                line=dict(color='green', width=5)))

fig_agg.add_trace(go.Scatter(x=df_seas_sel_agg['date'], y=df_seas_sel_agg['mean'],
                                name=f'{per_rng} Avg',
                                legendrank=4,
                                line=dict(color='orange', width=5)))

fig_agg.update_xaxes(
    tickformat="%b-%d",title='Day')

fig_agg.update_yaxes(title='Air Temperature (\u00B0C)')
showleg = st.checkbox('Show Legend')
fig_agg.update_layout(showlegend=showleg)

st.plotly_chart(fig_agg, use_container_width=True)

st.markdown('''Above :arrow_up: timeseries shows development of 100 years
Min, Max and Mean (green line) Air Temperature on a particular day from Nov to Apr as well as
Develpment of selectable period Mean (orange line) and individual seasons.
The plot indicated that average of periods from about 1970s up to present
are generally above 100 years average, while periods prior to 1970s are below
100 years average.
''')

with st.echo():
    # Distribution Fitting

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

            a_fit,loc_fit,scale_fit=scipy.stats.gamma.fit(df_per["at"])
            y_pdf=scipy.stats.gamma.pdf(x_pdf,a_fit,loc=loc_fit,scale=scale_fit)
            dist_params_s=f'**a={a_fit:.3f}, loc={loc_fit:.3f}, scale={scale_fit:.3f}**'

        elif dist_type=='Normal':

            mean_fit,std_fit=scipy.stats.norm.fit(df_per["at"])
            y_pdf=scipy.stats.norm.pdf(x_pdf,mean_fit,std_fit)
            dist_params_s=f'**Mean={mean_fit:.3f}, SD={std_fit:.3f}**'

        st.markdown(f'{peri}. Distribution Parameters: '+dist_params_s)
        df_dist=pd.DataFrame({'x':x_pdf,'y':y_pdf})
        df_dist['period']=peri
        df_dist_lst.append(df_dist)

    df_dist=pd.concat(df_dist_lst)

fig_hist=go.Figure()
fig_hist.add_trace(px.histogram(df_winter, x="at", color="period",
                    barmode="overlay",
                    labels={'seas':'Winter Season (Nov-Apr)',
                            'at':'Air Temperature (\u00B0C)'},
                    range_x=[-35,35],histnorm='probability density').data[0])

fig_hist.add_trace(px.histogram(
                        df_winter, x="at", color="period", barmode="overlay",
                        labels={'seas':'Winter Season (Nov-Apr)',
                                'at':'Air Temperature (\u00B0C)'},
                                range_x=[-35,35],
                                histnorm='probability density'
                    ).data[1])
fig_hist.add_trace(px.line(df_dist,x='x',y='y', color="period").data[0])
fig_hist.add_trace(px.line(df_dist,x='x',y='y', color="period").data[1])
fig_hist.update_yaxes(title='Probability Density')
fig_hist.update_xaxes(title='Air Temperature (\u00B0C)')

st.write(fig_hist)
st.markdown('''Above :arrow_up: comparison of Distributions and
fitted PDF for periods from 1922-1972 and 1972-2022
Show that there are less extreme cold observations for the last 50 years (1972-2022).
\nMean Air Temperature for the period 1972-2022
is 0.6 degree warmer than for the period 1922-1972
''')
fig_histan=px.histogram(df_winter, x="at", animation_frame="seas",
           range_x=[-35,35], range_y=[0,60],
           labels={'seas':'Winter Season (Nov-Apr)',
                    'at':'Air Temperature (\u00B0C)'})

st.write(fig_histan)

with st.echo():

    # Years grouping:
    nmax=df_fdd.shape[0]
    yr_start=df_winter['year'].min()

    group_lst=[i for i in range(1,nmax+1,1) if nmax % i ==0] # list of only divisible groups
    step=st.selectbox('Group years ', group_lst, index=group_lst.index(10))

    date_range=np.array([datetime(yr_start+i,6,1).strftime('%Y%m')
                                for i in range(step,nmax+1,step)
                        ]).astype(int)

    date_bin_n=np.array([f'{yr_start+i}-{yr_start+step+i}' for i in range(0,nmax,step)])

    df_winter_g=df_winter.copy()
    df_winter_g['date_bin']=date_bin_n[np.digitize(df_winter_g['dt'], date_range)]

mon_lst=df_winter['month'].unique()
st_mon_lst = st.multiselect("Months Used", mon_lst, default=mon_lst)
df_mon=df_winter_g[df_winter_g['month'].isin(st_mon_lst)]

split_mon=st.checkbox('Split by Month')
if split_mon:
    color_mon='month'
    color_order={"month": ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']}
else:
    color_mon=None
    color_order=None

fig_box = px.box(df_mon, x="date_bin", y="at",color=color_mon,
                category_orders=color_order,
                labels={'seas':'Winter Season',
                        'at':'Air Temperature (\u00B0C)',
                        'date_bin':'Date Range'},)

# fig_box.update_layout(width=800,height=600)
st.write(fig_box)

st.markdown('''Above :arrow_up: Box Plot represents
Air Temperature distribution in a given year group.
If winters are grouped every 10 years,
median air temperature steadily increases from 1972-1982.
''')

st.subheader('Freezing Degree Days')

st.markdown('''
**Freezing Degree Days (FDD)** -
is an absolute sum of negative daily mean air temperature
and is commonly used as an indicator of
winter severity: colder winters have higher FDD.\n\n
FDD is estimated with the following equasion:
''')

st.latex(r'''
FDD=\sum{|min(0,at_i)|}
''')

st.markdown('''Where *at* is daily mean air temperature

For example, if during 5 days Daily Mean Air Temperature
was -4$^\circ$C, +2$^\circ$C, -3$^\circ$C, -2$^\circ$C, 0$^\circ$C,
than FDD is calculated as *4+0+3+2+0=9*\n\n
In this analysis, FDD is estimated for date range from 1st November to 30th April.\n\n
Plot below :arrow_down: shows that FDD Trend is descending, meaning that winters
in Almaty are generally warming.
''')

poly_deg=st.slider('Degree of a Polynomial Trend',
                    min_value=1, max_value=20, value=1, step=1)

trend_x = np.arange(df_fdd.shape[0])
trend_fit = np.polyfit(trend_x, df_fdd['fdd'], poly_deg)
trend_fit_fn = np.poly1d(trend_fit)

fig_fdd=go.Figure()
fig_fdd.add_trace(px.line(x=df_fdd['seas'],y=df_fdd['fdd']).data[0])
fig_fdd.add_trace(px.line(x=df_fdd['seas'],y=trend_fit_fn(trend_x)).data[0])
fig_fdd['data'][1]['line']['color']="black"
fig_fdd.update_yaxes(title='FDD (\u00B0C)')
fig_fdd.update_xaxes(title='Winter Season (Nov-Apr)')

st.write(fig_fdd)

st.markdown('''#### Mann-Kendall Test of FDD Trend
Mann-Kendall trend test is used to determine if there is a
statistically significant trend in a time series.\n\n
The test is performed with `pymannkendall`
[library](https://pypi.org/project/pymannkendall/) as following:
''')

fdd_yr_s,fdd_yr_e=st.slider('Year Range for MK Test',
                    min_value=1922, max_value=2021, value=(1922,2021), step=1)
fdd_yr_e=fdd_yr_e+1
fdd_seas_lst=[f'{i+fdd_yr_s}-{i+fdd_yr_s+1}' for i in range(fdd_yr_e-fdd_yr_s)]
df_fdd_seas=df_fdd[df_fdd['seas'].isin(fdd_seas_lst)].sort_values(by=['seas'])

with st.echo():
    if fdd_yr_e-fdd_yr_s>2:
        mk_res=mk.original_test(df_fdd_seas['fdd']) # results of Mann-Kendall Test

        trend_mk=mk_res.trend # Trend: Increasing, Decreasing or No Trend
        pval_mk=mk_res.p      # P-Value of MK Test

        if pval_mk<0.05: # alpha=0.05
            p_str='**statistically significant**'
            alpha_compr='smaller'
        else:
            p_str='**not statistically significant**'
            alpha_compr='greater'

        #Markdown string generated to present MK Test results of FDD Trend
        mk_str=f'''Mann-Kendall Test shows that FDD trend is **{trend_mk}**.
        \n\n**P-value** is **{pval_mk:.3e}**, which is **{alpha_compr}**
        than **0.05**, meaning that the trend in the data is {p_str}'''
    else:
        mk_str='''
        Insufficient data for MK test. Increase Year Range.
        '''

st.markdown(mk_str)
st.markdown('#### Winter Severity Ranking with FDD')

top_c=st.slider('Number of winters to rank: ',
                min_value=1, max_value=100, value=5, step=1)
st.markdown(f'**Top {top_c} Warmest and Coldest Winters for the past 100 years by FDD**')
col_w, col_c = st.columns((1,1))

with col_w:
    st.markdown(f'**Top {top_c} Warmest:**')
    st.dataframe(df_fdd.sort_values(by=['fdd']).head(top_c).reset_index(drop=True))
with col_c:
    st.markdown(f'**Top {top_c} Coldest:**')
    st.dataframe(df_fdd.sort_values(by=['fdd'],
                ascending=[False]).head(top_c).reset_index(drop=True))

(st.markdown('''-----\n\n
[GitHub repo](https://github.com/YevKad/alaclim)
'''))
