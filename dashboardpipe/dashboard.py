from load_data import tables, conversation, stringToTimestamp, all_ids, allTags, db, tagset
from utilities import split_data
from preprocess import get_data
from nontext_feature_preprocessing import NontextFeature
from nontext_feature import feature
from status_model import SimpleTreeModel

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from collections import defaultdict
from datetime import datetime, timedelta, date


import pickle
import boto3
import argparse
import streamlit as st 


np.set_printoptions(precision = 4)
# ========================================
#===========================================
#========= load data and process ===========
#===========================================

@st.cache(suppress_st_warning=False)
def create_df():

    completed_id = all_ids().completed
    incomplete_id = all_ids().incomplete
    initiated_only_id = all_ids().initiated_only
    compl_status_untagged_id = all_ids().compl_status_untagged

    # find out those id that have more than one tags in [completed, incomplete, initiated_only]
    multi_tagged_comp_status = []
    for i in completed_id:
        if (i in incomplete_id) or (i in initiated_only_id):
            multi_tagged_comp_status.append(i)
    for i in incomplete_id:
        if i in initiated_only_id:
            multi_tagged_comp_status.append(i)
            
    # clean 
    id_with_issue = [str(2388427861)]
    exclude = multi_tagged_comp_status + id_with_issue

    # remove multi-tagged data
    completed_id_clean = [i for i in completed_id if i not in exclude]
    incomplete_id_clean = [i for i in incomplete_id if i not in exclude]
    initiated_only_id_clean = [i for i in initiated_only_id if i not in exclude]

    # create nontext feature dataframe
    ntf_df_comp = NontextFeature(completed_id_clean).df
    ntf_df_incomp = NontextFeature(incomplete_id_clean).df
    ntf_df_init = NontextFeature(initiated_only_id_clean).df
    ntf_df_untagged = NontextFeature(compl_status_untagged_id).df
    adf = pd.concat([ntf_df_comp, ntf_df_incomp, ntf_df_init, ntf_df_untagged], ignore_index=True)
    adf = adf.sample(frac=1).reset_index(drop=True).copy(deep=True)
    for i in range(adf.shape[0]):
        if adf['completion_status'][i] == 'Incompl_init_only':
            adf.at[i, 'init_indicator'] = 1
        else:
           adf.at[i, 'init_indicator'] = 0
        if adf['completion_status'][i] == 'Completed':
            adf.at[i, 'completion_indicator'] = 1
        else:
            adf.at[i, 'completion_indicator'] = 0

    # +++++++add predicted initiated-only status and comp/incomp status++++++++++++

    # feature matrix
    fm = adf.drop(['id', 'start_time', 'end_time', 'completion_status', 'init_indicator', 'completion_indicator'], axis=1).copy()

    bucket = '[redacted_bucket_name]'
    init_model_key = 'saved_model/init_simple_tree.pkl'
    comp_model_key = 'saved_model/comp_simple_tree.pkl'

    init_tree = SimpleTreeModel()
    init_tree.load_model(bucket, init_model_key)
    adf['pred_init_indicator'] = init_tree.pred(fm)

    comp_tree = SimpleTreeModel()
    comp_tree.load_model(bucket, comp_model_key)
    comp_feature = adf[adf['pred_init_indicator']==0].copy().drop(['id', 'start_time', 'end_time', 'completion_status', 'init_indicator', 'completion_indicator', 'pred_init_indicator'], axis=1).copy()
    comp_pred = comp_tree.pred(comp_feature)

    j = 0
    for i in range(adf.shape[0]):
        if adf['pred_init_indicator'][i] == 1:
            adf.at[i, 'pred_comp_indicator'] = 0
        else:
            if comp_pred[j] == 0:
                adf.at[i, 'pred_comp_indicator'] = 0
                j += 1
            else:
                adf.at[i, 'pred_comp_indicator'] = 1
                j += 1


    for i in range(adf.shape[0]):
        if adf['pred_init_indicator'][i] == 1:
            adf.at[i, 'pred_completion_status'] = 'Initiated only'
        if adf['pred_init_indicator'][i] == 0 and adf['pred_comp_indicator'][i] == 0:
            adf.at[i, 'pred_completion_status'] = 'Incomplete'
        if adf['pred_comp_indicator'][i] == 1:
            adf.at[i, 'pred_completion_status'] = 'Completed'
    
    return adf

st.cache(suppress_st_warning=False)
def get_current_tag():
    return tagset().current_tag

adf = create_df()


current_tag = get_current_tag()




#=========================================
#========== dashboard layout =============
#=========================================




def intro_section():
    """This section introduces the app"""
    st.markdown(
        """
# Empower Work
"""
    )

intro_section()

st.sidebar.text('Select the date range:')
db_start_date = min(pd.to_datetime(db.conv_df['start_time']))
db_end_date = max(pd.to_datetime(db.conv_df['start_time']))
select_start = st.sidebar.date_input('Start date:', db_start_date)
select_end = st.sidebar.date_input('End date:', db_end_date)

st.sidebar.text("")
comp_status_method = st.sidebar.radio("Select a method for completion status classification:", ('manual tagging', 'rule-based tagging'))




# ###########   Dashboard part   #############
st.subheader('Dashboard')





# Number of conversations overtime

st.markdown('''
# Conversations Count
''')

# ========================RULE-BAED CODE======================================

if comp_status_method == 'rule-based tagging':
    display_firstday = select_start
    display_lastday = select_end
    #data_firstday_dt = datetime.strptime(data_firstday, '%Y-%m-%d %H:%M:%S')

    conv_ct_itvl = st.selectbox("Select interval size: ", ("Day", "Week", "Month", "Year"), index = 1)
    conv_ct_breakdown = st.radio("Show completion status breakdown within bars: ", ['No', 'Yes'])

    w_size = '604800000' # 604800000 is the number of milliseconds in a week 
    d_size = 'D'
    m_size = 'M1'
    y_size = 'M12'

    def conv_ct_fig(breakdown_size = '604800000'): 
        if breakdown_size == '604800000':
            # find out the first day of the week of the first day in data  
            fig_start_date = display_firstday - timedelta(days = display_firstday.weekday())
        if breakdown_size == 'M12':
            fig_start_date = date(display_firstday.year, 1, 1)
        if breakdown_size == 'D':
            fig_start_date = display_firstday
        if breakdown_size == 'M1':
            fig_start_date = display_firstday.replace(day=1)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x = adf[adf['pred_comp_indicator'].isin([1])]['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            #marker_color='#330C73',
            marker_color = 'royalblue',
            opacity=0.75,
            name = 'Completed'
            ))
        fig.add_trace(go.Histogram(
            x = adf[adf['pred_comp_indicator'].isin([0]) & adf['pred_init_indicator'].isin([0])]['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            marker_color='red',
            opacity=0.75,
            name = 'Incomplete'
            ))
        fig.add_trace(go.Histogram(
            x = adf[adf['pred_init_indicator'].isin([1])]['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            marker_color='lightpink',
            opacity=0.75,
            name = 'Initiated only'
            ))   
        
        fig.update_layout(
            title_text='Number of Conversations over Time', # title of plot
            xaxis_title_text='Date', # xaxis label
            yaxis_title_text='Conversation Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1, # gap between bars of the same location coordinates
            barmode = 'stack'
        )
        return fig


    def conv_ct_no_breakdown_fig(breakdown_size = '604800000'):
        if breakdown_size == '604800000':
            # find out the first day of the week of the first day in data  
            fig_start_date = display_firstday - timedelta(days = display_firstday.weekday())
        if breakdown_size == 'M12':
            fig_start_date = date(display_firstday.year, 1, 1)
        if breakdown_size == 'D':
            fig_start_date = display_firstday
        if breakdown_size == 'M1':
            fig_start_date = display_firstday.replace(day=1)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x = db.conv_df['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            marker_color='#330C73',
            opacity=0.75
            ))
        fig.update_layout(
            title_text='Number of Conversations over Time', # title of plot
            xaxis_title_text='Date', # xaxis label
            yaxis_title_text='Conversation Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1 # gap between bars of the same location coordinates
        )
        return fig



    w_ct_fig = conv_ct_fig(breakdown_size = w_size)
    m_ct_fig = conv_ct_fig(breakdown_size = m_size)
    d_ct_fig = conv_ct_fig(breakdown_size = d_size)
    y_ct_fig = conv_ct_fig(breakdown_size = y_size)

    w_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = w_size)
    m_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = m_size)
    y_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = y_size)
    d_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = d_size)

    if conv_ct_breakdown == 'Yes':
        if conv_ct_itvl == "Week": 
            st.plotly_chart(w_ct_fig)
        if conv_ct_itvl == "Month":
            st.plotly_chart(m_ct_fig)
        if conv_ct_itvl == 'Day':
            st.plotly_chart(d_ct_fig)
        if conv_ct_itvl == "Year":
            st.plotly_chart(y_ct_fig)
    if conv_ct_breakdown == 'No':
        if conv_ct_itvl == "Week": 
            st.plotly_chart(w_ct_no_breakdown_fig)
        if conv_ct_itvl == "Month":
            st.plotly_chart(m_ct_no_breakdown_fig)
        if conv_ct_itvl == 'Day':
            st.plotly_chart(d_ct_no_breakdown_fig)
        if conv_ct_itvl == "Year":
            st.plotly_chart(y_ct_no_breakdown_fig)
#=======================END RULE-BASED CODE======================================================


# =================== Manual tagging code ======================================
if comp_status_method == 'manual tagging':
    display_firstday = select_start
    display_lastday = select_end
    #data_firstday_dt = datetime.strptime(data_firstday, '%Y-%m-%d %H:%M:%S')

    conv_ct_itvl = st.selectbox("Select interval size: ", ("Day", "Week", "Month", "Year"), index = 1)
    conv_ct_breakdown = st.radio("Show completion status breakdown within bars: ", ['No', 'Yes'])

    w_size = '604800000' # 604800000 is the number of milliseconds in a week 
    d_size = 'D'
    m_size = 'M1'
    y_size = 'M12'

    def conv_ct_fig(breakdown_size = '604800000'): 
        if breakdown_size == '604800000':
            # find out the first day of the week of the first day in data  
            fig_start_date = display_firstday - timedelta(days = display_firstday.weekday())
        if breakdown_size == 'M12':
            fig_start_date = date(display_firstday.year, 1, 1)
        if breakdown_size == 'D':
            fig_start_date = display_firstday
        if breakdown_size == 'M1':
            fig_start_date = display_firstday.replace(day=1)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x = adf[adf['completion_status'].isin(['Completed'])]['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            #marker_color='#330C73',
            marker_color = 'royalblue',
            opacity=0.75,
            name = 'Completed'
            ))
        fig.add_trace(go.Histogram(
            x = adf[adf['completion_status'].isin(['Incomplete'])]['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            marker_color='red',
            opacity=0.75,
            name = 'Incomplete'
            ))
        fig.add_trace(go.Histogram(
            x = adf[adf['completion_status'].isin(['Incompl_init_only'])]['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            marker_color='lightpink',
            opacity=0.75,
            name = 'Incompl_init_only'
            ))   
        fig.add_trace(go.Histogram(
            x = adf[adf['completion_status'].isin(['no_compl_status_tag'])]['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            marker_color='gray',
            opacity=0.75,
            name = 'no_compl_status_tag'
            ))
        fig.update_layout(
            title_text='Number of Conversations over Time', # title of plot
            xaxis_title_text='Date', # xaxis label
            yaxis_title_text='Conversation Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1, # gap between bars of the same location coordinates
            barmode = 'stack'
        )
        return fig


    def conv_ct_no_breakdown_fig(breakdown_size = '604800000'):
        if breakdown_size == '604800000':
            # find out the first day of the week of the first day in data  
            fig_start_date = display_firstday - timedelta(days = display_firstday.weekday())
        if breakdown_size == 'M12':
            fig_start_date = date(display_firstday.year, 1, 1)
        if breakdown_size == 'D':
            fig_start_date = display_firstday
        if breakdown_size == 'M1':
            fig_start_date = display_firstday.replace(day=1)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x = db.conv_df['start_time'],
            xbins = dict(
                end = display_lastday,
                size = breakdown_size,
                start = fig_start_date
            ),
            marker_color='#330C73',
            opacity=0.75
            ))
        fig.update_layout(
            title_text='Number of Conversations over Time', # title of plot
            xaxis_title_text='Date', # xaxis label
            yaxis_title_text='Conversation Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1 # gap between bars of the same location coordinates
        )
        return fig



    w_ct_fig = conv_ct_fig(breakdown_size = w_size)
    m_ct_fig = conv_ct_fig(breakdown_size = m_size)
    d_ct_fig = conv_ct_fig(breakdown_size = d_size)
    y_ct_fig = conv_ct_fig(breakdown_size = y_size)

    w_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = w_size)
    m_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = m_size)
    y_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = y_size)
    d_ct_no_breakdown_fig = conv_ct_no_breakdown_fig(breakdown_size = d_size)

    if conv_ct_breakdown == 'Yes':
        if conv_ct_itvl == "Week": 
            st.plotly_chart(w_ct_fig)
        if conv_ct_itvl == "Month":
            st.plotly_chart(m_ct_fig)
        if conv_ct_itvl == 'Day':
            st.plotly_chart(d_ct_fig)
        if conv_ct_itvl == "Year":
            st.plotly_chart(y_ct_fig)
    if conv_ct_breakdown == 'No':
        if conv_ct_itvl == "Week": 
            st.plotly_chart(w_ct_no_breakdown_fig)
        if conv_ct_itvl == "Month":
            st.plotly_chart(m_ct_no_breakdown_fig)
        if conv_ct_itvl == 'Day':
            st.plotly_chart(d_ct_no_breakdown_fig)
        if conv_ct_itvl == "Year":
            st.plotly_chart(y_ct_no_breakdown_fig)
# =================== END Manual tagging code ======================================


st.markdown('''
# Overall Completion Rate
''')

# overall completion rate
#================================RULE-BASED CODE=======================================
if comp_status_method == 'rule-based tagging':
    compl_rate_incl_untag_fig = px.pie(adf[(adf['start_time'] >= select_start) & (adf['start_time'] <= select_end)],
                                names = 'pred_completion_status', title = 'Rule-based tagged completion status: {} to {}'.format(select_start, select_end, color = 'pred_completion_status'))


    comp_rate_exclude_option = st.multiselect(
        'View overall completion rate excluding:',
        ['Initiated only']
    )

    if comp_rate_exclude_option == ['Initiated only']:
        comp_select_df = adf[adf['pred_completion_status'].isin(['Completed', 'Incomplete'])]
    
    if comp_rate_exclude_option == []:
        comp_select_df = adf

    completion_rate_fig = px.pie(comp_select_df[(comp_select_df['start_time'] >= select_start) & (comp_select_df['start_time'] <= select_end)],
                                names = 'pred_completion_status', 
                                title = 'Overall Completion Rate: {} to {}'.format(select_start, select_end), 
                                color = 'pred_completion_status', 
                                color_discrete_map={'Completed':'royalblue', 'Incomplete':'red', 'Initiated only':'lightpink'})


    st.plotly_chart(completion_rate_fig)
#=============================END RULE-BASED CODE======================================


# =================== Manual tagging code ======================================
if comp_status_method == 'manual tagging':
    compl_rate_incl_untag_fig = px.pie(adf[(adf['start_time'] >= select_start) & (adf['start_time'] <= select_end)],
                                names = 'completion_status', title = 'Manually tagged completion status: {} to {}'.format(select_start, select_end, color = 'completion_status'))



    comp_rate_exclude_option = st.multiselect(
        'View overall completion rate excluding:',
        ['Untagged', 'Initiated only']
    )

    if comp_rate_exclude_option == ['Untagged', 'Initiated only']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete'])]
    if comp_rate_exclude_option == ['Untagged']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete', 'Incompl_init_only'])]
    if comp_rate_exclude_option == ['Initiated only']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete', 'no_compl_status_tag'])]
    if comp_rate_exclude_option == []:
        comp_select_df = adf

    completion_rate_fig = px.pie(comp_select_df[(comp_select_df['start_time'] >= select_start) & (comp_select_df['start_time'] <= select_end)],
                                names = 'completion_status', 
                                title = 'Overall Completion Rate: {} to {}'.format(select_start, select_end), 
                                color = 'completion_status', 
                                color_discrete_map={'Completed':'royalblue', 'Incomplete':'red', 'no_compl_status_tag':'lightgray', 'Incompl_init_only':'lightpink'})


    st.plotly_chart(completion_rate_fig)
# =================== END Manual tagging code ======================================


# Completion rate over time

st.write('Completion Rate Over Time')

#==============================RULE-BASED CODE============================================
if comp_status_method == 'rule-based tagging':

    comp_rate_overtime_exclude_option = st.multiselect(
        'View completion rate over time excluding:',
        ['Initiated only']
    )

    if comp_rate_overtime_exclude_option == ['Initiated only']:
        comp_select_df = adf[adf['pred_completion_status'].isin(['Completed', 'Incomplete'])]
    if comp_rate_overtime_exclude_option == []:
        comp_select_df = adf

    compl_rate_breakdown = st.selectbox("Break down by:", ("Month", "Week"))

    mm = comp_select_df['start_time'].dt.to_period('M')
    dd = comp_select_df[(comp_select_df['start_time'] >= select_start) & (comp_select_df['start_time'] < select_end)][['start_time', 'pred_comp_indicator']].copy()
    dd.index = dd['start_time']

    if compl_rate_breakdown == "Month":
        fff = dd.resample('M').mean()
        label_m = []
        for i in range(fff.shape[0]):
            label_m.append(str(fff.index.to_period('M').year[i]) + '-' + str(fff.index.to_period('M').month[i]))
        m_fig = go.Figure(data=go.Scatter(x = label_m, y=fff['pred_comp_indicator'], mode='markers', text = label_m), 
                    layout = {"width": 800, "height" : 500, "xaxis": {"title": "Date"}, "yaxis": {"range":[0.0, 1.0], "title":"Completion Rate"}})
        st.plotly_chart(m_fig)
    if compl_rate_breakdown == "Week":
        www = dd.resample('W').mean()
        label_w = []
        for i in range(www.shape[0]):
            label_w.append('Week starting: ' + str(www.index.to_period('W').year[i]) + '-' +
                            str(www.index.to_period('W').month[i]) + '-' + 
                            str(www.index.to_period('W').day[i]))
        w_fig = go.Figure(data=go.Scatter(x = label_w, y=www['pred_comp_indicator'], mode='markers'), 
                        layout = {"width": 800, "height" : 640, "xaxis": {"title": "Date"}, "yaxis": {"range":[0.0, 1.0], "title":"Completion Rate"}})
        st.plotly_chart(w_fig)
#=============================END RULED-BASED CODE===========================================


# =================== Manual tagging code ======================================
if comp_status_method == 'manual tagging':

    comp_rate_overtime_exclude_option = st.multiselect(
        'View completion rate over time excluding:',
        ['Untagged', 'Initiated only']
    )

    if comp_rate_overtime_exclude_option == ['Untagged', 'Initiated only']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete'])]
    if comp_rate_overtime_exclude_option == ['Untagged']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete', 'Incompl_init_only'])]
    if comp_rate_overtime_exclude_option == ['Initiated only']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete', 'no_compl_status_tag'])]
    if comp_rate_overtime_exclude_option == []:
        comp_select_df = adf

    compl_rate_breakdown = st.selectbox("Break down by:", ("Month", "Week"))

    mm = comp_select_df['start_time'].dt.to_period('M')
    dd = comp_select_df[(comp_select_df['start_time'] >= select_start) & (comp_select_df['start_time'] < select_end)][['start_time', 'completion_indicator']].copy()
    dd.index = dd['start_time']

    if compl_rate_breakdown == "Month":
        fff = dd.resample('M').mean()
        label_m = []
        for i in range(fff.shape[0]):
            label_m.append(str(fff.index.to_period('M').year[i]) + '-' + str(fff.index.to_period('M').month[i]))
        m_fig = go.Figure(data=go.Scatter(x = label_m, y=fff['completion_indicator'], mode='markers', text = label_m), 
                    layout = {"width": 800, "height" : 500, "xaxis": {"title": "Date"}, "yaxis": {"range":[0.0, 1.0], "title":"Completion Rate"}})
        st.plotly_chart(m_fig)
    if compl_rate_breakdown == "Week":
        www = dd.resample('W').mean()
        label_w = []
        for i in range(www.shape[0]):
            label_w.append('Week starting: ' + str(www.index.to_period('W').year[i]) + '-' +
                            str(www.index.to_period('W').month[i]) + '-' + 
                            str(www.index.to_period('W').day[i]))
        w_fig = go.Figure(data=go.Scatter(x = label_w, y=www['completion_indicator'], mode='markers'), 
                        layout = {"width": 800, "height" : 640, "xaxis": {"title": "Date"}, "yaxis": {"range":[0.0, 1.0], "title":"Completion Rate"}})
        st.plotly_chart(w_fig)
# =================== END Manual tagging code ======================================



# Issue tracking

st.markdown('''
# Issue Tracking
''')

view_tag_by = st.radio('Select tagging scheme option:', ('View all tags used in the database within selected date range (tags are not categorized)', 'View tags by tagging scheme (tags are categorized; may not include recently added tags)' ))


if view_tag_by == 'View all tags used in the database within selected date range (tags are not categorized)':

    st.markdown('''
    ### Disregarding tagging scheme - showing all tags appearing in the database within selected date range:
    ''')
    st.write(current_tag)

    st.markdown(
        '''
        ### View all tags ordered by frequency
        '''
    )
    conversations = db.conv_df

    select = conversations[(pd.to_datetime(conversations['start_time'])>select_start) & (pd.to_datetime(conversations['start_time'])<select_end)]
    
    # view most frequent tags
    # n_most_frequent = st.slider('Showing most frequent tags:', min_value = 1, max_value=len(current_tag), value=10)

    all_tag_freq_df = db.conv_tag_df[db.conv_tag_df['conv_id'].isin(select['conv_id'].values)]
    all_tag_freq_df = pd.merge(all_tag_freq_df, adf, how='left', left_on='conv_id', right_on='id')[['id', 'tag', 'completion_status', 'pred_completion_status']].dropna()
    default_exclude_tag = ['1.1 Completed', '1.2 Incomplete', '1.3 Incomplete: Initiated only', '2.1 Core Issue', '2.2 Ideal Outcome', '2.3 Action Plan', '3.0 Exclude','zSentSurvey','z-repeat',
    'zReturnAfterIncomplete', 'zReturnAfterCompleted', 'z Test Rule: New Msg', 'zbug', 'untagged', 'Return Texter New Topic', 'zReturnAfterInitiated']
    excluede_tag = st.multiselect('Exclude (type to add):', list(current_tag), default=default_exclude_tag)
    #brkdwn1 = st.radio('show completion status breakdown', ('No', 'Yes'))
    
    all_tags_freq_fig = px.histogram(all_tag_freq_df[~all_tag_freq_df.tag.isin(excluede_tag)], x="tag", color_discrete_sequence=['#330C73']).update_xaxes(categoryorder = 'total descending')
    st.plotly_chart(all_tags_freq_fig)

    # view tags by selection
    st.markdown(
        '''
        ### View individual tag counts
        '''
    )
    selected_tags = st.multiselect('Select tags:', current_tag)
    freq_by_tags_df = db.conv_tag_df[db.conv_tag_df['conv_id'].isin(select['conv_id'].values) & db.conv_tag_df['tag'].isin(selected_tags)]
    freq_by_tags_df = pd.merge(freq_by_tags_df, adf, how='left', left_on='conv_id', right_on='id') # data has issue: this dataframe includes data with date 'Sep 21, 1677'
    freq_by_tags_df = freq_by_tags_df[['id', 'tag', 'completion_status', 'pred_completion_status']].dropna()
    #st.write(freq_by_tags_df[freq_by_tags_df['id'].isnull()])
    brkdwn2 = st.radio('show completion status breakdown:', ('No', 'Yes'))
    if brkdwn2 == 'No':
        tags_freq_fig = px.histogram(freq_by_tags_df, x="tag", color_discrete_sequence=['#330C73']).update_xaxes(categoryorder = 'total descending')
        st.plotly_chart(tags_freq_fig)
    if brkdwn2 == 'Yes':
        if selected_tags:
            if comp_status_method == 'manual tagging':
                tags_freq_fig = px.histogram(
                    freq_by_tags_df, x="tag", 
                    color = 'completion_status', 
                    color_discrete_map={'Completed':'royalblue', 'Incomplete':'red', 'no_compl_status_tag':'lightgray', 'Incompl_init_only':'lightpink'}
                    ).update_xaxes(categoryorder = 'total descending')
                st.plotly_chart(tags_freq_fig)
            if comp_status_method == 'rule-based tagging':
                tags_freq_fig = px.histogram(
                    freq_by_tags_df, x="tag", 
                    color = 'pred_completion_status', 
                    color_discrete_map={'Completed':'royalblue', 'Incomplete':'red', 'Initiated only':'lightpink'}
                    ).update_xaxes(categoryorder = 'total descending')
                st.plotly_chart(tags_freq_fig)

    


if view_tag_by == 'View tags by tagging scheme (tags are categorized; may not include recently added tags)':


    st.write("Number of Conversations by Tag")

    tag_cat_option = st.selectbox('Select a tag category:', ('Issue', 'People', 'Emotion', 'Factor', 'Industry', 'Texter'))
    @st.cache(suppress_st_warning=False)
    def create_tag_dict():
        ts = tagset()
        return ts 

    tag_category = tag_cat_option
    conversations = db.conv_df
    tags = create_tag_dict()
    select = conversations[(pd.to_datetime(conversations['start_time'])>select_start) & (pd.to_datetime(conversations['start_time'])<select_end)]
    tagset = tags.by_cat(tag_category)
    ttt = db.conv_tag_df[db.conv_tag_df['conv_id'].isin(select['conv_id'].values) & db.conv_tag_df['tag'].isin(tagset)]
    issue_fig = px.histogram(ttt, x="tag").update_xaxes(categoryorder = 'total descending')
    st.write('Tags under this category: ')
    st.write(tagset)
    st.plotly_chart(issue_fig)

# Completion rate vs tag
st.markdown(
    '''
    # Completion Rate by Tag
    '''
    )



#tags_to_choose = list(tags.by_cat('Emotion')) + list(tags.by_cat('Issue')) + list(tags.by_cat('People')) + list(tags.by_cat('Factor')) + list(tags.by_cat('Industry')) + list(tags.by_cat('Texter'))

tag_option = st.multiselect('Select tags to view completion rate:\n (You may type to find a tag.)', current_tag)

#============================RULE-BASED CODE=========================================
if comp_status_method == 'rule-based tagging':

    comp_rate_by_tag_exclude_option = st.multiselect(
        'View completion rate by tag excluding:',
        ['Initiated only']
    )
    if comp_rate_by_tag_exclude_option == ['Initiated only']:
        comp_select_df = adf[adf['pred_completion_status'].isin(['Completed', 'Incomplete'])]
    if comp_rate_by_tag_exclude_option == []:
        comp_select_df = adf

    if len(tag_option) == 0:
        st.write('No tag selected.')
    if len(tag_option) != 0:
        
        #comp_by_tag_fig = make_subplots(rows = len(tag_option), cols = 1)
        for i in range(len(tag_option)):
            tag_name = tag_option[i]
            id_by_tag = db.conv_tag_df[db.conv_tag_df['tag'] == tag_name].conv_id.unique()
            if len(id_by_tag) == 0:
                st.write('The selected tag \"{}\" was not used in the database in the selected date range.'.format(tag_name))
            else:
                conversations = db.conv_df
                select = comp_select_df[(comp_select_df['start_time']>=select_start) & (comp_select_df['start_time']<=select_end) & (comp_select_df['id'].isin(id_by_tag))]
                tag_comp_rate_fig = px.pie(select,  names = 'pred_completion_status', title = 'Completion rate by tag - {}'.format(tag_name), color = 'pred_completion_status', color_discrete_map={'Completed':'royalblue', 'Incomplete':'red', 'Initiated only':'lightpink'})
                tag_comp_rate_fig.update_traces(textposition='inside', textinfo='percent+label')
                tag_comp_rate_fig.update_layout(height=500, width=500, title_text=tag_name)
                st.plotly_chart(tag_comp_rate_fig)
#===========================END RULE-BASED CODE======================================


#=========================== Manual tagging code ======================================
if comp_status_method == 'manual tagging':
    comp_rate_by_tag_exclude_option = st.multiselect(
        'View completion rate by tag excluding:',
        ['Untagged', 'Initiated only']
    )
    if comp_rate_by_tag_exclude_option == ['Untagged', 'Initiated only']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete'])]
    if comp_rate_by_tag_exclude_option == ['Untagged']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete', 'Incompl_init_only'])]
    if comp_rate_by_tag_exclude_option == ['Initiated only']:
        comp_select_df = adf[adf['completion_status'].isin(['Completed', 'Incomplete', 'no_compl_status_tag'])]
    if comp_rate_by_tag_exclude_option == []:
        comp_select_df = adf

    if len(tag_option) == 0:
        st.write('No tag selected.')
    if len(tag_option) != 0:
        
        #comp_by_tag_fig = make_subplots(rows = len(tag_option), cols = 1)
        for i in range(len(tag_option)):
            tag_name = tag_option[i]
            id_by_tag = db.conv_tag_df[db.conv_tag_df['tag'] == tag_name].conv_id.unique()
            if len(id_by_tag) == 0:
                st.write('The selected tag \"{}\" was not used in the database in the selected date range.'.format(tag_name))
            else:
                conversations = db.conv_df
                select = comp_select_df[(comp_select_df['start_time']>=select_start) & (comp_select_df['start_time']<=select_end) & (comp_select_df['id'].isin(id_by_tag))]
                tag_comp_rate_fig = px.pie(select,  names = 'completion_status', title = 'Completion rate by tag - {}'.format(tag_name), color = 'completion_status', color_discrete_map={'Completed':'royalblue', 'Incomplete':'red', 'no_compl_status_tag':'lightgray', 'Incompl_init_only':'lightpink'})
                tag_comp_rate_fig.update_traces(textposition='inside', textinfo='percent+label')
                tag_comp_rate_fig.update_layout(height=500, width=500, title_text=tag_name)
                st.plotly_chart(tag_comp_rate_fig)
#=========================== END Manual tagging code ======================================

# Tracking incoming message volume by day/time

st.markdown('''
# Traffic Tracking
''')



messages = db.messages_df
messages['message_date'] = pd.to_datetime(messages['message_date'])
select = messages[(messages['direction'] == 'Inbound') & (messages['message_date']>select_start) & (messages['message_date']<select_end)]
message_day_time = select[['message_date']].groupby([select['message_date'].dt.weekday, select['message_date'].dt.hour]).count()

message_day_time.index = message_day_time.index.set_names(['day of week', 'hour'])

message_day_time = message_day_time.unstack(level=0).unstack(level=0).fillna(0)

count_range = max(message_day_time) + 100

h = 24
weekday_message = {}
weekday_message['Mon'] = message_day_time[0:h]
weekday_message['Tue'] = message_day_time[h:2*h]
weekday_message['Wed'] = message_day_time[2*h:3*h]
weekday_message['Thu'] = message_day_time[3*h:4*h]
weekday_message['Fri'] = message_day_time[4*h:5*h]
weekday_message['Sat'] = message_day_time[5*h:6*h]
weekday_message['Sun'] = message_day_time[6*h:]


days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekday_msg = {}
for dd in days:
    weekday_msg[dd] = pd.DataFrame({'message count':weekday_message[dd].values.astype(int), 'hour of day':np.arange(1,len(weekday_message[dd])+1 )})

volume_tracking_fig = make_subplots(rows=7, cols=1, subplot_titles=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

for i in range(7):
    volume_tracking_fig.append_trace(go.Scatter(
    y = weekday_msg[days[i]]['message count'].values,
    x = weekday_msg[days[i]]['hour of day'].values, 
    name=days[i]
    ), row = i+1, col = 1)

volume_tracking_fig.update_yaxes(range=[0, count_range])
volume_tracking_fig.update_xaxes(title_text='hour', title_standoff = 0)
volume_tracking_fig.update_yaxes(title_text='messages')
volume_tracking_fig.update_layout(height=1200, width=800, title_text="Volume of Incoming Messages by Day and Hour")
st.plotly_chart(volume_tracking_fig)


# Volunteer activities

st.markdown('''
# Volunteer Activities
''')
volunteer_list = db.messages_df.attributed_to.unique()
slt_vlt = st.selectbox('Select a volunteer to view recent conversations (you may type to find a volunteer)', volunteer_list) #selected volunteer
slt_vlt_conv_id = db.messages_df[db.messages_df['attributed_to'] == slt_vlt].conversation_id.unique() #conversations associated with the selected volunteer
st.text('This volunteer has {} conversations in the database \n (from {} to {})'.format(len(slt_vlt_conv_id), db_start_date, db_end_date))


vlt_conv_num_show = st.slider('Showing most recent conversations:', min_value = 1, max_value=min(len(slt_vlt_conv_id), 100), value=3)
exclude = st.radio('Excluding initiated only?', ['Yes', 'No'])


slt_vlt_df = db.conv_df[db.conv_df.conv_id.isin(slt_vlt_conv_id)].drop(['completion_status'], axis=1).copy()
slt_vlt_df = pd.merge(slt_vlt_df, adf[['id', 'completion_status', 'pred_completion_status']], left_on = 'conv_id', right_on = 'id')
slt_vlt_df['start_time'] = pd.to_datetime(slt_vlt_df['start_time'])

if exclude == 'No':
    slt_vlt_df_no_exclude = slt_vlt_df.sort_values(by = ['start_time'], ascending=False)
    st.write(slt_vlt_df_no_exclude.drop(['completion_status', 'pred_completion_status', 'id'], axis = 1).head(vlt_conv_num_show))
if exclude == 'Yes' and comp_status_method == 'manual tagging':
    slt_vlt_df_exclude = slt_vlt_df[slt_vlt_df.completion_status.isin(['Completed', 'Incomplete'])].sort_values(by = ['start_time'], ascending=False)
    st.write(slt_vlt_df_exclude.drop(['completion_status', 'pred_completion_status', 'id'], axis = 1).head(vlt_conv_num_show))
if exclude == 'Yes' and comp_status_method == 'rule-based tagging':
    slt_vlt_df_exclude = slt_vlt_df[slt_vlt_df.pred_completion_status.isin(['Completed', 'Incomplete'])].sort_values(by = ['start_time'], ascending=False)
    st.write(slt_vlt_df_exclude.drop(['completion_status', 'pred_completion_status', 'id'], axis = 1).head(vlt_conv_num_show))





# Exploration part

st.subheader('Exploration')

ttr_to_completion_fig1 = go.Figure()
ttr_to_completion_fig1.add_trace(go.Histogram(
     
    x=adf[adf['completion_status'] == 'Completed']['in_resp_rate'].values, 
    name='Completed',
    xbins=dict(
        start=0, 
        end=600, 
        size=20
    ),
    marker_color='#EB89B5',
    opacity=0.75
))
ttr_to_completion_fig1.add_trace(go.Histogram(
     
    x=adf[adf['completion_status'] == 'Incomplete']['in_resp_rate'].values, 
    name='Incomplete',
    xbins=dict(
        start=0, 
        end=800, 
        size=20
    ),
    marker_color='#330C73',
    opacity=0.75
))
ttr_to_completion_fig1.add_trace(go.Histogram(
     
    x=adf[adf['completion_status'] == 'Incompl_init_only']['in_resp_rate'].values, 
    name='Incomplete - initiated only',
    xbins=dict(
        start=0, 
        end=600, 
        size=20
    ),
    marker_color='red',
    opacity=0.75
))

ttr_to_completion_fig1.update_layout(
    title_text='Time to Respond by Completion Status',
    xaxis_title_text='texter time to respond (seconds)',
    yaxis_title_text = 'conversation count',
    #gargap=2
)
ttr_to_completion_fig1.show()
st.plotly_chart(ttr_to_completion_fig1)


