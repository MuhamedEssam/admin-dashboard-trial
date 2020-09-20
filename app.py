# -*- coding: utf-8 -*-
import base64
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import pathlib
from dash.dependencies import Input, Output, State
from scipy import stats
import xlrd
import plotly.express as px
import dash_table
import pyrebase
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()

from firebase_admin import credentials
from firebase_admin import firestore
import os 

########################################## Talking to Firebase 
# credential_path = 'D:\\Projects\\Nitros Application v2\\fire.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
# cred = credentials.Certificate(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
# firebase_admin.initialize_app(cred)
# db = firestore.client()
# Actions = list(db.collection(u'Nitrous').document(u'Actions').collection(u'Actions').stream())
# Actions = list(map(lambda x: x.to_dict(), Actions))

######################################### Data Wrangling 
df = pd.read_excel('dataframe.xlsx')
df.loc[df['username'].isnull(),'username'] = df['userName']
#Dropping
df=df.drop(['formattedDate','endMonth','endDay','endYear','id','status','value','userName','userID','factor','status'],1)
#Renaming
df['powerBar'] = df['powerBar'].replace([1.0],'Int.Comm')
df['powerBar'] = df['powerBar'].replace([2.0],'Ext.Comm')
df['powerBar'] = df['powerBar'].replace([3.0],'Learn')
df['powerBar'] = df['powerBar'].replace([4.0],'Tech')
df['powerBar'] = df['powerBar'].replace([5.0],'Reletive')
df['powerBar'] = df['powerBar'].replace([6.0],'Teach')
df['powerBar'] = df['powerBar'].replace([0],'Break')
# Replacing -ve value with 120 mins
df['duration']=df['duration'].mask(df['duration'] < 0, 120)
# Dropping Early Dirty 42 Rows
df=df[42:]
df=df.dropna()
df.head(5)
df=df.reset_index(drop=True)
df["username"]=df["username"].str.lower()

########################################################## Dash        

fig1 =px.sunburst(df, path=['title', 'topic'], values='duration')
fig2 =px.sunburst(df, path=['username'], values='duration')

fig1.update_layout(
    title={
        'text': "Cumulative Users Time Distribution",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    })

fig2.update_layout(
    title={
        'text':'Documentation Time',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    })
    
app =dash.Dash()
app.layout=html.Div([
   
    html.Div([html.A([html.H2('Nitrous Dashboard'),html.Img(src='/assets/ants.png')], href='https://www.antscoin.org/')],className="banner"),
#     html.Div([dcc.Dropdown(id='demo-dropdown',
#         options=[
#             {'label': 'Hegazy', 'value': 0},
#             {'label': 'Nada', 'value': 1},
#             {'label': 'Admin', 'value': 'Admin'}],value= 0 ),
#               html.Img(id='image',style={'height': '100%','width': '100%'})],
#                                      style={'margin-bottom': '10px','textAlign':'center','width':'220px','margin':'auto'}),
    
    
    

     html.Div([html.Div(dcc.Graph(id="Pie1",figure=fig1))],className="five columns"),
     html.Div([html.Div(dcc.Graph(id="Pie2",figure=fig2))],className="five columns"),
#     html.Div([html.Div(dcc.Graph(id="Violin"))],className="ten columns"),
#     html.Div([html.Div(dcc.Graph(id="Table"))],className="ten columns")

    
  ])




if __name__ == "__main__":
    app.run_server(debug=True)