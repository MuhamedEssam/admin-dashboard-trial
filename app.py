# Admin Dashboard 
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
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os 
import threading
import sklearn
import xlrd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# ###################################################### Talking to Firebase 
# credential_path = 'D:\\Projects\\Nitros Application v2\\fire.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
# cred = credentials.Certificate(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
# firebase_admin.initialize_app(cred)
# db = firestore.client()
# Actions = list(db.collection(u'Nitrous').document(u'Actions').collection(u'Actions').stream())
# Actions = list(map(lambda x: x.to_dict(), Actions))

######################################################### Python

df = pd.read_excel('dataframe.xlsx')
#Merging 
df.loc[df['username'].isnull(),'username'] = df['userName']
df=df.drop(['formattedDate','endMonth','endDay','endYear','id','status','value','userName','userID','factor','status'],1)
df.dropna()
#Renaming
df["username"] = df["username"].str.lower()
df['powerBar'] = df['powerBar'].replace([1.0],'Int.Comm')
df['powerBar'] = df['powerBar'].replace([2.0],'Ext.Comm')
df['powerBar'] = df['powerBar'].replace([3.0],'Learn')
df['powerBar'] = df['powerBar'].replace([4.0],'Tech')
df['powerBar'] = df['powerBar'].replace([5.0],'Reletive')
df['powerBar'] = df['powerBar'].replace([6.0],'Teach')
df['powerBar'] = df['powerBar'].replace([0],'Break')
df['username'] = df['username'].replace('HEGAZY','hegazy')
df['username'] = df['username'].replace('hegazi','hegazy')
df['username'] = df['username'].replace( 'Nada hashim','nada hashim')
df['username'] = df['username'].replace( 'nahla.khaled','nahla khaled')
df['username'] = df['username'].replace('amansour','mansour' )
# Replacing -ve value with 120 mins
df['duration']=df['duration'].mask(df['duration'] < 0, 120)
df=df[42:]
df.dropna(subset=['powerBar'], inplace = True)
df=df.reset_index(drop=True)
users = df['username'].unique()
dataframes=[]
banned_list =['hossary','mostafa','mostaf','yyyy']
i=-1
Names=[]
for j, user in enumerate( users) :
    if not user in banned_list :
        i+=1

        Names.append(user)
        hk = df.groupby('username') 
        df_user=hk.get_group(user)
        df_user=df_user.reset_index(drop=True)
        
        df_user=df_user.rename(columns={'startYear':"year", 'startMonth':"month", 'startDay':"day", 'startHour':"hour", 'startMin':"minute"})
        df_user['year'] = df_user['year'].replace([20],2020)
        
        dataframes.append(df_user)
        # to export  to excel 
        #dataframes[i].to_excel(str(user)+'_df.xlsx')
N=len(users)-len(banned_list)
x=pd.to_datetime(dataframes[0][['year', 'month', 'day', 'hour', 'minute']])
g=pd.DataFrame(x,columns=['date'])
dataframes[0]=pd.concat([g,dataframes[0]],axis=1)
dataframes[0]['week_number_of_year'] = dataframes[0]['date'].dt.week
num=max(dataframes[0]['week_number_of_year'].values)-min(dataframes[0]['week_number_of_year'].values)
num=num+1

dataforml=[0] * N

l=[]
for v in range (35,35+num):
    l.append(v)
for k in range(N-1):
    dataforml[k] = pd.DataFrame(np.zeros((num, 9)), index =l)
    dataforml[k].columns = ["Int.Comm","Ext.Comm","Learn","Tech","Reletive","Break","Teach","Total Hours","UserName"]
    dataforml[k]['UserName'][:k+num]=Names[k]
    dataforml[k].index.name = 'week_number_of_year'
    
    
    
for k in range(N-1):
    
    gk = dataframes[k].groupby('powerBar')

    try :

        h=gk.get_group('Int.Comm') 
        h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
        h['week_number_of_year'] = h['date'].dt.week 
        v=h.groupby(["week_number_of_year"]).sum()/60
        dataforml[k]["Int.Comm"]=v['duration']
    except:
        pass
        
        

    try:
        h=gk.get_group('Ext.Comm') 
        h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
        h['week_number_of_year'] = h['date'].dt.week 
        v=h.groupby(["week_number_of_year"]).sum()/60
        dataforml[k]["Ext.Comm"]=v['duration']
    except:
        pass
    
    try:
        h=gk.get_group('Learn') 
        h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
        h['week_number_of_year'] = h['date'].dt.week 
        v=h.groupby(["week_number_of_year"]).sum()/60
        dataforml[k]["Learn"]=v['duration']
    except:
        pass
    try:
        
        h=gk.get_group('Tech') 
        h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
        h['week_number_of_year'] = h['date'].dt.week 
        v=h.groupby(["week_number_of_year"]).sum()/60
        dataforml[k]["Tech"]=v['duration']
        
    except:
        pass

    try:
        h=gk.get_group('Reletive') 
        h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
        h['week_number_of_year'] = h['date'].dt.week 
        v=h.groupby(["week_number_of_year"]).sum()/60
        dataforml[k]["Reletive"]=v['duration']
    except:
        pass
    try:

        h=gk.get_group('Break') 
        h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
        h['week_number_of_year'] = h['date'].dt.week 
        v=h.groupby(["week_number_of_year"]).sum()/60
        dataforml[k]["Break"]=v['duration']
    
    except:
        pass
    try:
        h=gk.get_group('Teach') 
        h['date']=pd.to_datetime(h[['year', 'month', 'day', 'hour', 'minute']])
        h['week_number_of_year'] = h['date'].dt.week 
        v=h.groupby(["week_number_of_year"]).sum()/60
        dataforml[k]["Teach"]=v['duration']
        
    except:
        pass

    dataframes[k].dropna(inplace=True)
    dataforml[k].fillna(0, inplace=True)

    dataforml[k]['Total Hours']=dataforml[k].iloc[:,:7].sum(axis=1)

#ML Part

data=pd.read_excel('Nitrous.xlsx')
Features=data.iloc[:,:8].values
Labels=data.iloc[:,9:].values
#Setting 20 % of the Data for Testing 
X_train, x_test_old, y_train, y_test_old = train_test_split(Features, Labels, test_size=0.20, random_state=4)
# Standardization
scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)

regressor = RandomForestRegressor(random_state=1)
regressor.fit(X_train,y_train)

Preds=[0] * N
Final=[0]*N
for k in range(N-1):

    X_pred=scaler.transform(dataforml[k].drop(['UserName'],axis=1))
    y_pred=regressor.predict(X_pred)
    Preds[k]=pd.DataFrame(y_pred,index=l)

    Preds[k].columns =["Work Ethics","Student Mentality","Self Management","Technical Skills","Interpersonal","LeaderShip"]
    Final[k]=pd.concat([dataforml[k],Preds[k]],axis=1)
    
    
current_states=[0]*N
for k in range (N-1):
    current_states[k]= Final[k].iloc[:,9:].sum(axis=0)+50
########################################################## Dash        
df.dropna(inplace=True)  
fig1 =px.sunburst(df, path=['startYear','startMonth','title', 'topic'], values='duration')
fig2 =px.sunburst(df, path=['startYear','startMonth','username'], values='duration')



categories = ["Work Ethics","Student Mentality","Self Management","Technical Skills","Interpersonal","LeaderShip"]

radar = go.Figure()
for k in range(N-1):
    radar.add_trace(go.Scatterpolar(
          r=current_states[k],
          theta=categories,
          fill='toself',
          name=Names[k] ))

radar.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 100]
    )),
  showlegend=True
)
radar.update_layout(
    title={
        'text': "Cumulative Users Power Bar Performance",
        'y':0.95,
        'x':0.48,
        'xanchor': 'center',
        'yanchor': 'top'
    })


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
    
app = dash.Dash(__name__)
server = app.server
app.title = 'Nitrous Admin Dashboard'
app.layout=html.Div([
   
    html.Div([html.A([html.H2('Nitrous Dashboard'),html.Img(src='/assets/ants.png')], href='https://www.antscoin.org/')],className="banner"),

    
    
     html.Div([html.Div(dcc.Graph(id="Radar",figure=radar))],className="twelve columns"),
     html.Div([html.Div(dcc.Graph(id="Pie1",figure=fig1))],className="five columns"),
     html.Div([html.Div(dcc.Graph(id="Pie2",figure=fig2))],className="five columns"),

#     html.Div([html.Div(dcc.Graph(id="Violin"))],className="ten columns"),
#     html.Div([html.Div(dcc.Graph(id="Table"))],className="ten columns")

    
  ])




if __name__ == '__main__':
    app.run_server(debug=True)