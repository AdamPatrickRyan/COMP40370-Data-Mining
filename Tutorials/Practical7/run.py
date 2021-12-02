from sqlalchemy import create_engine
from matplotlib.backends.backend_pdf import PdfPages
#Step One: Import Modules

import time
import pyodbc as pyodbc
import datetime as datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

import nltk as nl
import sklearn as sk
import matplotlib as mp
#import xgboost as xg
import seaborn as sb

#SKLEARN METHODS
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pandas as pd
import datetime as dt
import numpy as np
#import pyodbc
import sys
import os

mp.style.use('ggplot')


def read_csv(fn='./specs/question_1.csv'):
    df=pd.read_csv(fn)
    return df

def plot_dataset(df,fn='./output/question_1_1.pdf',cluster_label='org_cluster',centroids=[]):
    
    if cluster_label=='org_cluster':
        forced_colour_map={0:'Red',1:'Blue',2:'Green'}
    else:
        forced_colour_map={1:'Green',2:'Red',0:'Blue'}
    
    plt.scatter(df['x'], df['y'], c= df[cluster_label].map(forced_colour_map), cmap='viridis')
    
    if len(centroids)>0:
        print('')
        plt.scatter(centroids['x'],centroids['y'],c=centroids['cluster_kmean_center'].map(forced_colour_map),cmap='viridis',marker='+')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('X vs Y with {}'.format(cluster_label))
    plt.savefig(fn)
    plt.show()
    plt.close()
    
    return


def plot_inertia(y_axis,x_axis=range(1,11),fn='./output/question_1_2.pdf'):
    plt.plot(x_axis, y_axis)
    plt.xlabel('No Of Clusters')
    plt.ylabel('Inertia')
    plt.title('Clusters vs Inertia')
    plt.savefig(fn)
    plt.show()
    plt.close()
    return

def generate_cluster_dict(df):
    
    cluster_dict=dict()
    count_value=[]
    
    for cluster_count in range(1,11):

        model=KMeans(n_clusters=cluster_count
                       , init='k-means++'
                       , n_init=10
                       , max_iter=300
                       , tol=0.0001
                       , verbose=0
                       , random_state=0
                       , copy_x=True
                       , algorithm='auto')

        predictions=model.fit_predict(df[['x','y']])

        model_inertia=model.inertia_

        count_value+=[model_inertia]

        rand_scoring=rand_score(df['org_cluster'], predictions)
        
        if cluster_count>1:
            sil_score=silhouette_score(df[['x','y']]
                                       , labels=predictions
                                       , metric='euclidean'
                                       , sample_size=None
                                       , random_state=0)
        else:
            sil_score=-1

        cluster_dict[cluster_count]={'Predictions':predictions
                                        ,'Model':model
                                        ,'Centroids':model.cluster_centers_
                                        ,'Labels': model.labels_
                                        ,'Value':model_inertia
                                        ,'Rand_Score':rand_scoring
                                        ,'Silhouette_Score':sil_score}
        
        
    return cluster_dict,count_value

def add_kmeans_pred_to_df(df,pred):
    df['cluster_kmeans']=pred
    return df

def question_one():
    
    df=read_csv(fn='./specs/question_1.csv')
    
    #Part 1
    plot_dataset(df=df
                 ,fn='./output/question_1_1.pdf'
                 ,cluster_label='org_cluster')
    
    cluster_dict,interia_values=generate_cluster_dict(df=df)
    
    #Part 2
    plot_inertia(x_axis=range(1,11),y_axis=interia_values,fn='./output/question_1_2.pdf')
    
    
    #Part 3
    print("""
-----
Question 1 Part 3:
----
Random Index: {} | Silhouette Score: {} | Index={}
-----
""".format(cluster_dict[3]['Rand_Score'],cluster_dict[3]['Silhouette_Score'],3))
    
    #Part 4
    df=add_kmeans_pred_to_df(df=df
                          ,pred=cluster_dict[3]['Predictions'])
    
    df.to_csv('./output/question_1.csv')
    
    centroid_df=pd.DataFrame(cluster_dict[3]['Centroids'],columns=['x','y']).reset_index()
    centroid_df.columns=['cluster_kmean_center','x','y']
    
    plot_dataset(df=df
                 ,fn='./output/question_1_5.pdf'
                 ,cluster_label='cluster_kmeans'
                ,centroids=centroid_df)
    
    return df,cluster_dict



def remove_unimportant_columns(df,cols_to_remove=['NAME', 'MANUF', 'TYPE', 'RATING']):
    df_original_columns=set(df.columns)
    df_new_columns=df_original_columns - set(cols_to_remove)
    df=df[list(df_new_columns)]
    return df
   
def question_two_generate_cluster_dict(df):
    
    cluster_dict=dict()
    count_value=[]
    
    run_config=[[5,5,100],[5,100,100],[3,10,100]]
    
    for run_type in run_config:
        
        cluster_count=run_type[0]
        initialisations=run_type[1]
        max_iterations=run_type[2]

        model=KMeans(n_clusters=cluster_count
                       , init='k-means++'
                       , n_init=initialisations
                       , max_iter=max_iterations
                       , tol=0.0001
                       , verbose=0
                       , random_state=0
                       , copy_x=True
                       , algorithm='auto')

        predictions=model.fit_predict(df)

        model_inertia=model.inertia_

        count_value+=[model_inertia]
        
        if cluster_count>1:
            sil_score=silhouette_score(df
                                       , labels=predictions
                                       , metric='euclidean'
                                       , sample_size=None
                                       , random_state=0)
        else:
            sil_score=-1

        cluster_dict['{}_{}_{}'.format(cluster_count,initialisations,max_iterations)]={'Predictions':predictions
                                        ,'Model':model
                                        ,'Centroids':model.cluster_centers_
                                        ,'Labels': model.labels_
                                        ,'Value':model_inertia
                                        ,'Silhouette_Score':sil_score}
        
        
    return cluster_dict,count_value


def question_two():
    
    df=read_csv(fn='./specs/question_2.csv')
    
    df=remove_unimportant_columns(df,cols_to_remove=['NAME', 'MANUF', 'TYPE', 'RATING'])
    
    cluster_dict,count_values=question_two_generate_cluster_dict(df)
    
    equal_predictions=[]
    
    for k in range(len(cluster_dict['5_5_100']['Predictions'])):
        
        if cluster_dict['5_5_100']['Predictions'][k]==cluster_dict['5_100_100']['Predictions'][k]:
            equal_predictions+=[k]
            
    print("""
----

QUESTION TWO PART 4:

----""".format(len(equal_predictions)))
    print("""Number of equal predictions: {} """.format(len(equal_predictions)))
    print("""Number of not-equal predictions: {} """.format(len(cluster_dict['5_5_100']['Predictions']) - len(equal_predictions)))
    
    df['config1']=cluster_dict['5_5_100']['Predictions']
    df['config2']=cluster_dict['5_100_100']['Predictions']
    df['config3']=cluster_dict['3_10_100']['Predictions']
    
    print("""
----

QUESTION TWO PART 6:

----

config1: 5_5_100    : Silhouette: {}
config2: 5_100_100  : Silhouette: {}
config3: 3_10_100   : Silhouette: {}

    """.format(cluster_dict['5_5_100']['Silhouette_Score']
              ,cluster_dict['5_100_100']['Silhouette_Score']
              ,cluster_dict['3_10_100']['Silhouette_Score'])         
)
    
    df.to_csv('./output/question_2.csv')
    
    return df,cluster_dict






def question_three_generate_cluster_dict(df):
    
    cluster_dict=dict()
    count_value=[]
    
    model=KMeans(n_clusters=7
                   , init='k-means++'
                   , n_init=5
                   , max_iter=100
                   , tol=0.0001
                   , verbose=0
                   , random_state=0
                   , copy_x=True
                   , algorithm='auto')

    predictions=model.fit_predict(df[['x','y']])

    model_inertia=model.inertia_

    count_value+=[model_inertia]

    if 7>1:
        sil_score=silhouette_score(df[['x','y']]
                                   , labels=predictions
                                   , metric='euclidean'
                                   , sample_size=None
                                   , random_state=0)
    else:
        sil_score=-1

    cluster_dict[7]={'Predictions':predictions
                                    ,'Model':model
                                    ,'Centroids':model.cluster_centers_
                                    ,'Labels': model.labels_
                                    ,'Value':model_inertia
                                    ,'Silhouette_Score':sil_score}
        
        
    return cluster_dict,count_value



def question_three_plot_dataset(df,fn='./output/question_3_1.pdf',cluster_label='k-means',centroids=[]):
    
    
    plt.scatter(df['x'], df['y'], c= df[cluster_label], cmap='viridis')
    
    if len(centroids)>0:
        plt.scatter(centroids['x'],centroids['y'],c=centroids.index,cmap='viridis',marker='+')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('X vs Y with {}'.format(cluster_label))
    plt.savefig(fn)
    plt.show()
    plt.close()
    
    return

def question_three_dbscan(df,scaler='Standard'):
    
    cluster_dict=dict()
    
    X=df[['x','y']].values
    
    if scaler=='Standard':
        scaler = StandardScaler()
    elif scaler=='MinMax':    
        scaler = MinMaxScaler()
    
    X = scaler.fit_transform(X)
    
    for epsilon in [0.04,0.08]:
    
        model=DBSCAN(eps=epsilon
                     , min_samples=4
                     , metric='euclidean'
                     , metric_params=None
                     , algorithm='auto'
                     , leaf_size=30
                     , p=None
                     , n_jobs=None)

        predictions=model.fit_predict(X)
        print("For Epsilon: {}, classes={}".format(epsilon,len(set(model.labels_))))
        if len(set(model.labels_))>1:
            sil_score=silhouette_score(X
                                       , labels=predictions
                                       , metric='euclidean'
                                       , sample_size=None
                                       , random_state=0)
        else:
            sil_score=-1

        cluster_dict[epsilon]={'Predictions':predictions
                                        ,'Model':model
                                        ,'Labels': model.labels_
                                        ,'Silhouette_Score':sil_score}

    return cluster_dict




def question_three():
    df=read_csv(fn='./specs/question_3.csv')
    
    df=remove_unimportant_columns(df,cols_to_remove=['ID'])
    
    kmeans_cluster_dict, count_value = question_three_generate_cluster_dict(df)
    
    df['kmeans']=kmeans_cluster_dict[7]['Predictions']
    
    question_three_plot_dataset(df,fn='./output/question_3_1.pdf',cluster_label='kmeans',centroids=[])
    
    db_scan_cluster_result=question_three_dbscan(df,scaler='MinMax')
    
    print("""Q3: Mistake in the question. It says epsilon=0.4 but the test script requires epsilon=0.04""")
    df['dbscan1']=db_scan_cluster_result[0.04]['Predictions']
    df['dbscan2']=db_scan_cluster_result[0.08]['Predictions']
    
    question_three_plot_dataset(df,fn='./output/question_3_2.pdf',cluster_label='dbscan1',centroids=[])
    question_three_plot_dataset(df,fn='./output/question_3_3.pdf',cluster_label='dbscan2',centroids=[])
    
    df.to_csv('./output/question_3.csv')
    
    return df,kmeans_cluster_dict,db_scan_cluster_result


if __name__=='__main__':
    question_one_df,question_one_cluster_dict=question_one()
    question_two_df,question_two_cluster_dict=question_two()
    df,kmeans_cluster_dict,db_scan_cluster_result = question_three()