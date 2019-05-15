# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:52:48 2019

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pyltr
from pyltr.models import LambdaMART

def clean(df):
    #print('original shape = ', df.shape)
    #print(round(df.isnull().sum()/len(df),2))
    df = df.drop(['date_time','site_id','gross_bookings_usd',
                  'visitor_hist_starrating','visitor_hist_adr_usd',
                  'srch_query_affinity_score', 'position',
                  'comp1_rate','comp1_inv','comp1_rate_percent_diff',
                  'comp2_rate','comp2_inv','comp2_rate_percent_diff',
                  'comp3_rate','comp3_inv','comp3_rate_percent_diff',
                  'comp4_rate','comp4_inv','comp4_rate_percent_diff',
                  'comp5_rate','comp5_inv','comp5_rate_percent_diff',
                  'comp6_rate','comp6_inv','comp6_rate_percent_diff',
                  'comp7_rate','comp7_inv','comp7_rate_percent_diff',
                  'comp8_rate','comp8_inv','comp8_rate_percent_diff',
                  ],axis = 1) # drop irrelevant columns
    df['score'] = df.click_bool + 4*df.booking_bool
    df.replace(np.NaN,value=-1,inplace = True) # change NaN into -1
    df = df[df.price_usd < 2000]
    df = df.drop(['booking_bool','click_bool'],axis = 1)
    #print('cleaned shape = ', df.shape)
    #print(round(df.isnull().sum()/len(df),2))
    return(df)
    
def clean_test(df):
    #print('original shape = ', df.shape)
    #print(round(df.isnull().sum()/len(df),2))
    df = df.drop(['date_time','site_id',
                  'visitor_hist_starrating','visitor_hist_adr_usd',
                  'srch_query_affinity_score',
                  'comp1_rate','comp1_inv','comp1_rate_percent_diff',
                  'comp2_rate','comp2_inv','comp2_rate_percent_diff',
                  'comp3_rate','comp3_inv','comp3_rate_percent_diff',
                  'comp4_rate','comp4_inv','comp4_rate_percent_diff',
                  'comp5_rate','comp5_inv','comp5_rate_percent_diff',
                  'comp6_rate','comp6_inv','comp6_rate_percent_diff',
                  'comp7_rate','comp7_inv','comp7_rate_percent_diff',
                  'comp8_rate','comp8_inv','comp8_rate_percent_diff',
                  ],axis = 1) # drop irrelevant columns
    df.replace(np.NaN,value=-1,inplace = True) # change NaN into -1
    #print('cleaned shape = ', df.shape)
    #print(round(df.isnull().sum()/len(df),2))
    return(df) 

def exploration(df):
    print('separate searches: ',len(df.srch_id.value_counts()))
    plt.figure(figsize=[14,12])
    #-----------------
    plt.subplot(3,3,1)
    plt.title('prices per night')
    plt.hist(df.price_usd, bins = 100)
    plt.xlim((0,1000))
    plt.ylim((0,800000))
    plt.hist(df.score, bins = 10)
    #---------------
    a = df.score.value_counts()
    x = ['0','1','5']
    y = [a[0],a[1],a[5]]
    plt.subplot(3,3,2)
    plt.title('score frequencies in train set')
    plt.bar(x,y)
    #--------------
    plt.subplot(3,3,3)
    plt.title('distribution of length of stay')
    plt.hist(df.srch_length_of_stay, bins = 57)
    #---------------
    plt.subplot(3,3,4)
    plt.title('distr. of location scores')
    plt.hist(df.prop_location_score1,bins = 10)
    #---------------------
    plt.subplot(3,3,5)
    a = df.promotion_flag.value_counts()
    x = ['0','1']
    y = [a[0],a[1]]
    plt.title('frequency of promotions')
    plt.bar(x,y)
    #--------------------------------  
    a = df.srch_adults_count.value_counts()
    x = ['1','2','3','4','5','6','7','8','9']
    y = [a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9]]
    plt.title('amount of adults in booking')
    plt.subplot(3,3,6)
    plt.bar(x,y)

def form_train_set(df,S):
    sample = df.iloc[0:S]
    good = sample[sample.score >0]
    sample = sample[sample.score <1]
    # below is a sample command ensuring multiple samples from each query
    bad = sample.groupby('srch_id').apply(lambda x: x.sample(2, replace = True)).reset_index(drop=True)
    # otherwise uncomment the line below for standard random sample
    # bad = df.sample(S, random_state = 132) # Random sample
    q = [good,bad]
    train = pd.concat(q)
    train = train.sort_values('srch_id')
    train = train.reset_index(drop = True)
    qids = np.array(train.srch_id)
    groups = np.array(train.srch_id.value_counts(sort=True,ascending=True).sort_index())
    y = np.array(train.score)
    train = train.drop(['srch_id','score'],axis = 1)
    X = np.array(train)
    return(X,y,qids,groups)

def form_valid_set(df,S):
    valid = df.iloc[S:len(df)]
    valid = valid.reset_index(drop = True)
    y = np.array(valid.score)
    qids = np.array(valid.srch_id)
    groups = np.array(valid.srch_id.value_counts(sort=True,ascending=True).sort_index())
    valid = valid.drop(['srch_id','score'],axis = 1)
    X = np.array(valid)
    return(X,y,qids,groups)


df = pd.read_csv('training_set_VU_DM.csv') # read in training data
df = clean(df) # clean training data
# exploration(df)
S = 500000 # Enter samplesize of trainingset
[TX,Ty,Tqids,Tgroups] = form_train_set(df,S) # form trainingset
[VX,Vy,Vqids,Vgroups] = form_valid_set(df,S) # form validationset



df_test = pd.read_csv('test_set_VU_DM.csv') # read in test data
df_test = clean_test(df_test) # clean test data
srch_id = df_test.srch_id
prop_id = df_test.prop_id
df_test = df_test.drop(['srch_id'],axis = 1)
EX = np.array(df_test)

def LMART():
    metric = pyltr.metrics.NDCG(k=10) # set parameters for LambdaMART
    monitor = pyltr.models.monitors.ValidationMonitor(
            VX, Vy, Vqids, metric=metric, stop_after=100)    
    model_LM = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=100,
    learning_rate=0.1,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=15,
    verbose=1
    ) # end parameters
    model_LM.fit(X=TX,y=Ty,qids=Tqids,monitor=monitor) # fit model    
    predLMART = model_LM.predict(EX)
    return(predLMART,monitor)



def XGBoost(TX,Ty,Tqids,VX,Vy,Vqids):
    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
          'gamma': 1.0, 'min_child_weight': 0.1,
          'max_depth': 6, 'n_estimators': 100}
    model = xgb.sklearn.XGBRanker(**params)
    model.fit(TX, Ty, Tgroups, eval_set=[(VX, Vy)], eval_group=[Vgroups])
    predXGBoost = model.predict(EX)
    return(predXGBoost)

def XGBoostLin(TX,Ty,Tqids,VX,Vy,Vqids):
    params = {'booster': 'gblinear', 'objective': 'rank:pairwise', 'learning_rate': 0.1,
          'gamma': 1.0, 'min_child_weight': 0.1,
          'max_depth': 6, 'n_estimators': 100}
    model = xgb.sklearn.XGBRanker(**params)
    model.fit(TX, Ty, Tgroups, eval_set=[(VX, Vy)], eval_group=[Vgroups])
    predXGBoost = model.predict(EX)
    return(predXGBoost)

#def create_result(p,srch_id,prop_id)


#[p,monitor] = LMART()
p1 = XGBoost(TX,Ty,Tgroups,VX,Vy,Vgroups)
p = XGBoostLin(TX,Ty,Tgroups,VX,Vy,Vgroups)
p2 = (p+p1)/2 # taking average of two gradient boosted algorithms

prediction = pd.DataFrame(p2,columns=['prob'])
prediction = prediction.join(srch_id)
prediction = prediction.join(prop_id)

result = prediction.sort_values(by=['srch_id', 'prob'], ascending=[True, False])
result.reset_index(drop = True, inplace = True)
result = result.drop('prob',axis =1)
result.to_csv('submissionGR45_XGBOOST_COMBI.csv',index = False)

ptree = pd.read_csv('submissionGR45_XGBOOST.csv')
plin = pd.read_csv('submissionGR45_XGBOOST_lin.csv')


