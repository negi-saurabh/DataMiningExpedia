# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:52:48 2019

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pyltr
from pyltr.models import LambdaMART

def clean(df):
    #print('original shape = ', df.shape)
    #print(round(df.isnull().sum()/len(df),2))
    df = df.drop(['date_time','site_id','gross_bookings_usd',
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

def form_train_set(df):
    sample = df.iloc[0:100000]
    good = sample[sample.score >0]
    sample = sample[sample.score <1]
    # below is a sample command ensuring multiple samples from each query
    bad = sample.groupby('srch_id').apply(lambda x: x.sample(3, replace = True)).reset_index(drop=True)
    # otherwise uncomment the line below for standard random sample
    # bad = df.sample(500000, random_state = 132) # Random sample
    q = [good,bad]
    train = pd.concat(q)
    train = train.sort_values('srch_id')
    train.reset_index(drop = True)
    # Uncomment section below to see score vs frequency.
    '''
    a = train.score.value_counts()
    b = ['0','1','5']
    c = [a[0],a[1],a[5]]
    plt.title('score frequencies in new train set')
    plt.bar(b,c)
    '''
    qids = np.array(train.srch_id)
    y = np.array(train.score)
    train = train.drop(['srch_id','score'],axis = 1)
    X = np.array(train)
    return(X,y,qids)
    
df = pd.read_csv('training_set_VU_DM.csv')
df = clean(df)
exploration(df)

[X,y,qids] = form_train_set(df)

metric = pyltr.metrics.NDCG(k=10)
    
model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(X=X,y=y,qids=qids)     

df_test = pd.read_csv('test_set_VU_DM.csv')
df_test = clean_test(df_test)
srch_id = df_test.srch_id
prop_id = df_test.prop_id
df_test = df_test.drop(['srch_id'],axis = 1)
X = np.array(df_test)

p = model.predict(X)
prediction = pd.DataFrame(p,columns=['prob'])
prediction = prediction.join(srch_id)
prediction = prediction.join(prop_id)
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:52:48 2019

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pyltr
from pyltr.models import LambdaMART

def clean(df):
    #print('original shape = ', df.shape)
    #print(round(df.isnull().sum()/len(df),2))
    df = df.drop(['date_time','site_id','gross_bookings_usd',
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

def form_train_set(df):
    sample = df.iloc[0:100000]
    good = sample[sample.score >0]
    sample = sample[sample.score <1]
    # below is a sample command ensuring multiple samples from each query
    bad = sample.groupby('srch_id').apply(lambda x: x.sample(3, replace = True)).reset_index(drop=True)
    # otherwise uncomment the line below for standard random sample
    # bad = df.sample(500000, random_state = 132) # Random sample
    q = [good,bad]
    train = pd.concat(q)
    train = train.sort_values('srch_id')
    train.reset_index(drop = True)
    # Uncomment section below to see score vs frequency.
    '''
    a = train.score.value_counts()
    b = ['0','1','5']
    c = [a[0],a[1],a[5]]
    plt.title('score frequencies in new train set')
    plt.bar(b,c)
    '''
    qids = np.array(train.srch_id)
    y = np.array(train.score)
    train = train.drop(['srch_id','score'],axis = 1)
    X = np.array(train)
    return(X,y,qids)
    
df = pd.read_csv('training_set_VU_DM.csv')
df = clean(df)
exploration(df)

[X,y,qids] = form_train_set(df)

metric = pyltr.metrics.NDCG(k=10)
    
model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(X=X,y=y,qids=qids)     

df_test = pd.read_csv('test_set_VU_DM.csv')
df_test = clean_test(df_test)
srch_id = df_test.srch_id
prop_id = df_test.prop_id
df_test = df_test.drop(['srch_id'],axis = 1)
X = np.array(df_test)

p = model.predict(X)
prediction = pd.DataFrame(p,columns=['prob'])
prediction = prediction.join(srch_id)
prediction = prediction.join(prop_id)
