#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:17:02 2022

@author: jonahkleijn
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from itertools import zip_longest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ndcg_score
import lightgbm
import time
import pickle
from scipy.stats import ttest_ind
import DMT_2_FE as FE
import glob
import seaborn as sb
import warnings
import time
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=Warning)
pd.reset_option('all')

RUN_FAST = False
FRACTION_DATA = 1
SEPARATE_MODELS_PER_COUNTRY = False
NR_COUNTRIES = 1

# =============================================================================
# FE, CLEANING, BALANCING
# =============================================================================
def prepare_df(df):
   df = add_features(df)
   df = fill_nan_values(df)
   df = select_columns_to_do(df)
   df = FE.add_normalized_columns(df)
   df = df.fillna(0)
   return df

def clean_nan(df):
    """TO DO: fix tresholds and leave out target as column"""
    df = drop_nan_columns(df)
    df = drop_nan_rows(df)
    df = fill_nan_values(df)
    df = df.fillna(0)
    df = df.values.astype(np.float)
    return df

def drop_nan_columns(df):
    '''removes columns with more than 80% nans, except for competitor columns'''
    comp_cols = [col for col in df.columns if 'comp' in col]
    remove_cols = df.loc[:, df.isnull().mean() > .8].columns.tolist()
    to_remove = list(set(remove_cols) - set(comp_cols))
    return to_remove, df.drop(to_remove, axis=1)

def drop_nan_rows(df):
    """TO DO: fix tresholds """
    return df.loc[df.isnull().sum(1) < 10]

def fill_nan_values(df):
    '''Fills nan values with 0, 1, mean, or mode'''
    
    # not present after dropping nan columns
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(df['visitor_hist_starrating'].mean())
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(df['visitor_hist_adr_usd'].mean())
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].mean())

    # fill score nans with mean
    df['prop_starrating'] = df['prop_starrating'].fillna(df['prop_starrating'].mean())
    df['prop_location_score1'] = df['prop_location_score1'].fillna(df['prop_location_score1'].mean())
    df['prop_location_score2'] = df['prop_location_score2'].fillna(df['prop_location_score2'].mean())
    df['prop_log_historical_price'] = df['prop_log_historical_price'].fillna(df['prop_log_historical_price'].mean())
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(df['orig_destination_distance'].mean())

    # fill nan with 1
    df['srch_length_of_stay'] = df['srch_length_of_stay'].fillna(1)
    df['srch_adults_count'] = df['srch_adults_count'].fillna(1)
    df['srch_room_count'] = df['srch_room_count'].fillna(1)

    # fill nan with median
    df['srch_booking_window'] = df['srch_booking_window'].fillna(df['srch_booking_window'].mode())
    
    nans_present = df.isna().any()
    nans_present = nans_present[nans_present == True].index
    
    # fill nan with 0
    for column in nans_present:
        df[column] = df[column].fillna(0)
    
    return df

def add_features(df):
    df = FE.timestaps_features(df)
    df = FE.count_competitors(df) 
    df = FE.price_features(df)
    df = FE.count_srch_per_day(df)
    df = FE.count_srch_per_loc(df)
    df = FE.booking_window_features(df)
    df = FE.trip_type(df)
    df = FE.mean_std_meadian_var(df)
    df = FE.length_of_stay_features(df)
    return df

def dummy_variabels(df, col):
    dummies = pd.get_dummies(df[[col]])
    res = pd.concat([df, dummies], axis=1)
    return res

def select_columns_to_do(df):
    if 'click_bool' in df.columns and  'booking_bool' in df.columns:
       cols = ['srch_id','prop_id','click_bool', 'booking_bool'] 
       #df = df.drop(['date_time', 'srch_date', 'vac_date_time', 'vac_date', 'srch_date_str', 'srch_id_date'], axis = 1)
       #df = df.drop(['prop_country_id','srch_destination_id','site_id', 'visitor_location_country_id','position','gross_bookings_usd'], axis = 1)
    else:
        #df = df.drop(['date_time', 'srch_date', 'vac_date_time', 'vac_date', 'srch_date_str', 'srch_id_date'], axis = 1)
        #df = df.drop(['prop_country_id','srch_destination_id','site_id', 'visitor_location_country_id'], axis = 1)
        
       cols = ['srch_id','prop_id']
        
    cols_extra = ['vac_date_season',
                  'count_comp_inv',
                 'vac_date_month',
                 'local_trip',
                 'srch_room_count',
                 'count_comp_rates', 
                 'srch_length_of_stay',
                 'prop_brand_bool',
                 'srch_adults_count',
                 'comp8_rate_percent_diff',
                 'srch_children_count',
                 'srch_query_affinity_score',
                 'visitor_hist_starrating',
                 'srch_booking_window',
                 'orig_destination_distance',
                 'promotion_flag',
                 'visitor_hist_adr_usd',
                 'random_bool',
                 'avg_booking_window_prop',
                 'prop_review_score',
                 'count_srch_per_day',
                 'avg_srch_length_of_stay_prop',
                 'prop_starrating',
                 'std_price_usd_srch',
                 'prop_log_historical_price',
                 'count_srch_per_loc',
                 'median_price_usd_srch',
                 'prop_location_score1',
                 'prop_location_score2',
                 'price_usd']
       
    total_cols = cols + cols_extra
    df = df[total_cols] 
    df = df.select_dtypes([np.number])
    return df

def balance_trainset(frame, col, upsample_minority):
    grouped = frame.groupby(col)
    n_samp = {
        True: grouped.size().max(),
        False: grouped.size().min(),
    }[upsample_minority]

    fun = lambda x: x.sample(n_samp, replace=upsample_minority)
    balanced = grouped.apply(fun)
    balanced = balanced.reset_index(drop=True)
    return balanced

# =============================================================================
# MODELS
# =============================================================================
def split_dataset(df_train): ##AANGPAST
    
    y_train = df_train['click_bool'] + df_train['booking_bool'] * 5
    X_train = df_train.drop(['click_bool', 'booking_bool'], axis=1)
    
    return y_train, X_train

def LGBM_model(train_x, train_y, val_x, val_y, train_groups, val_groups, country_id):
       val_x = val_x.set_index(['srch_id','prop_id'])
       train_x = train_x.set_index(['srch_id','prop_id'])
       model = lightgbm.LGBMRanker(
        boosting_type='gbdt',
        objective="lambdarank",
        n_estimators=5000,
        learning_rate=0.01,
        random_state=200,
        )
       
       
    
       model.fit(
            train_x,
            train_y,
            eval_set = [(val_x, val_y)],
            eval_group = [val_groups],
            group = train_groups,
            eval_metric = "ndcg",
            eval_at = 5,
            verbose = 25,
            early_stopping_rounds = 5000,
         )
       
       pickle.dump(model, open(f'lgbm model ({country_id}).sav', 'wb'))
       
       feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, train_x.columns)), columns=['Value','Feature'])
       feature_imp.to_excel('feature_imp_all.xlsx')
       return model
     
def predict_testset(df_test, model): ##AANGEPAST
    "(maybe ADD predict per country_ID)"
    
    X_test = df_test
    df_test  = df_test.set_index(['srch_id','prop_id'])
    org_cols = df_test.columns.to_list()
    
    results_array = model.predict(df_test[org_cols])
    
    X_test['predicted'] = results_array.tolist()
    X_test['prob'] = results_array.tolist()
    
        
    return X_test

def run_model_all(df_train_all, df_test_all, df_results_total, countries):
       start = time.time()
       print(f'Classification for leftover prop country ids') 
       df_train_sub = df_train_all[df_train_all['prop_country_id'].isin(countries)]       
       
       df_test = df_test_all[df_test_all['prop_country_id'].isin(countries)]
       #df_train 80%, df_val %20% of training_dataset
       df_train = df_train_sub.sample(frac=0.8,random_state=200) 
       df_val = df_train_sub.drop(df_train.index) 
    
    ##Steps TRAIN SET and validation set
       df_train = prepare_df(df_train)  
       y_train, X_train = split_dataset(df_train)          
       df_val = prepare_df(df_val) 
       y_val, X_val = split_dataset(df_val)
    
       train_groups = df_train["srch_id"].value_counts(sort = False).sort_index()
       val_groups = df_val["srch_id"].value_counts(sort = False).sort_index()
       lgbm_model = LGBM_model(X_train, y_train, X_val, y_val, train_groups, val_groups, 'other countries')
    
    
    #Steps TEST SET
       df_test = prepare_df(df_test)     
       start_sol = time.time()
       df_res_test = predict_testset(df_test, lgbm_model) 
       df_results_total = df_results_total.append(df_res_test, ignore_index=True)
       
    
       result_df = get_solution(df_results_total, True)
       
       print(f'finished in {round(time.time() - start)} seconds')
       return df_results_total
    
def run_model_per_country(df_train_all, df_test_all, df_results_total, big_countries):
       start_sol = time.time()
       
       for country in tqdm(big_countries):  
           print('\n')
           print(f'Classification for prop country id: {country}')
           
           df_train_sub = df_train_all[df_train_all['prop_country_id'] == country] 
           df_test = df_test_all[df_test_all['prop_country_id'] == country]
           
           #df_train 80%, df_val %20% of training_dataset
           df_train = df_train_sub.sample(frac=0.8,random_state=200) 
           df_val = df_train_sub.drop(df_train.index) 
           
           ##Steps TRAIN SET and validation set
           df_train = prepare_df(df_train)  
           y_train, X_train = split_dataset(df_train)          
           df_val = prepare_df(df_val) 
           y_val, X_val = split_dataset(df_val)
    
           train_groups = df_train["srch_id"].value_counts(sort = False).sort_index()
           val_groups = df_val["srch_id"].value_counts(sort = False).sort_index()
           lgbm_model = LGBM_model(X_train, y_train, X_val, y_val, train_groups, val_groups, country)
           
           #Steps TEST SET
           df_test = prepare_df(df_test)          
           df_res_test = predict_testset(df_test, lgbm_model) 
           df_results_total = df_results_total.append(df_res_test, ignore_index=True) 
           print(f'\nfinding the solution took {round(time.time() - start_sol,3)} seconds.\n') 
           result_df = get_solution(df_results_total, True)
       
       return df_results_total

# =============================================================================
# SOLUTION
# =============================================================================   
def get_solution(df_test, test):    
    # df_test = df_test.sort_values(by='prob', ascending=False)
    df_test['srch_id'] = df_test['srch_id'].astype(float)
    df_test = df_test.sort_values(['srch_id', 'prob'],ascending = [True, False])    
    df_test['srch_id'] = df_test['srch_id'].astype(int)
    # df_test.to_csv('df_test_maybe_unsorted.csv')
    
    result_df = df_test[['srch_id','prop_id']]
    
    if test:
        result_df.to_csv('solution all countries 30 FE lbgm model.csv', index = False)
    
    return result_df

def validate_solution(result_df, val_df): 
    val_df_sub = val_df[['srch_id','prop_id','booking_bool','click_bool']]
    sol_df = result_df.merge(val_df_sub,how='left', on=['srch_id','prop_id'])
    sol_df['booking_bool'] = sol_df['booking_bool'] * 5
    score =  sol_df['booking_bool'].sum() +  sol_df['click_bool'].sum()
    print(f'total score validation set = {score}')
    
    return sol_df
    
 
def NDCG_score(df, y_true, k=5):
    '''Calculates the NDCG score based on the predictions'''
    y_score = df[['srch_id', 'prop_id', 'predicted_rf']].sort_values(['srch_id', 'prop_id'])
    y_true = y_true.sort_values(['srch_id', 'prop_id'])
    y_true['booking_bool'] = y_true['booking_bool'] * 5
    y_score['predicted_rf'] = y_score['predicted_rf'] * 5
    
    y_score = y_score.groupby('srch_id')['predicted_rf'].apply(list).to_list()
    y_true = y_true.groupby('srch_id')['booking_bool'].apply(list).to_list()
    y_score = np.array(list(zip_longest(*y_score, fillvalue=0))).T
    y_true = np.array(list(zip_longest(*y_true, fillvalue=0))).T
    return ndcg_score(y_true, y_score, k=5)

# =============================================================================
# MAIN
# =============================================================================

def main():  
    # loaded_model = pickle.load(open(filename, 'rb'))
   start_time = time.time()

   df_train_all = pd.read_pickle('training_set_VU_DM.pkl') 
   df_test_all = pd.read_pickle('test_set_VU_DM.pkl')

   if RUN_FAST:
       df_train_all = df_train_all[['srch_id','prop_id', 'price_usd', 'click_bool', 'booking_bool']]
       df_test_all = df_test_all[['srch_id','prop_id','price_usd']]
   
   all_countries = df_train_all.groupby('prop_country_id').size().sort_values(ascending=False).index.tolist()
   big_countries = all_countries[:NR_COUNTRIES]
   #big_countries.reverse() #smaller countries first to spot errors more quickly
   
   df_results_total = pd.DataFrame()
   if not SEPARATE_MODELS_PER_COUNTRY:
       sub_countries = all_countries
   else:
       sub_countries =  all_countries[NR_COUNTRIES:]
       df_results_total = run_model_per_country(df_train_all, df_test_all, df_results_total, big_countries)
   
   if len(sub_countries) > 0:
       df_result_total = run_model_all(df_train_all, df_test_all, df_results_total, sub_countries)
                  
   print(f'-------------------------------------\nfinished after {round(time.time() - start_time)} seconds')

if __name__ == '__main__':
    main()








        