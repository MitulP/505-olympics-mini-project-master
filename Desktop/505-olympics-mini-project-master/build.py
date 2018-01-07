# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def load_data():
    df1 = pd.read_csv('/home/mitul/Desktop/505-olympics-mini-project-master/files/olympics.csv', index_col=0, skiprows=1)
    new_columns = ("summer_participations",
         "summer_gold",
         "summer_silver",
         "summer_bronze",
         "summer_total",
         "winter_participations",
         "winter_gold",
         "winter_silver",
         "winter_bronze",
         "winter_total",
         "total_participations",
         "total_gold",
         "total_silver",
         "total_bronze",
         "total_combined")
    assert len(new_columns) == len(df1.columns)
    column_remap = dict(zip(df1.columns, new_columns))
    if df1.columns[0] != "summer_participations":
        for column in df1.columns:
            df1.rename(columns={column:column_remap[column]}, inplace=True)


    names_ids = df1.index.str.split('\s\(')
    df1.index = names_ids.str[0]
    df1 = df1.drop('Totals')
    #df1['ID'] = names_ids.str[1].str[:3]
    df1.drop(["summer_total","winter_total","total_combined",],  axis = 1,inplace=True, errors='raise')
    
    return df1

def first_country(df):
    return df.iloc[0]

def gold_medal(df):
    return df.summer_gold.argmax()


def biggest_difference_in_gold_medal(df):
    return (df.summer_gold - df.winter_gold).abs().argmax()


def get_points(df):
    points = np.zeros(len(df))
    points += df.total_gold * 3
    points += df.total_silver * 2
    points += df.total_bronze
    return pd.Series(points, index=df.index)

def kmn(df):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    centers = kmeans.cluster_centers_ 
    return centers

#df = load_data()
#print(df)
#print(first_country(df)["summer_participations"])
#print(gold_medal(df))
#print(biggest_difference_in_gold_medal(df))
#print(get_points(df))
#print(kmn(df))