import numpy as np
import pandas as pd
import encoderTitanic as et

""" 過学習を避けるため，階級分けをして年齢のグループを生成 """
def simplify_ages(df):
    #     NaN値を-0.5で穴埋め
    df.Age = df.Age.fillna(-0.5)
    #     (-1,0],(0,5],(5,12],(12,18],(18,25],(25,35],(35,60],(60,120]
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    #     https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    #     年齢データをラベルづけしてbinで分ける
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

""" Cabin numberのNaN値をNで埋める，letterが大事(続く数字は除去) """
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

""" Fareの値を統計情報をもとに階級分けしてグループを生成"""
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    #     mean:32 median:14 max:512
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

""" Last nameとName prefixのみを名前情報とする """
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

""" いらない特徴を除去 """
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

""" 特徴の選択と除去"""
def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

def preprocessingTitanic(df_train, df_test):
    df_train = transform_features(df_train)
    df_test = transform_features(df_test)
    df_train, df_test = et.encode_features(df_train, df_test)
    return df_train, df_test
