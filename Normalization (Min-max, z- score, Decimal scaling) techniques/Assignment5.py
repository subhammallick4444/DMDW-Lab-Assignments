import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
df = pd.read_csv("DownloadS/Toyota.csv")
df

df.head(100)

df.info

df.MetColor.head(10)

df.HP.unique()

copy_df = df.copy()
copy_df.fillna(0, inplace=True)

#replace all string data objects to iteger bject
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Iterate through columns and apply label encoding to each string column
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column contains strings
        df[column] = label_encoder.fit_transform(df[column])

  # apply normalization techniques
def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

# view normalized data   
df["Price"]=normalize_column(df["Price"]);
df["Age"]=normalize_column(df["Age"]);
df["KM"]=normalize_column(df["KM"]);
display(df)

df2= df.copy()
df2.fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in df2.columns:
    if df2[column].dtype == 'object':
        df2[column] = label_encoder.fit_transform(df2[column])

  def decimal_scaling(column):
    max_val = column.max()
    j = len(str(max_val))
    normalized_column = column / (10 ** j)
    return normalized_column

df2["Price"]=decimal_scaling(df2["Price"]);
df2["Age"]=decimal_scaling(df2["Age"]);
df2["KM"]=decimal_scaling(df2["KM"]);
display(df2)

df3 = df.copy()
df3.fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in df3.columns:
    if df3[column].dtype == 'object':
        df3[column] = label_encoder.fit_transform(df3[column])

  import statistics as st
def z_score(column):
    mean = st.mean(column)
    std_dev = st.stdev(column)
    normalized_column = abs(column - mean) / std_dev
    return normalized_column

df3["Price"]=z_score(df3["Price"]);
df3["Age"]=z_score(df3["Age"]);
df3["KM"]=z_score(df3["KM"]);
display(df3)
