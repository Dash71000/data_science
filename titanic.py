# -*- coding: utf-8 -*-


import kagglehub


path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/titanic-dataset/Titanic-Dataset.csv")
df.head()

df.info()

df.shape

df.isnull().sum()

# Fill missing values in 'Cabin' with 'Unknown'
df['Cabin'] = df['Cabin'].fillna('Unknown')

# Fill missing values in 'Age' with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing values in 'Embarked' with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

display(df.isnull().sum())

df.dropna(inplace=True)

df.describe()

df.head()

df.duplicated().sum()

df["family"] = df["SibSp"] + df["Parch"]

sns.countplot(x="family", hue = "Survived" , data = df)
plt.title("survival of family")

df["Survived"].value_counts()

df["Sex"].value_counts()

sns.countplot(x="Sex",data=df)
plt.title("Distribution of passengers w.r.t Gender")

df.groupby("Sex", as_index= False)["Survived"].value_counts()

sns.countplot(x="Sex",hue="Survived",data=df)
plt.title("Survival Rate across Gender")

df.groupby("family" , as_index= False)["Survived"].value_counts()

embarked_survived=df[df["Survived"]==1].groupby(["Embarked"],as_index=False)["Survived"].size()

embarked_survived

sns.barplot(x="Embarked",y="size",data=embarked_survived)
plt.title("Survival Rate across Embarked")

df.groupby(["Age"])["Survived"].mean()

sns.histplot(x = "Age", data = df, kde = True, color=sns.color_palette("muted")[1])
plt.title("Distribution of Age who Travelled")

passengers_survived=df[df["Survived"]==1].groupby(["Pclass","Sex"],as_index=False)["Survived"].value_counts()

sns.barplot(data=passengers_survived,x="Pclass",y="count",hue="Sex")
plt.title("Passengers Survived according to their sex and Pclass")

passengers_dead=df[df["Survived"]==0].groupby(["Pclass","Sex"],as_index=False)["Survived"].value_counts()

sns.barplot(data=passengers_survived,x="Pclass",y="count",hue="Sex")
plt.title("Passengers Dead according to their Pclass and Sex")

passenger_fare=df.groupby(["Pclass"],as_index=False)["Fare"].mean()

passenger_fare

survival_pclass_sexcount=df[df["Survived"]==1].groupby(["Pclass","Sex"],as_index=False)[["Survived"]].count()
survival_pclass_sexcount

sns.barplot(x="Sex",y="Survived",hue="Pclass",data=survival_pclass_sexcount)
plt.title("survival_pclass_sexcount")

death_pclass_sexcount=df[df["Survived"]==0].groupby(["Pclass","Sex"],as_index=False)[["Survived"]].count()
death_pclass_sexcount.columns=["Pclass","Sex","Dead"]
death_pclass_sexcount

sns.barplot(x="Sex",y="Dead",hue="Pclass",data=death_pclass_sexcount)
plt.title("death_pclass_sexcount")

