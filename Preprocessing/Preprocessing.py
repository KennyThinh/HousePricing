#!/usr/bin/python3
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


path=os.path.join(os.path.dirname(__file__),"../data/train.csv")
df_train=pd.read_csv(path,delimiter=",",header=0,index_col="Id")
#print(df_train["SalePrice"].describe())
print(df_train.shape)

"""
See distribution of SalePrice ---------------------------------------------------------------------------
"""
ax=sns.distplot(df_train["SalePrice"])
plt.show()

#Skewness: measures the asymmetry in data: data is inclined to which sides (left or right)
#Kurtosis measures the peak of a distribution curve: the curve is more or less higher than the normal curve.
print(df_train["SalePrice"].skew()) #1.8828757597682129
print(df_train["SalePrice"].kurt()) #6.536281860064529
print(df_train["YearBuilt"].describe())

"""
 Relationship with other variables ----------------------------------------------------------------
"""
plt.subplot(2,2,1)
#GrLivArea
plt.scatter(df_train["GrLivArea"],df_train["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
# --> Linear
# TotalBsmtSF
plt.subplot(2,2,2)
plt.scatter(df_train["TotalBsmtSF"],df_train["SalePrice"])
plt.xlabel("TotalBsmtSF")
plt.ylabel("SalePrice")
# -->Strong linear (exponential maybe)
plt.subplot(2,2,3)
sns.boxplot(df_train["OverallQual"],df_train["SalePrice"])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
# --> Linear
plt.subplot(2,2,4)
sns.boxplot(df_train["YearBuilt"],df_train["SalePrice"])
# ax.set_xtick([0, 1870, 1910, 1950, 1990, 2010])
plt.xlabel("YearBuilt")
plt.ylabel("SalePrice")
plt.show()
# --> From 1995, the price is going up a little

""" 
Relationship between attributes ----------------------------------------------------------------------------
"""
plt.subplots(figsize=(12,9))
corrmat = df_train.corr() # corr is a DF
sns.heatmap(corrmat,square=True,vmax=.8)
plt.title("Heatmap")
plt.show()

# Choose 10 largest correlation
plt.subplots(figsize=(12,9))
# choose 10 largests from columns "SalePrice" of DataFrame corr
cols=corrmat.nlargest(10,"SalePrice")["SalePrice"].index #cols=["SalePrice","OveralQual","GrLivArea","GarageCars","GarageArea",.....,"YearBuilt"]
df_values = df_train[cols].values
df_transform=df_values.T # df_transform = (10,1460). We need transform cause np.corrcoef requires row: a variable, a column: a single observation of all those variables
cm = np.corrcoef(df_transform)
sns.set(font_scale=1.25)
hm=sns.heatmap(cm,square=True,vmax=0.8,annot=True,yticklabels=cols.values,xticklabels=cols.values)
plt.title("Heatmap of 10 largest")
plt.show()
# -->GarageCars ~ GarageArea --> do not need GarageArea
# -->TotalBsmtSF ~ 1stFloor --> do not need 1stFloor
# -->GrLivArea ~ TotRmsAbvGrd --> do not need TotRmsAbvGrd


# Scatter plot bwt SalePrice and each of 10 attr above
sns.set()
cols=["SalePrice","OverallQual","GrLivArea","GarageCars","TotalBsmtSF","FullBath","YearBuilt"]
sns.pairplot(df_train[cols],size=1.5)
plt.show()