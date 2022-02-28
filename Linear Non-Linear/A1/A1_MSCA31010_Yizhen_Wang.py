#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 20:52:24 2022

@author: yimi
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2, norm, anderson


#functions used in this assignment
#cite: code from class
def RegModel (X, y):
    # X: A Pandas DataFrame, rows are observations, columns are regressors
    # y: A Pandas Series, rows are observations of the response variable

    Z = X.join(y)
    n_sample = Z.shape[0]
    n_param = Z.shape[1] - 1

    ZtZ = Z.transpose().dot(Z)
    diag_ZtZ = np.diagonal(ZtZ)
    eps_double = np.finfo(np.float64).eps
    tol = np.sqrt(eps_double)

    ZtZ_transf, aliasParam, nonAliasParam = SWEEPOperator ((n_param+1), ZtZ, diag_ZtZ, sweepCol = range(n_param), tol = tol)

    b = ZtZ_transf[0:n_param, n_param]
    b[aliasParam] = 0.0

    XtX_Ginv = - ZtZ_transf[0:n_param, 0:n_param]
    XtX_Ginv[:, aliasParam] = 0.0
    XtX_Ginv[aliasParam, :] = 0.0

    residual_SS = ZtZ_transf[n_param, n_param]

    return (b, residual_SS, XtX_Ginv, aliasParam, nonAliasParam)

#cite: code from class
def SWEEPOperator (pDim, inputM, origDiag, sweepCol = None, tol = 1e-7):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # origDiag: the original diagonal elements before any SWEEPing
    # sweepCol: a list of columns numbers to SWEEP
    # tol: singularity tolerance, positive real

    if (sweepCol is None):
        sweepCol = range(pDim)

    aliasParam = []
    nonAliasParam = []

    A = np.copy(inputM)

    for k in sweepCol:
        Akk = A[k,k]
        pivot = abs(Akk)
        if (pivot >= abs(tol * origDiag[k])):
            nonAliasParam.append(k)
            ANext = A - np.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / pivot
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:,k] = np.zeros(pDim)
            ANext[k, :] = np.zeros(pDim)
        A = ANext
    return (A, aliasParam, nonAliasParam)






#Set answers into .7E scientific notation
pd.options.display.float_format = '{:.7e}'.format

#load file
file_loc = '/Users/yimi/Desktop/Uchi/Winter2022/Linear-NonLinear/A1/WeightDiary.xlsx'
df = pd.read_excel(file_loc)

#Get name of month and name of DayOfWeek
df['Month'] = df['Date'].dt.strftime("%B")
df['DayOfWeek'] = df['Date'].dt.strftime('%A')


m_order = ['January', 'February', 'March', 'April', 'May', 'June', 
           'July', 'August', 'September', 'October', 'November', 'December']
mfreq = df.groupby(['Month']).Date.count().reindex(m_order, axis=0)
print(mfreq)

mfreq.plot(kind="bar")

w_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
wfreq = df.groupby(['DayOfWeek']).Date.count().reindex(w_order, axis=0)
print(wfreq)

wfreq.plot(kind="bar")


#Question 1
X = pd.get_dummies(df[['Month','DayOfWeek']])
X = X.join(df[['Weight']])
n_sample = X.shape[0]
X.insert(0, 'Intercept', 1.0)

y = X['Weight']

Xt = X.transpose()
XtX = Xt.dot(X)


#Q1-b Weight ~ Intercept
Ab, aliasParam, nonAliasParam = SWEEPOperator(XtX.shape[0],XtX,np.diag(XtX),[0])
Ab
#Q1-c Weight ~ Intercept + Month
Ac, aliasParam, nonAliasParam = SWEEPOperator(XtX.shape[0],XtX,np.diag(XtX),range(13))
Ac
#Q1-d Weight ~ Intercept + DayOfWeek
Ad, aliasParam, nonAliasParam = SWEEPOperator(XtX.shape[0],XtX,np.diag(XtX),[0,13,14,15,16,17,18,19])
Ad
#Q1-e generalized inverse 
Ad1 = pd.DataFrame(Ad)
Ae = Ad1.head(-1)
Ae = Ae.iloc[:,:-1]
Ae = Ae*(-1)
Ae = Ae.replace(-0,0)
Ae

Ae.to_csv('generalized inverse.csv')

#Q1-f model Weight ~ Intercept + Month + DayOfWeek
Af, aliasParamf, nonAliasParamf = SWEEPOperator(XtX.shape[0],XtX,np.diag(XtX),range(20))
Af
#Q1-g smallest Residual Sum of Squares: F
def getSSE (X):
    n_param = X.shape[1] - 1
    residual_SS = X[n_param, n_param]
    return (residual_SS)

print(getSSE(Ab),
getSSE(Ac),
getSSE(Ad),
getSSE(Af))



#Q1-h regression parameters : 19
aliasParamf
nonAliasParamf

#Q-i the regression coefficients 
n_param = Af.shape[1] - 1
b = Af[0:n_param, n_param]
b



#Q2-a boxplot heter bcuz med is not bell shape
# Model: MSRP ~ 1 + Horsepower + Length
X = pd.get_dummies(df[['Month','DayOfWeek']])
X.insert(0, 'Intercept', 1.0)
y = df['Weight']
b, residual_SS, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X, y)
pred_y = np.matmul(X, b)
resid_y = y - pred_y

# Examine the residuals
K1 = df[['Month']]
K1.insert (1, "r", resid_y)

import seaborn as sns
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x='Month', y='r', data=K1)

K2 = df[['DayOfWeek']]
K2.insert (1, "r", resid_y)
ax = sns.boxplot(x='DayOfWeek', y='r', data=K2)



#Q2-b the Anderson-Darling Test statistic / Normality Q-Q Plot for the residuals
anderson_test = anderson(resid_y, dist = 'norm')
print('  Anderson Test = ', anderson_test[0])
print('Critical Values = ', anderson_test[1])
print('       p-values = ', anderson_test[2]/100.0)

# Normal Q-Q Plot
y_new = pd.Series(resid_y * resid_y, name = 'Square_Residual')
n_obs = len(y_new)
obs_quantile = np.sort(resid_y)
z_p = np.array(range(n_obs))
z_p = (1.0 + z_p) / (n_obs + 0.5)
z_quantile = norm.ppf(z_p, loc = np.mean(obs_quantile), scale = np.std(obs_quantile))

fig, ax00 = plt.subplots(1, 1, dpi = 600, figsize = (6,6))
ax00.scatter(obs_quantile, z_quantile)
ax00.set_title('Normal Q-Q Plot')
ax00.set_xlabel('Observed Quantile')
ax00.set_ylabel('Theoretical Quantile')
ax00.axline((0,0), slope = 1, color = 'red', linestyle = '--')
ax00.grid(axis = 'both')
plt.show()


#Q2-c Breusch-Pagan Test / the White Test of Heteroskedasticity

# Breusch-Pagan Test of Homoskedasticity
y_new = pd.Series(resid_y * resid_y, name = 'Square_Residual')
n_obs = len(y_new)
b, SSE0, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X[['Intercept']], y_new)
b, SSE1, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X, y_new)

r_squared = 1.0 - (SSE1 / SSE0)
breusch_test = n_obs * r_squared
breusch_df = len(nonAliasParam) - 1
breusch_pvalue = chi2.sf(breusch_test, breusch_df)



# White Test of Homoskedasticity
def getproduct (X):
    X_new = X
    for i in range(1,13):
        for k in range(13,20):
            u = pd.DataFrame(X.iloc[:,i]*X.iloc[:,k])
            u.columns =[X_new.columns[i] + X_new.columns[k]]
            X_new = X_new.join(u)
    return(X_new)

X_new = getproduct(X)

b, SSE0, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X_new[['Intercept']], y_new)
b, SSE1, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X_new, y_new)

r_squared = 1.0 - (SSE1 / SSE0)

white_test = n_obs * r_squared
white_df = len(nonAliasParam) - 1
white_pvalue = chi2.sf(white_test, white_df)


print('breusch_test = ', breusch_test)
print('breusch_df = ', breusch_df)
print('breusch_pvalue = ', breusch_pvalue)

print('white_test = ', white_test)
print('white_df = ', white_df)
print('white_pvalue = ', white_pvalue)




#Q2-d Durbin-Watson
# Durbin-Watson Test of Autocorrelation
z1 = resid_y[0:(n_obs-1)].to_numpy()
z2 = resid_y[1:n_obs].to_numpy()

durbin_watson_test = np.sum((z1-z2)**2) / np.sum(resid_y**2)
print('Durbin-Watson Test = ', durbin_watson_test)


#Q2-e Shapley values 

# Weight ~ Intercept + Month
b, SSE0, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X[['Intercept']], y)
b, SSE1, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X.iloc[: , :13], y)
r_squared_Intercept_M= 1.0 - (SSE1 / SSE0)


# Weight ~ Intercept + DayOfWeek
X_sub = X[['Intercept']]
X_sub = X_sub.join(X.iloc[: , 13:20])
b, SSE0, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X[['Intercept']], y)
b, SSE1, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X_sub, y)
r_squared_Intercept_W = 1.0 - (SSE1 / SSE0)


# Weight ~ Intercept + Month + DayOfWeek
b, SSE0, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X[['Intercept']], y)
b, SSE1, XtX_Ginv, aliasParam, nonAliasParam = RegModel(X, y)
r_squared_Intercept_M_W = 1.0 - (SSE1 / SSE0)


Shapley_value_M = (r_squared_Intercept_M + r_squared_Intercept_M_W-r_squared_Intercept_W)/2
Shapley_value_W = (r_squared_Intercept_M_W-r_squared_Intercept_M + r_squared_Intercept_W)/2
print('Sharpley value of Month =  %.7e' % (Shapley_value_M))
print('Sharpley value of DayOfWeek = %.7e' % (Shapley_value_W))

Per_Shapley_value_M = Shapley_value_M/r_squared_Intercept_M_W
Per_Shapley_value_W = Shapley_value_W/r_squared_Intercept_M_W
print('Percent sharpley value of Month = %.7e' % (Per_Shapley_value_M))
print('Percent sharpley value of DayOfWeek = %.7e' % (Per_Shapley_value_W))




