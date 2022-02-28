#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:05:41 2022

@author: yimi
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

from scipy.special import loggamma
from scipy.stats import norm, chi2

sys.path.append('/Users/yimi/Desktop/Uchi/Winter2022/Linear-NonLinear/WEEK4')
import Regression

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.set_option('precision', 10)

pandas.options.display.float_format = '{:,.7e}'.format

claim_history = pandas.read_csv('/Users/yimi/Desktop/Uchi/Winter2022/Linear-NonLinear/WEEK3/claim_history.csv')



trainData = claim_history[['CLM_COUNT', 'EXPOSURE', 
                            'CAR_TYPE', 'MSTATUS',
                           'HOMEKIDS', 'KIDSDRIV',
                           'REVOKED', 'URBANICITY',
                           'CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME']].dropna()
trainData.reset_index(inplace = True)


#----------------QUESTION1-------------
# Reorder the categories in ascending order of frequencies of the target field
#CAR_TYPE
u = trainData['CAR_TYPE'].astype('category')
u_freq = u.value_counts(ascending = True)
pm = u.cat.reorder_categories(list(u_freq.index))

# Visualize the CLM_COUNT versus CAR_TYPE
df = pandas.DataFrame(pm).join(trainData['CLM_COUNT'])
xtab = df.groupby(['CAR_TYPE', 'CLM_COUNT']).size().reset_index(name = 'N')

ssize = xtab['N']/5
plt.figure(dpi = 200)
scatter = plt.scatter(xtab['CAR_TYPE'], xtab['CLM_COUNT'], s = 100, c = ssize)
plt.xlabel('CAR_TYPE')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])
cbar = plt.colorbar(scatter)
cbar.set_label('Number of Observations')
cbar.set_ticks(range(0,500,50))
plt.show()


#MSTATUS
i = trainData['MSTATUS'].astype('category')
i_freq = i.value_counts(ascending = True)
pm1 = i.cat.reorder_categories(list(i_freq.index))

# Visualize the CLM_COUNT versus MSTATUS
df = pandas.DataFrame(pm1).join(trainData['CLM_COUNT'])
xtab = df.groupby(['MSTATUS', 'CLM_COUNT']).size().reset_index(name = 'N')

ssize = xtab['N']/5
plt.figure(dpi = 200)
scatter = plt.scatter(xtab['MSTATUS'], xtab['CLM_COUNT'], s = 100, c = ssize)
plt.xlabel('MSTATUS')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])
cbar = plt.colorbar(scatter)
cbar.set_label('Number of Observations')
plt.show()


#REVOKED
o = trainData['REVOKED'].astype('category')
o_freq = o.value_counts(ascending = True)
pm3 = o.cat.reorder_categories(list(o_freq.index))

# Visualize the CLM_COUNT versus REVOKED
df = pandas.DataFrame(pm3).join(trainData['CLM_COUNT'])
xtab = df.groupby(['REVOKED', 'CLM_COUNT']).size().reset_index(name = 'N')

ssize = xtab['N']/5
plt.figure(dpi = 200)
scatter = plt.scatter(xtab['REVOKED'], xtab['CLM_COUNT'], s = 100, c = ssize)
plt.xlabel('REVOKED')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])
cbar = plt.colorbar(scatter)
cbar.set_label('Number of Observations')
plt.show()


#URBANICITY
p = trainData['URBANICITY'].astype('category')
p_freq = p.value_counts(ascending = True)
pm4 = p.cat.reorder_categories(list(p_freq.index))

# Visualize the CLM_COUNT versus URBANICITY
df = pandas.DataFrame(pm4).join(trainData['CLM_COUNT'])
xtab = df.groupby(['URBANICITY', 'CLM_COUNT']).size().reset_index(name = 'N')

ssize = xtab['N']/5
plt.figure(dpi = 200)
scatter = plt.scatter(xtab['URBANICITY'], xtab['CLM_COUNT'], s = 100, c = ssize)
plt.xlabel('URBANICITY')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])
cbar = plt.colorbar(scatter)
cbar.set_label('Number of Observations')
plt.show()


#HOMEKIDS
scatter = plt.scatter(trainData['HOMEKIDS'], trainData['CLM_COUNT'], s = 10, alpha=0.2)
plt.xlabel('HOMEKIDS')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])


#KIDSDRIV
scatter = plt.scatter(trainData['KIDSDRIV'], trainData['CLM_COUNT'], s = 10, alpha=0.2)
plt.xlabel('KIDSDRIV')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])


#CAR_AGE
scatter = plt.scatter(trainData['CAR_AGE'], trainData['CLM_COUNT'], s = 10, alpha=0.2)
plt.xlabel('CAR_AGE')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])


#MVR_PTS
scatter = plt.scatter(trainData['MVR_PTS'], trainData['CLM_COUNT'], s = 10, alpha=0.2)
plt.xlabel('MVR_PTS')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])

#TIF
scatter = plt.scatter(trainData['TIF'], trainData['CLM_COUNT'], s = 10, alpha=0.2)
plt.xlabel('TIF')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])

#TRAVTIME
scatter = plt.scatter(trainData['TRAVTIME'], trainData['CLM_COUNT'], s = 10, alpha=0.2)
plt.xlabel('TRAVTIME')
plt.ylabel('CLM_COUNT')
plt.yticks([0,1,2,3,4,5,6,7,8,9])







#----------------QUESTION2-------------

term_car_type = pandas.get_dummies(pm)
term_mstatus = pandas.get_dummies(pm1)
term_mstatus = term_mstatus.rename(columns={"Yes": "mstatusYES", "No": "mstatusNO"})
term_revoke = pandas.get_dummies(pm3)
term_revoke = term_revoke.rename(columns={"Yes": "revokeYES", "No": "revokeNO"})
term_urbanicity = pandas.get_dummies(pm4)

term_homekids = trainData[['HOMEKIDS']]
term_kidsdriv = trainData[['KIDSDRIV']]
term_car_age = trainData[['CAR_AGE']]
term_mvr_pts = trainData[['MVR_PTS']]
term_tif = trainData[['TIF']]
term_travtime = trainData[['TRAVTIME']]

y_train = trainData['CLM_COUNT']
e_train = trainData['EXPOSURE']
o_train = numpy.log(e_train)

# Intercept only model
X_train = trainData[['CLM_COUNT']].copy()
X_train.insert(0, 'Intercept', 1.0)
X_train.drop(columns = ['CLM_COUNT'], inplace = True)

step_summary = pandas.DataFrame()

outList = Regression.PoissonModel(X_train, y_train, o_train)
llk_0 = outList[3]
df_0 = len(outList[4])
step_summary = step_summary.append([['Intercept', df_0, llk_0, numpy.nan, numpy.nan, numpy.nan]], ignore_index = True)

# Find the first predictor
step_detail = pandas.DataFrame()

# 1Try Intercept + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 3Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 4Try Intercept + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 5Try Intercept + REVOKED
X = X_train.join(term_revoke)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 6Try Intercept + URBANICITY
X = X_train.join(term_urbanicity)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ URBANICITY', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 7Try Intercept + CAR_AGE
X = X_train.join(term_car_age)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_AGE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 8Try Intercept + MVR_PTS
X = X_train.join(term_mvr_pts)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MVR_PTS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Current model is Intercept + URBANICITY
row = step_detail[step_detail[0] == '+ URBANICITY']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_urbanicity)






# Find the second predictor
step_detail = pandas.DataFrame()

# 1Try Intercept + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 3Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 4Try Intercept + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 5Try Intercept + REVOKED
X = X_train.join(term_revoke)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 7Try Intercept + CAR_AGE
X = X_train.join(term_car_age)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_AGE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 8Try Intercept + MVR_PTS
X = X_train.join(term_mvr_pts)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MVR_PTS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS
row = step_detail[step_detail[0] == '+ MVR_PTS']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_mvr_pts)







# Find the third predictor
step_detail = pandas.DataFrame()

# 1Try Intercept + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 3Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 4Try Intercept + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 5Try Intercept + REVOKED
X = X_train.join(term_revoke)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 7Try Intercept + CAR_AGE
X = X_train.join(term_car_age)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_AGE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE
row = step_detail[step_detail[0] == '+ CAR_AGE']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_car_age)





# Find the fourth predictor
step_detail = pandas.DataFrame()

# 1Try Intercept + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 3Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 4Try Intercept + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 5Try Intercept + REVOKED
X = X_train.join(term_revoke)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS
row = step_detail[step_detail[0] == '+ MSTATUS']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_mstatus)






# Find the fifth predictor
step_detail = pandas.DataFrame()

# 1Try Intercept + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 3Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 5Try Intercept + REVOKED
X = X_train.join(term_revoke)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE
row = step_detail[step_detail[0] == '+ CAR_TYPE']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_car_type)




# Find the sixth predictor
step_detail = pandas.DataFrame()

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 3Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 5Try Intercept + REVOKED
X = X_train.join(term_revoke)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKE
row = step_detail[step_detail[0] == '+ REVOKED']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_revoke)







# Find the seventh predictor
step_detail = pandas.DataFrame()

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 3Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKE + KIDSDRIV
row = step_detail[step_detail[0] == '+ KIDSDRIV']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_kidsdriv)







# Find the eighth predictor
step_detail = pandas.DataFrame()

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 10Try Intercept + TRAVTIME
X = X_train.join(term_travtime)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKE + KIDSDRIV + TRAVTIME
row = step_detail[step_detail[0] == '+ TRAVTIME']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_travtime)






# Find the ninth predictor
step_detail = pandas.DataFrame()

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# 9Try Intercept + TIF
X = X_train.join(term_tif)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKE + KIDSDRIV + TRAVTIME + TIF
row = step_detail[step_detail[0] == '+ TIF']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_tif)






# Find the tenth predictor
step_detail = pandas.DataFrame()

# 2Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Current model is Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKE + KIDSDRIV + TRAVTIME + TIF + HOMEKIDS
row = step_detail[step_detail[0] == '+ HOMEKIDS']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_homekids)






outList = Regression.PoissonModel(X, y_train, o_train)
final = outList[0]
















#----------------QUESTION3-------------
y_pred = outList[6]

# Plot predicted number of claims versus observed number of claims
plt.figure(dpi = 200)
sg = plt.scatter(y_train, y_pred, c = e_train, marker = 'o')
plt.xlabel('Observed Number of Claims')
plt.ylabel('Predicted Number of Claims')
plt.xticks(range(10))
plt.grid(axis = 'both')
cbar = plt.colorbar(sg, label = 'Exposure')
cbar.set_ticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()





# Calculate deviance residuals
#cite: https://www.datascienceblog.net/post/machine-learning/interpreting_generalized_linear_models/
dR2 = y_train*numpy.log(numpy.where(y_train == 0,1,y_train/y_pred)) - (y_train - y_pred)
devResid = numpy.where(y_train > y_pred, 1.0, -1.0) * numpy.where(dR2 > 0.0, numpy.sqrt(2.0 * dR2), 0.0)


plt.figure(dpi = 200)
sg = plt.scatter(y_train, devResid, c = e_train, marker = 'o')
plt.xlabel('Observed CLM_COUNT')
plt.ylabel('Deviance Residual')
plt.xticks(range(10))
plt.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()












#----------------QUESTION4-------------
n_sample = len(y_train)
# Root Mean Squared Error
y_resid = y_train - y_pred 
print('Sum of Residuals = ', numpy.sum(y_resid))

mse = numpy.sum(numpy.power(y_resid, 2)) / n_sample

rmse = numpy.sqrt(mse)
print('Root Mean Squared Error = ', rmse)

# Relative Error
relerr = mse / numpy.var(y_train, ddof = 0)
print('Relative Error = ', relerr)

#R-squared metrics
sqcor = numpy.power(numpy.corrcoef(y_train, y_pred), 2)
print('Squared Correlation = ', sqcor[0,1])


