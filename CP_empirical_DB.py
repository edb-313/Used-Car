import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.metrics import regression_coverage_score
from joblib import  load
import os
os.chdir('C:/Users/39329/OneDrive - ISEG/000 Dissertation/Used Car/Used-Car')
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Dataset/vehicles_clean.csv', sep=',')
X = df.drop('price',axis=1)
y = df['price']

N_TRIALS = 100

## ============================================================================
##    COVERAGES AND WIDTHS: Normal quantile regression
## ============================================================================

cover = np.zeros(N_TRIALS)
width = np.zeros(N_TRIALS)
for j in range(N_TRIALS):    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    m_lower = load('NQR_lower.joblib') 
    m_upper = load('NQR_upper.joblib') 
    m_lower.fit(X_train, y_train)
    m_upper.fit(X_train, y_train)
    y_lower = m_lower.predict(X_test)
    y_upper = m_upper.predict(X_test)
    cover[j] = regression_coverage_score(y_test, y_lower, y_upper)
    width[j] = np.median(np.abs(y_upper - y_lower) / y_test)
print(np.mean(cover), np.mean(width))

## ============================================================================
##    COVERAGES AND WIDTHS: Conformal quantile regression
## ============================================================================

cuts = [5000,20000] # *** CUT-OFF POINTS FOR CONDITION ANALYSIS: CHANGE ACCORDING TO VARIABLE

m_cqr = load('CQR.joblib')       
m_mapie = MapieQuantileRegressor(m_cqr, alpha = 0.1)  
cover, width = np.zeros(N_TRIALS), np.zeros(N_TRIALS)
cover_cond, width_cond = np.zeros([N_TRIALS, 3]), np.zeros([N_TRIALS, 3])
for j in range(N_TRIALS):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.2)
    m_mapie.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal)
    yhat, y_pis = m_mapie.predict(X_test)
    y_lower = y_pis[:, 0, 0]
    y_upper = y_pis[:, 1, 0]
    # Marginal values
    cover[j] = regression_coverage_score(y_test, y_lower, y_upper)
    width[j] = np.median((y_upper - y_lower) / y_test)
    # Conditional values
    X_bin = []
    X_bin.append(X_test.odometer < cuts[0]) # Change variable
    X_bin.append((X_test.odometer >= cuts[0]) & (X_test.odometer < cuts[1]))
    X_bin.append(X_test.odometer >= cuts[1])
    for k in range(len(X_bin)):
        cover_cond[j,k] = regression_coverage_score(y_test[X_bin[k]], y_lower[X_bin[k]], y_upper[X_bin[k]])
        width_cond[j,k] = np.median(np.abs(y_upper[X_bin[k]] - y_lower[X_bin[k]]) / y_test[X_bin[k]])          
print(np.mean(cover), np.mean(width), np.mean(cover_cond, axis=0), np.mean(width_cond, axis=0))


# =============================================================================
# =============================================================================
#    *** PLOTS FOR PAPER ***
# =============================================================================
# =============================================================================
def get_intervals():
    df = pd.read_csv('Dataset/vehicles_clean.csv', sep=',')
    X = df.drop('price',axis=1)
    y = df['price']
    m_cqr = load('CQR.joblib')    
    m_mapie = MapieQuantileRegressor(m_cqr, alpha = 0.1)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.2,
                                                      random_state=7)
    m_mapie.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal)
    yhat, y_pis = m_mapie.predict(X_test)
    y_lower = y_pis[:, 0, 0]
    y_upper = y_pis[:, 1, 0]
    x_var  = np.array(df.odometer.iloc[y_test.index]) # Change Variable
    # remove reversals
    mask = y_upper > y_lower
    y_test = y_test[mask]
    yhat = yhat[mask]
    y_lower = y_lower[mask]
    y_upper = y_upper[mask]
    x_var = x_var[mask]
    return y_lower, y_upper, y_test, yhat, x_var

plt.rc('font', size=14)
N0 = 0
N1 = N0 + 200
fig, ax = plt.subplots(figsize=(10, 5))
y_lower, y_upper, y_test, yhat, x_var = get_intervals()
y_lower = y_lower[N0:N1]
y_upper = y_upper[N0:N1]
yhat = yhat[N0:N1]
y_test = np.array(y_test[N0:N1])
x_var = x_var[N0:N1]
y0 = (y_upper + y_lower) / 2
d = (y_upper - y_lower) / 2
flag = (y_test < (y0 - d)) | (y_test > (y0 + d))
ax.errorbar(x_var[~flag], y0[~flag], yerr=d[~flag], color="black", capsize=4, linewidth=2, ls='none')
ax.errorbar(x_var[flag],  y0[flag],  yerr=d[flag], color="darkgray", capsize=4, linewidth=3, ls='none')
ax.scatter(x_var[flag], y_test[flag], color='black', marker='o', s=15)
ax.set_xlabel('x_var') # *** Change Variable
ax.set_ylabel('Production Year')
print(sum(flag))
plt.savefig('images/image_name.png', bbox_inches='tight', pad_inches = 0.1) 
plt.show()

