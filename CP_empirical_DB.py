import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from scipy.stats import randint, uniform
from joblib import dump, load
from statsmodels.nonparametric.smoothers_lowess import lowess
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

N_TRIALS = 100
option_type = 'C'
M_cut = [0.98,1.02]
T_cut = [30/252,60/252]

for i in company_list:    
    company = i
    df = pd.read_csv('../data/options_data.csv', sep=';')
    df = df[df['company'] == company] 
    df = clean_data(df, company, option_type = option_type)
    X = df[['money','t','vol','r']].copy()
    y = df['price'] / df['K']
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    m_cqr = load('saved/'+company+'_CQR_'+option_type+'.joblib')       
    m_mapie = MapieQuantileRegressor(m_cqr, alpha = 0.1)  
    cover, width = np.zeros(N_TRIALS), np.zeros(N_TRIALS)
    cover_con_M, width_con_M = np.zeros([N_TRIALS, 3]), np.zeros([N_TRIALS, 3])
    cover_con_T, width_con_T = np.zeros([N_TRIALS, 3]), np.zeros([N_TRIALS, 3])
    for j in range(N_TRIALS):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.2)
        K = np.array(df.K.iloc[y_test.index])
        m_mapie.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal)
        yhat, y_pis = m_mapie.predict(X_test)
        y_lower = y_pis[:, 0, 0]
        y_upper = y_pis[:, 1, 0]
        # Marginal values
        cover[j] = regression_coverage_score(y_test, y_lower, y_upper)
        width[j] = np.median((y_upper - y_lower) / y_test)
        # Conditional values | Moneyness
        X_bin = []
        X_bin.append(X_test.money < M_cut[0])
        X_bin.append((X_test.money >= M_cut[0]) & (X_test.money < M_cut[1]))
        X_bin.append(X_test.money >= M_cut[1])
        for k in range(len(X_bin)):
            cover_con_M[j,k] = regression_coverage_score(y_test[X_bin[k]], y_lower[X_bin[k]], y_upper[X_bin[k]])
            width_con_M[j,k] = np.median(np.abs(y_upper[X_bin[k]] - y_lower[X_bin[k]]) / y_test[X_bin[k]])     
        # Conditional values | Maturity
        X_bin = []
        X_bin.append(X_test.t < T_cut[0])
        X_bin.append((X_test.t >= T_cut[0]) & (X_test.t < T_cut[1]))
        X_bin.append(X_test.t >= T_cut[1])
        for k in range(len(X_bin)):
            cover_con_T[j,k] = regression_coverage_score(y_test[X_bin[k]], y_lower[X_bin[k]], y_upper[X_bin[k]])
            width_con_T[j,k] = np.median(np.abs(y_upper[X_bin[k]] - y_lower[X_bin[k]]) / y_test[X_bin[k]])
    print(company, np.mean(cover), np.mean(width),
          np.mean(cover_con_M, axis=0), np.mean(width_con_M, axis=0),
          np.mean(cover_con_T, axis=0), np.mean(width_con_T, axis=0))



## ============================================================================
##    TESTS AND STATISTICS
## ============================================================================

company_list = ['AMZN','AMD','FB','BA','NFLX','DIS','CRM','PYPL','ADBE','GOOG']
option_type = 'C'
for i in company_list:    
    company = i
    df = pd.read_csv('../data/options_data.csv', sep=';')
    df = df[df['company'] == company] 
    df = clean_data(df, company, option_type = option_type)
    #print(company, len(df), np.mean(df.S), np.mean(df.K), np.mean(df.price), np.mean(df.t), np.mean(df.vol))
    print(company, len(df), np.std(df.S), np.std(df.K), np.std(df.price), np.std(df.t), np.std(df.vol))



# =============================================================================
# =============================================================================
#    *** PLOTS FOR PAPER ***
# =============================================================================
# =============================================================================

def get_intervals(company, option_type):
    cal_size = 0.2
    df = pd.read_csv('../data/options_data.csv', sep=';')
    df = df[df['company'] == company] 
    df = clean_data(df, company, option_type = option_type)
    X = df[['money','t','vol','r']].copy()
    y = df['price'] / df['K']
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)    
    m_cqr = load('saved/'+company+'_CQR_'+option_type+'.joblib')       
    m_mapie = MapieQuantileRegressor(m_cqr, alpha = 0.1)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=cal_size,
                                                      random_state=7)
    m_mapie.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal)
    yhat, y_pis = m_mapie.predict(X_test)
    y_lower = y_pis[:, 0, 0]
    y_upper = y_pis[:, 1, 0]
    K = np.array(df.K.iloc[y_test.index])
    money  = np.array(df.money.iloc[y_test.index])
    t = np.array(df.t.iloc[y_test.index])
    # remove reversals
    mask = y_upper > y_lower
    y_test = y_test[mask]
    yhat = yhat[mask]
    y_lower = y_lower[mask]
    y_upper = y_upper[mask]
    K = K[mask]
    money = money[mask]
    t = t[mask]
    return y_lower, y_upper, y_test, yhat, K, money, t

plt.rc('font', size=14)
N0 = 0
N1 = N0 + 200
company = 'PYPL'
fig, ax = plt.subplots(1,2, figsize=(12, 6))
y_lower, y_upper, y_test, yhat, K, money, t = get_intervals(company, 'C')
K = K[N0:N1]
y_lower = y_lower[N0:N1] * K
y_upper = y_upper[N0:N1] * K
yhat = yhat[N0:N1] * K
y_test = np.array(y_test[N0:N1]) * K
money = money[N0:N1]
y0 = (y_upper + y_lower) / 2
d = (y_upper - y_lower) / 2
flag = (y_test < (y0 - d)) | (y_test > (y0 + d))
ax[0].errorbar(money[~flag], y0[~flag], yerr=d[~flag], color="black", capsize=4, linewidth=2, ls='none')
ax[0].errorbar(money[flag],  y0[flag],  yerr=d[flag], color="darkgray", capsize=4, linewidth=3, ls='none')
ax[0].scatter(money[flag], y_test[flag], color='black', marker='o', s=15)
ax[0].set_xlabel('Moneyness')
ax[0].set_ylabel('Call price')
ax[0].set_title('Calls', fontsize="medium", x=0.5, y=0.9)
print(sum(flag))
# ===
y_lower, y_upper, y_test, yhat, K, money, t = get_intervals(company, 'P')
K = K[N0:N1]
y_lower = y_lower[N0:N1] * K
y_upper = y_upper[N0:N1] * K
yhat = yhat[N0:N1] * K
y_test = np.array(y_test[N0:N1]) * K
money = money[N0:N1]
y0 = (y_upper + y_lower) / 2
d = (y_upper - y_lower) / 2
flag = (y_test < (y0 - d)) | (y_test > (y0 + d))
ax[1].errorbar(money[~flag], y0[~flag], yerr=d[~flag], color="black", capsize=4, linewidth=2, ls='none')
ax[1].errorbar(money[flag],  y0[flag],  yerr=d[flag], color="darkgray", capsize=4, linewidth=3, ls='none')
ax[1].scatter(money[flag], y_test[flag], color='black', marker='o', s=15)
ax[1].set_xlabel('Moneyness')
ax[1].set_ylabel('Put price')
ax[1].set_title('Puts', fontsize="medium", x=0.5, y=0.9)
print(sum(flag))
plt.savefig('images/'+company+'_price_money.png', bbox_inches='tight', pad_inches = 0.1)
plt.show()

