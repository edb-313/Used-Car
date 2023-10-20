import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from mapie.quantile_regression import MapieQuantileRegressor
from joblib import dump
import os
os.chdir('C:/Users/39329/OneDrive - ISEG/000 Dissertation/Used Car/Used-Car')
# from CP_init import BS_price, binomial_price #
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Dataset/vehicles_clean.csv', sep=',')

# Model has not been listed as categorical col since for a categorical feature with high cardinality (#category is large), it often works best to treat the feature as numeric.
categorical_cols = ['manufacturer','condition', 'cylinders', 'transmission', 'drive', 'type']

"""
for col in categorical_cols:
    df[col] = df[col].astype('category')
"""

X = df.drop('price',axis=1)
y = df['price']

## ============================================================================
##    OPTIMIZE HYPER-PARAMETERS | NORMAL QUANTILE PREDICTION
## ============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

    
df_MAE = []
for i in [5000]:
    for j in [8,16,32,64]:
        for k in [32,64,128,256]:
            for l in [0.01,0.05,0.1]:
                m = LGBMRegressor(objective='quantile', categorical_feature = categorical_indices, alpha=0.5, n_estimators=i,
                                  max_depth=j, num_leaves=k, learning_rate=l, random_state=42)
                m.fit(X_train, y_train)
                yhat = m.predict(X_test)
                df_MAE.append([i,j,k,l,mean_absolute_error(np.array(y_test), yhat)])
                print(i,j,k,l, mean_absolute_error(np.array(y_test), yhat))

df_MAE = pd.DataFrame(df_MAE,columns=['n_trees', 'max_depth', 'num_leaves', 'lr','MAE'])
df_MAE = df_MAE.sort_values(by=['MAE'], ignore_index=True)

m_opt_lower = LGBMRegressor(objective='quantile', alpha=0.05, n_estimators = df_MAE.n_trees[0],
                            max_depth=df_MAE.max_depth[0], num_leaves=df_MAE.num_leaves[0],
                            learning_rate=df_MAE.lr[0], random_state=42)
m_opt_upper = LGBMRegressor(objective='quantile', alpha=0.95, n_estimators = df_MAE.n_trees[0],
                            max_depth=df_MAE.max_depth[0], num_leaves=df_MAE.num_leaves[0],
                            learning_rate=df_MAE.lr[0], random_state=42)
dump(m_opt_lower, 'NQR_lower.joblib') 
dump(m_opt_upper, 'NQR_upper.joblib') 

## ============================================================================
##    OPTIMIZE HYPER-PARAMETERS | CONFORMAL QUANTILE PREDICTION
## ============================================================================

# I'm saving the optimized (in respect to the MAE value of the CQP) LGBM regressor. So these are the values of the LGBM optimized for the CFQ.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
df_MAE = []
for i in [5000]:
    for j in [8,16,32,64]:
        for k in [32,64,128,256]:
            for l in [0.01,0.05,0.1]:
                m = LGBMRegressor(objective='quantile', alpha=0.5, n_estimators=i,
                                  max_depth=j, num_leaves=k, learning_rate=l, random_state=42)
                m_mapie = MapieQuantileRegressor(m, alpha = 0.1)
                m_mapie.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal)
                yhat, y_pis = m_mapie.predict(X_test)             
                df_MAE.append([i,j,k,l,mean_absolute_error(np.array(y_test), yhat)])
                print(i,j,k,l, mean_absolute_error(np.array(y_test), yhat)) 
                             
df_MAE = pd.DataFrame(df_MAE,columns=['n_trees', 'max_depth', 'num_leaves', 'lr','MAE'])
df_MAE = df_MAE.sort_values(by=['MAE'], ignore_index=True)
m_opt = LGBMRegressor(objective='quantile', alpha=0.5, n_estimators = df_MAE.n_trees[0],
                      max_depth=df_MAE.max_depth[0], num_leaves=df_MAE.num_leaves[0],
                      learning_rate=df_MAE.lr[0], random_state=42)
dump(m_opt, 'CQR.joblib') 


