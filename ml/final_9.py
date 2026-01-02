#### 第九题
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
import os
os.chdir('/Users/miaaaa/Desktop/python')
train_df = pd.read_excel('ml/data/回归预测.xlsx', sheet_name='训练集', header=None)
test_df  = pd.read_excel('ml/data/回归预测.xlsx', sheet_name='测试集', header=None)
X_train = train_df.iloc[:, :31]
y_train = train_df.iloc[:, 31]
X_test  = test_df.iloc[:, :31]
y_test  = test_df.iloc[:, 31]
cat_cols = X_train.columns[X_train.dtypes == 'object']
num_cols = X_train.columns.difference(cat_cols)
pre = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols),
     ('num', StandardScaler(), num_cols)]
)
X_tr = pre.fit_transform(X_train)
X_te = pre.transform(X_test)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
# 2.1 Ridge
ridge_param = {'alpha': np.logspace(0, 3, 20)}
ridge_grid = GridSearchCV(Ridge(), ridge_param, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_nested = cross_val_score(ridge_grid, X_tr, y_train, cv=outer_cv, scoring='neg_mean_squared_error')
ridge_grid.fit(X_tr, y_train)          # 全量重拟合
pred_ridge_test = ridge_grid.predict(X_te)
results['ridge'] = {
    'nested_rmse': (-ridge_nested.mean())**0.5,
    'best_params': ridge_grid.best_params_,
    'test_mse': mean_squared_error(y_test, ridge_grid.predict(X_te)),
    'test_re_mean': (np.abs((y_test - pred_ridge_test) / y_test)).mean() * 100,  # 相对误差均值
    'test_re_std': (np.abs((y_test - pred_ridge_test) / y_test)).std() * 100,   # 相对误差标准差
}
# 2.2 ElasticNet
enet_param = {'alpha': np.logspace(0, 2, 10), 'l1_ratio': [.1, .5, .7, .9, 1]}
enet_grid = GridSearchCV(ElasticNet(max_iter=2000), enet_param, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
enet_nested = cross_val_score(enet_grid, X_tr, y_train, cv=outer_cv, scoring='neg_mean_squared_error')
enet_grid.fit(X_tr, y_train)
pred_enet_test = enet_grid.predict(X_te)
results['enet'] = {
    'nested_rmse': (-enet_nested.mean())**0.5,
    'best_params': enet_grid.best_params_,
    'test_mse': mean_squared_error(y_test, enet_grid.predict(X_te)),
    'test_re_mean': (np.abs((y_test - pred_enet_test) / y_test)).mean() * 100,
    'test_re_std': (np.abs((y_test - pred_enet_test) / y_test)).std() * 100,
}
# 2.3 RandomForest
rf_param = {
    'n_estimators': [300, 500],
    'max_depth': [None, 10],
    'min_samples_leaf': [2, 5, 10]
}
rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),  # 单线程
        rf_param, cv=3, scoring='neg_mean_squared_error', n_jobs=1
)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
rf_nested = cross_val_score(rf_grid, X_tr, y_train,
                            cv=outer_cv, scoring='neg_mean_squared_error',
                            n_jobs=1)
rf_grid.fit(X_tr, y_train)
pred_rf_test = rf_grid.predict(X_te)
results['rf'] = {
    'nested_rmse': (-rf_nested.mean())**0.5,
    'best_params': rf_grid.best_params_,
    'test_mse': mean_squared_error(y_test, rf_grid.predict(X_te)),
    'test_re_mean': (np.abs((y_test - pred_rf_test) / y_test)).mean() * 100,
    'test_re_std': (np.abs((y_test - pred_rf_test) / y_test)).std() * 100,
}
# results
for m, v in results.items():
    print(f'{m:5s} | nested RMSE: {v["nested_rmse"]:.3f} | best params: {v["best_params"]} | Test MSE: {v["test_mse"]:.3f}')
    print(f'      | Test MSE: {v["test_mse"]:.3f} | RE均值: {v["test_re_mean"]:.2f}% | RE标准差: {v["test_re_std"]:.2f}%')
    print('-' * 80)
pred_ridge = ridge_grid.predict(X_te)
pred_enet  = enet_grid.predict(X_te)
pred_rf    = rf_grid.predict(X_te)
simple_mean = (pred_ridge + pred_enet + pred_rf) / 3
print('Simple-mean (after tuning) Test MSE:', mean_squared_error(y_test, simple_mean))
se = (y_test - simple_mean) ** 2               
re = np.abs((y_test - simple_mean) / y_test)    
print('=== 平方误差 SE ===')
print('均值:', se.mean())
print('方差:', se.var())
print('标准差:', se.std())
print('=== 相对误差 |RE| ===')
print('均值 (%):', re.mean() * 100)
print('方差 (%%²):', re.var() * 10000)
print('标准差 (%):', re.std() * 100)