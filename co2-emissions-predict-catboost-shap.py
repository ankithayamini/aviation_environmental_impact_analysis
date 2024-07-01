# import libraries
import numpy as np
import pandas as pd

import warnings 
warnings.filterwarnings("ignore")

import shap
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from feature_engine.encoding import RareLabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import re

pd.set_option('display.max_rows', 1000)

df = pd.read_csv("CO2_emissions_by_Aviation.csv").drop_duplicates()
print(df.shape)
df.sample(5).T

df.info()

df.describe().T


# # Data transformation
# extract and log-transform main label
main_label = 'log10_CO2'
df[main_label] = df['value'].apply(lambda x: np.log10(x))
# convert date to month
df['month_year'] = df['date'].apply(lambda x: x[3:]).astype(str)
# df['month_year'] = pd.to_datetime(df['month_year'], format='%m/%Y')
df['month_year'] = pd.to_datetime(df['month_year'], format='%m-%Y')

print('debug-shap!!')
df.info()
# set up the rare label encoder limiting number of categories to max_n_categories
for col in ['country', 'sector']:
    encoder = RareLabelEncoder(n_categories=1, max_n_categories=70, replace_with='Other', tol=10.0/df.shape[0])
    df[col] = encoder.fit_transform(df[[col]])
# finally, drop unused columns
cols2drop = ['value', 'timestamp', 'date']
df = df.drop(cols2drop, axis=1)
print(df.shape)
df.sample(5).T


# # Machine learning

# initialize data

y = df[main_label].values.reshape(-1,)
X = df.drop([main_label], axis=1) # drop extra labels
cat_cols = X.select_dtypes(include=['object']).columns
cat_cols_idx = [list(X.columns).index(c) for c in cat_cols]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.5, random_state=83)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# initialize Pool
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=cat_cols_idx)
test_pool = Pool(X_test,
                 y_test,
                 cat_features=cat_cols_idx)
# specify the training parameters 
model = CatBoostRegressor(iterations=1000,
                          depth=5,
                          learning_rate=0.52,
                          verbose=0,
                          loss_function='RMSE')
# train the model
model.fit(train_pool)
# make the prediction using the resulting model
y_train_pred = model.predict(train_pool)
y_test_pred = model.predict(test_pool)

rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"RMSE score for train {round(rmse_train,4)} dex, and for test {round(rmse_test,4)} dex")

# Baseline scores (assuming the same prediction for all data samples)
rmse_bs_train = mean_squared_error(y_train, [np.mean(y_train)]*len(y_train), squared=False)
rmse_bs_test = mean_squared_error(y_test, [np.mean(y_train)]*len(y_test), squared=False)
print(f"RMSE baseline score for train {round(rmse_bs_train,4)} dex, and for test {round(rmse_bs_test,4)} dex")

# Evaluate the model
print(f"Training Set R-squared: {r2_score(y_train, y_train_pred)}")
print(f"Test Set R-squared: {r2_score(y_test, y_test_pred)}")

print(f"Training Set MAE: {mean_absolute_error(y_train, y_train_pred)}")
print(f"Test Set MAE: {mean_absolute_error(y_test, y_test_pred)}")

print(f"Training Set RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}")
print(f"Test Set RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}")

# Baseline model predictions
mean_co2_emissions_train = np.mean(y_train)
mean_co2_emissions_test = np.mean(y_test)

baseline_predictions_train = [mean_co2_emissions_train] * len(y_train)
baseline_predictions_test = [mean_co2_emissions_test] * len(y_test)

# Baseline model evaluation
r2_baseline_train = r2_score(y_train, baseline_predictions_train)
r2_baseline_test = r2_score(y_test, baseline_predictions_test)
mse_baseline_train = mean_squared_error(y_train, baseline_predictions_train)
mse_baseline_test = mean_squared_error(y_test, baseline_predictions_test)
mae_baseline_train = mean_absolute_error(y_train, baseline_predictions_train)
mae_baseline_test = mean_absolute_error(y_train, baseline_predictions_test)

print("\nBaseline Model:")
print(f"Training Set R-squared: {r2_baseline_train}")
print(f"Test Set R-squared: {r2_baseline_test}")
print(f"Training Set Mean Squared Error: {mse_baseline_train}")
print(f"Test Set Mean Squared Error: {mse_baseline_test}")
print(f"Training Set Mean Absolute Error: {mae_baseline_train}")
print(f"Test Set Mean Absolute Error: {mae_baseline_test}")

# # Explanations with SHAP values
shap.initjs()
ex = shap.TreeExplainer(model)
shap_values = ex.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

expected_values = ex.expected_value
print(f"Average predicted CO2 emissions is {round(10**(expected_values),3)}")
print(f"Average actual CO2 emissions is {round(10**(np.mean(y_test)),3)}")

def show_shap(col, shap_values, label, X_test):
    df_infl = X_test.copy()
    df_infl['shap_'] = shap_values[:, df_infl.columns.tolist().index(col)]
    df_infl['shap_'] = pd.to_numeric(df_infl['shap_'])  # Convert SHAP values to numeric
    gain = round(df_infl.groupby(col)['shap_'].mean(), 4)  # Compute the mean of SHAP values for each category
    gain_std = round(df_infl.groupby(col)['shap_'].std(), 4)
    cnt = df_infl.groupby(col).count()['shap_']
    dd_dict = {'col': list(gain.index), 'gain': list(gain.values), 'gain_std': list(gain_std.values), 'count': cnt}
    df_res = pd.DataFrame.from_dict(dd_dict).sort_values('gain', ascending=False).set_index('col')
    plt.figure(figsize=(12,9))
    plt.errorbar(df_res.index, df_res['gain'], yerr=df_res['gain_std'], fmt="o", color="r")
    plt.title(f'SHAP values for column {col}, label {label}')
    plt.ylabel('dex')
    plt.tick_params(axis="x", rotation=90)
    plt.show();
    print(df_res)
    return

for col in X_test.columns:
    print()
    print(col)
    print()
    show_shap(col, shap_values, label=main_label, X_test=X_test)



