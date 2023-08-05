import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from math import sqrt
import datetime
from tqdm import tqdm
import numpy as np
import itertools
import warnings
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore")

def load_data():
    # load dataset
    train_df = pd.read_csv('./data/train.csv')
    building_info_df = pd.read_csv('./data/building_info.csv')

    return train_df, building_info_df

def preprocess_data(train_df, building_info_df):
    # preprocess dataset
    train_df = train_df.drop(['num_date_time'], axis=1)
    train_df['ds'] = pd.to_datetime(train_df['일시'])
    train_df = train_df.fillna(0)
    building_info_df = building_info_df.replace('-', 0)
    train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
    train_df.rename(columns={'전력소비량(kWh)': 'y'}, inplace=True)

    return train_df

def smape(y_true, y_pred):
    # convert to numpy array
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def run_prophet(train_df):
    params_grid = {  
    'changepoint_prior_scale': [0.001, 0.05, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
    }
    grid = ParameterGrid(params_grid)

    # Define a dataframe to hold the results
    results_df = pd.DataFrame(columns=['건물번호', 'Best_param', 'Best_RMSE', 'Best_SMAPE'])

    # make folder "logs/prophet_log/{now}" to save the log
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f'logs/prophet_log/{now}'):
        os.makedirs(f'logs/prophet_log/{now}')
        os.makedirs(f'logs/prophet_log/{now}/buildings_param')
        os.makedirs(f'logs/prophet_log/{now}/buildings_plot')
    save_dir = f'logs/prophet_log/{now}'

    building_numbers = train_df['건물번호'].unique()
    for num in tqdm(building_numbers, desc='Building', position=0):
        param_rsme_smape_df = pd.DataFrame(columns=['Parameter', 'RMSE', 'SMAPE'])

        train_df_num = train_df[train_df['건물번호'] == num]
        train_df_num = train_df_num.drop(['건물번호'], axis=1)

        train = train_df_num.iloc[:-168, :]
        test = train_df_num.iloc[-168:, :]

        best_smape = float("inf")
        best_param = None
        best_exog = None

        for param in tqdm(grid, desc='Parameter', position=1, leave=False):
            # fit the model
            model =Prophet(**param)
            model.add_country_holidays(country_name='KR')
            model.fit(train)

            # make forecast
            future = model.make_future_dataframe(periods=168, freq='H')
            forecast = model.predict(future)

            # compute the root mean square error
            rmse = sqrt(mean_squared_error(test['y'], forecast['yhat'][-168:]))
            smape_val = smape(test['y'], forecast['yhat'][-168:])

            param_rsme_smape_df = pd.concat([param_rsme_smape_df, pd.DataFrame([[param, rmse, smape_val]], columns=['Parameter', 'RMSE', 'SMAPE'])], ignore_index=True)


            if smape_val < best_smape:
                best_smape = smape_val
                best_param = param

                # save train[-168:] as blue, test as green, forecast as orange graph to the folder "logs/prophet_log/{now}/buildings/{num}.png"
                fig = plt.figure(figsize=(20, 10))
                plt.plot(train['ds'][-168:], train['y'][-168:], color='blue', label='train')
                plt.plot(test['ds'], test['y'], color='green', label='test')
                plt.plot(forecast['ds'][-168:], forecast['yhat'][-168:], color='orange', label='forecast')
                plt.legend()
                plt.savefig(f'{save_dir}/buildings_plot/{num}.png')
                plt.close(fig)
        
        param_rsme_smape_df = param_rsme_smape_df.sort_values(by='SMAPE', ascending=True)
        param_rsme_smape_df.to_csv(f'{save_dir}/buildings_param/{num}.csv', index=False)

        # Append results to the dataframe and save
        results_df = pd.concat([results_df, pd.DataFrame([[num, best_param, rmse, smape_val]], columns=['건물번호', 'Best_param', 'Best_RMSE', 'Best_SMAPE'])], ignore_index=True)
        results_df.to_csv(f'{save_dir}/optimized_parameters.csv', index=False)



if __name__ == "__main__":
    train_df, building_info_df = load_data()
    train_df = preprocess_data(train_df, building_info_df)
    run_prophet(train_df)
