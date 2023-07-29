import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
from tqdm import tqdm
import numpy as np
import itertools
import warnings

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
    # Define a dataframe to hold the results
    results_df = pd.DataFrame(columns=['건물번호', 'Best_RMSE', 'Best_SMAPE'])

    # make folder "logs/prophet_log/{now}" to save the log
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f'logs/prophet_log/{now}'):
        os.makedirs(f'logs/prophet_log/{now}')
        os.makedirs(f'logs/prophet_log/{now}/buildings_plot')
    save_dir = f'logs/prophet_log/{now}'

    building_numbers = train_df['건물번호'].unique()
    for num in tqdm(building_numbers, desc='Building', position=0):
        train_df_num = train_df[train_df['건물번호'] == num]

        # split train_df into train and test. train: 2022-06-01 ~ 2022-08-17, test: 2022-08-18 ~ 2022-08-24
        train = train_df_num.iloc[:-168, :]
        test = train_df_num.iloc[-168:, :]

        # fit the model
        model = Prophet()
        model.fit(train)

        # make forecast
        future = model.make_future_dataframe(periods=168, freq='H')
        forecast = model.predict(future)

        # compute the root mean square error
        rmse = sqrt(mean_squared_error(test['y'], forecast['yhat'][-168:]))
        smape_value = smape(test['y'], forecast['yhat'][-168:])

        # Append results to the dataframe and save
        results_df = pd.concat([results_df, pd.DataFrame([[num, rmse, smape_value]], columns=['건물번호', 'Best_RMSE', 'Best_SMAPE'])], ignore_index=True)
        results_df.to_csv(f'{save_dir}/optimized_parameters.csv', index=False)

        # save train[-168:] as blue, test as green, forecast as orange graph to the folder "logs/prophet_log/{now}/buildings/{num}.png"
        fig = plt.figure(figsize=(20, 10))
        plt.plot(train['ds'][-168:], train['y'][-168:], color='blue', label='train')
        plt.plot(test['ds'], test['y'], color='green', label='test')
        plt.plot(forecast['ds'][-168:], forecast['yhat'][-168:], color='orange', label='forecast')
        plt.legend()
        plt.savefig(f'{save_dir}/buildings_plot/{num}.png')
        plt.close(fig)

if __name__ == "__main__":
    train_df, building_info_df = load_data()
    train_df = preprocess_data(train_df, building_info_df)
    run_prophet(train_df)
