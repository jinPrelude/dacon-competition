import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
import argparse
import warnings
import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_data():
    # load dataset
    train_df = pd.read_csv('./data/train.csv')
    building_info_df = pd.read_csv('./data/building_info.csv')

    return train_df, building_info_df

def preprocess_data(train_df, building_info_df):
    # preprocess dataset
    train_df = train_df.drop(['num_date_time'], axis=1)
    train_df['일시'] = pd.to_datetime(train_df['일시'])
    train_df = train_df.fillna(0)
    train_df['습도(%)'] = train_df['습도(%)'].astype(int)
    building_info_df = building_info_df.replace('-', 0)

    # concat train_df and building_info_df by building_number
    train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
    ec_column = train_df.pop('전력소비량(kWh)')
    train_df['전력소비량(kWh)'] = ec_column
    train_df = train_df.set_index('일시')

    return train_df

def run_arima_grid_search(train_df, pdq, exog_columns, use_exog):
    # Define a dataframe to hold the results
    results_df = pd.DataFrame(columns=['건물번호', 'Best_Order', 'Best_RMSE'])

    # make folder "logs/arimax_param_rsme_log/{now}" to save the log
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f'logs/arimax_param_rsme_log/{now}'):
        os.makedirs(f'logs/arimax_param_rsme_log/{now}')
        os.makedirs(f'logs/arimax_param_rsme_log/{now}/buildings')
    save_dir = f'logs/arimax_param_rsme_log/{now}'

    building_numbers = train_df['건물번호'].unique()
    for num in tqdm(building_numbers, desc='Building', position=0):
        # make dataframe to track all the parameters and the corresponding RMSE
        param_rsme_df = pd.DataFrame(columns=['Parameter', 'Exog', 'RMSE'])

        train_df_num = train_df[train_df['건물번호'] == num]
        train_df_num = train_df_num.drop(['건물번호'], axis=1)

        # split train_df into train and test. train: 2022-06-01 ~ 2022-08-17, test: 2022-08-18 ~ 2022-08-24
        train = train_df_num.iloc[:-168, :]
        test = train_df_num.iloc[-168:, :]

        best_rmse = float("inf")
        best_param = None

        for param in tqdm(pdq, desc='ARIMAX', position=1, leave=False):
            try:
                model = ARIMA(train['전력소비량(kWh)'], exog=(train[exog_columns] if use_exog else None),
                                order=(param[0], param[1], param[2])
                                )
                results = model.fit()

                # get the forecast
                forecast = results.get_forecast(steps=168, exog=(test[exog_columns] if use_exog else None))

                # compute the root mean square error
                rmse = sqrt(mean_squared_error(test['전력소비량(kWh)'], forecast.predicted_mean))

                param_rsme_df = param_rsme_df.append({'Parameter': param, 'Exog': exog_columns if use_exog else None, 'RMSE': rmse}, ignore_index=True)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_param = param

            except Exception as e:
                print(e)
                break

        # Append results to the dataframe and save
        param_rsme_df = param_rsme_df.sort_values(by=['RMSE'])
        param_rsme_df.to_csv(f'{save_dir}/buildings/{num}.csv', index=False)

        results_df = results_df.append({'건물번호': num, 'Best_Order': best_param, 'Best_RMSE': best_rmse}, ignore_index=True)
        results_df.to_csv(f'{save_dir}/optimized_parameters.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARIMA grid search for building energy consumption prediction.')
    parser.add_argument('--use_exog', action='store_true', help='Whether to use exogenous variables in the ARIMA model.')

    args = parser.parse_args()
    use_exog = args.use_exog

    # define the p, d and q parameters to take any value between 0 and 2
    p = q = range(0, 3) # 0, 1, 2, 3, 4
    d = range(0, 3) # 0, 1, 2

    # generate all different combinations of p, d and q triplets
    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]

    exog_columns = ['기온(C)', '습도(%)']

    train_df, building_info_df = load_data()
    train_df = preprocess_data(train_df, building_info_df)
    run_arima_grid_search(train_df, pdq, exog_columns, use_exog)