import pandas as pd
import pickle
from .user import User
from .util import map_commence_time_game_id
import os
import csv
import shutil
from .result_updater import result_updater
import numpy as np



class database():
    def __init__(self):
        self = self

    def get_all_usernames(self):
      df = pd.read_csv('website/users/login_info.csv')
      self.users= df['username'].tolist()
    
    def add_user(self, firstname, lastname, username, password):
       new_user = User(username)

       new_user.create_user(firstname, lastname, username, password)

       self.users = self.get_all_usernames()

    def check_login_credentials(self, username, password):
      df = pd.read_csv('website/users/login_info.csv')
      user_info = df[df['username'] == username]
      if user_info.empty:
        return False
      else:
        if password == user_info['password'].item():
          return True

    def make_data(self, strategy_name):
        df = self.get_data(strategy_name)
        
        # Group the DataFrame by date and get the last cumulative result for each day
        grouped = df.groupby('date')['cumulative_result'].last()
        
        datapoints = []
        total_p_l = df['total_p_l'].iloc[0].item()
        total_precision = round(df['total_precision'].iloc[0].item() * 100, 1)
        best_day = df['best_day_prof'].iloc[0].item()
        worst_day = df['worst_day_prof'].iloc[0].item()
        total_bets_placed = df['total_bets_placed'].iloc[0].item()

        for date, result_sum in grouped.items():
            teams = df[df['date'] == date][['team', 'result']].to_dict(orient='records')
            day_results = df[df['date'] == date][['date', 'daily_result']].tail(1).to_dict(orient='records')

            datapoints.append({'date': date, 'result_sum': result_sum, 'teams': teams, 'day_results': day_results, 'total_p_l':total_p_l, 'total_precision': total_precision, 'best_day': best_day, 'worst_day': worst_day, 'total_bets_placed': total_bets_placed})

        
        
        return datapoints

    def get_all_user_strategies(self):
      df = pd.read_csv('website/users/user_strategy_names.csv')
      return df['strategy_name'].tolist()

    def get_data(self, strategy_name):
       return pd.read_csv(f'website/live_performance_data/{strategy_name}.csv')
    
    def get_user_strategies(self, username):
      this_user = User(username)
      strategies = this_user.get_strategies_associated_with_user()
      return strategies
    
    def delete_user_strategy(self, user, strategy_name):
       # delete the strategy from the user_strategy_names.csv

       # delete the strategy from models

       # delete the live_performance_data from performance folder 
       pass

    def check_if_strategy_exists_and_handle_duplicate(self, input_name, input_strat_params):
      
      # if the params are equal to eachother, handle, else do nothing
      for strat_filename in os.listdir('models/params'):
        if strat_filename == '.DS_Store':
          pass
        else:
          with open(f'models/params/{strat_filename}', "rb") as file:
            strategy_params_ordered_dict = pickle.load(file)
            strategy_params_dict = dict(strategy_params_ordered_dict)
            if strategy_params_dict.items() == input_strat_params.items():
              self.handle_duplicate_strategy(input_name, strat_filename)
              return True
      return False
    
    def handle_duplicate_strategy(self, input_name, pre_existing_strategy_filename):

      pre_existing_strategy_name = pre_existing_strategy_filename.split('.pkl')[0]

      existing_obj_path = f'models/model_objs/{pre_existing_strategy_name}.pth'
      new_obj_path = f'models/model_objs/{input_name}.pth'

      shutil.copy(existing_obj_path, new_obj_path)

      existing_encoder_path = f'models/encoders/{pre_existing_strategy_name}.pkl'
      new_encoder_path = f'models/encoders/{input_name}.pkl'

      shutil.copy(existing_encoder_path, new_encoder_path)

      existing_scaler_path = f'models/scalers/{pre_existing_strategy_name}.pkl'
      new_scaler_path = f'models/scalers/{input_name}.pkl'

      shutil.copy(existing_scaler_path, new_scaler_path)

      existing_params_path = f'models/params/{pre_existing_strategy_name}.pkl'
      new_params_path = f'models/params/{input_name}.pkl'

      shutil.copy(existing_params_path, new_params_path)

      existing_performance_path = f'live_performance_data/{pre_existing_strategy_name}.csv'

      new_performance_path = f'live_performance_data/{input_name}.csv'

      shutil.copy(existing_performance_path, new_performance_path)

    def update_winning_teams_data(self):
      result_update_instance = result_updater()
      result_update_instance.update_results()

    def update_strategy_performance_files(self):

       game_id_to_commence_time = map_commence_time_game_id()
       
       for file in os.listdir('live_performance_data'):
          
          if file.endswith('.csv'):

            df = pd.read_csv(f'live_performance_data/{file}')

            count_bets = int(len(df))

            df['result'] = np.where(df['target'] == 1, df['highest_bettable_odds']*100-100, -100).round().astype(int)

            df['date'] = pd.to_datetime(df['game_id'].replace(game_id_to_commence_time))

            df = df.sort_values(by='date')

            df['date'] = df['date'].dt.date

            df['daily_result'] = df.groupby(df['date'])['result'].transform('sum')

            df['cumulative_result'] = df['result'].cumsum()

            total_pl =  int(df['result'].sum())

            df['total_p_l'] = total_pl

            df['total_ev_per_bet'] = total_pl / count_bets

            count_wins = len(df[df['result'] > 0])
            count_losses = df[df['result'] > 0]

            df['total_precision'] = float(int(count_wins)/int(count_bets))
            
            best_idx = df['daily_result'].idxmax()
            best_row = df.loc[best_idx]
            df['best_day_prof'] = best_row['daily_result']
            worst_idx = df['daily_result'].idxmin()
            worst_row = df.loc[worst_idx]
            df['worst_day_prof'] = worst_row['daily_result']

            df['total_bets_placed'] = count_bets

            df = df.apply(lambda x: x.astype(int) if x.dtypes == 'int64' else x)

            df.to_csv(f'live_performance_data/{file}')
