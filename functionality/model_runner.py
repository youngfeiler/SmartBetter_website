import os
import time as ttime
import torch
import pickle
from .database import database
from .live_data_collector import data_collector
from .util import *
from .result_updater import result_updater
from .texter import texter

class model_runner():
    def __init__(self):
      self = self
      self.model_objs_dir = 'models/model_objs/'

      self.amount_of_models = None
      self.list_of_model_names = None
      self.list_of_models = None
      self.list_of_encoders = None
      self.list_of_scalers = None
      self.list_of_params = None

      self.model_storage = {}
      self.database_instance = database()
      self.result_updater_instace = result_updater()
      self.amount_of_models = self.set_amount_of_models()

      self.check_amount_of_models()

      self.run()

    def set_amount_of_models(self):
      return len(os.listdir(self.model_objs_dir))

    def check_amount_of_models(self):
      new_model_count = len(os.listdir(self.model_objs_dir))

      if new_model_count >= self.amount_of_models:
         self.set_amount_of_models()
         self.store_model_info()

    def store_model_info(self):
      for strat_obj_filename in os.listdir(self.model_objs_dir):
        if strat_obj_filename == '.DS_Store':
         pass
        else:
         strat_name = strat_obj_filename.split('.pth')[0]

         if strat_name not in self.model_storage:
          loaded_model = torch.load(f'models/model_objs/{strat_obj_filename}')
          loaded_model.eval()
          with open(f'models/encoders/{strat_name}.pkl', 'rb') as f:
            loaded_encoder = pickle.load(f)
          with open(f'models/scalers/{strat_name}.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
          with open(f'models/params/{strat_name}.pkl', 'rb') as f:
            loaded_ordered_params_dict = pickle.load(f)
            loaded_params_dict = dict(loaded_ordered_params_dict)
            
          loaded_data_collector = data_collector(
            encoders=loaded_encoder,
            scaler = loaded_scaler,
            min_minutes_since_commence=loaded_params_dict['min_minutes_since_commence'],
            max_minutes_since_commence=loaded_params_dict['max_minutes_since_commence'],
            min_avg_odds=loaded_params_dict['min_avg_odds'],
            max_avg_odds=loaded_params_dict['max_avg_odds'],
            min_ev=loaded_params_dict['min_ev'],
            bettable_books=loaded_params_dict['bettable_books']
            )
          
          this_model_dict = {
            'model': loaded_model,
            'encoder': loaded_encoder,
            'scaler': loaded_scaler,
            'params': loaded_params_dict,
            'data_collector': loaded_data_collector,
            'pred_thresh': loaded_params_dict['pred_thresh']
            }
          
          self.model_storage[strat_name] = this_model_dict

    def run(self):
      i = 1
      while i <=999999:
        self.check_amount_of_models()
        market_odds_df = get_odds()

        self.text_list = []

        combined_market_extra_df = preprocess(market_odds_df)
        self.market_odds = combined_market_extra_df
        self.stacked_df = make_stacked_df(combined_market_extra_df)

        for strategy_name, strategy_dict in self.model_storage.items():

            this_model_raw_data_point = strategy_dict['data_collector'].format(self.stacked_df)
            if this_model_raw_data_point is not False:
              
              input_tensor = torch.tensor(this_model_raw_data_point.values, dtype=torch.float32)

              predictions = strategy_dict['model'](input_tensor)

              ind_list = []
              for idx, pred in enumerate(predictions):
                pred_float = pred.detach().numpy()[0]
                print(f'{strategy_name} {pred_float} {strategy_dict["pred_thresh"]}')
                if pred_float >= strategy_dict['pred_thresh']:
                #if pred_float >= -100:
                  ind_list.append(idx)
                  print(f'Bet Found! {strategy_name}')

              if len(ind_list) > 0:
                bet_list = self.get_team_odds_book(this_model_raw_data_point, ind_list, strategy_dict)
                self.handle_bets(bet_list, self.stacked_df, strategy_name, strategy_dict['params']['bettable_books'])
                

        # Once we've ran, we should send a batch of texts
        self.send_texts(self.text_list)
        self.result_updater_instace.update_results() 
        self.database_instance.update_winning_teams_data()
        self.database_instance.update_strategy_performance_files()
        print(f'Ran {i} times')
        i+=1
        ttime.sleep(300)


    def format_sportsbook_names_from_column_names(self, cols):
        formatted_cols = [col.split('_')[0] for col in cols]
        return formatted_cols[:-1]
        
    def de_standardize(self, arr, cols, this_scaler):

        odds = this_scaler.inverse_transform(arr)

        return pd.DataFrame(odds, columns=cols)

    def decode(self, col_name, arr, this_encoder):
        return pd.DataFrame(this_encoder[col_name].inverse_transform(arr))
          
    def get_team_odds_book(self, datapoint_full, indices, strategy):

      datapoint = datapoint_full.iloc[indices]

      this_model_numerical_data = datapoint.iloc[:, :44]
      numerical_column_names = datapoint.columns[0:44].tolist()
      numerical_data_unstandardized = self.de_standardize(this_model_numerical_data, numerical_column_names, strategy['scaler'])

      # need to place highest_bettable_odds somewhere in here
      team_data = datapoint.iloc[:, 46:76]
      team_data_decoded = self.decode('team_1', team_data, strategy['encoder'])
      info_data = pd.concat([numerical_data_unstandardized, team_data_decoded], axis=1)
      info_data = info_data.rename(columns={info_data.columns[-1]: 'team'})
      return info_data

    def handle_bets(self, bet_df, stacked_df, strategy_name, bettable_books):

      self.text_list.append(strategy_name)

      self.append_to_live_performance_sheet(bet_df, stacked_df, strategy_name, bettable_books)

    def append_to_live_performance_sheet(self, bet_df, stacked_df, strategy_name, bettable_books):

      live_results_df = pd.read_csv(f'live_performance_data/{strategy_name}.csv')

      stacked_df = stacked_df.rename(columns={'team_1': 'team'})

      
      for idx, row in bet_df.iterrows():
        team = row['team']

        stacked_df_team = stacked_df[stacked_df['team'] == team]

        for sidx, srow in stacked_df_team.iterrows():
          if abs(srow['minutes_since_commence'] - row['minutes_since_commence']) <= 1:
            common_columns = stacked_df_team.columns.intersection(live_results_df.columns)

            df_to_append = stacked_df[common_columns]

            row_to_append = df_to_append.iloc[sidx].to_frame().T

            row_to_append  = self.fill_extra_cols(row_to_append, bettable_books)
            live_results_df = live_results_df.append(row_to_append, ignore_index=True)
            
            live_results_df.to_csv(f'live_performance_data/{strategy_name}.csv', index=False)

    def fill_extra_cols(self, df, bettable_books):
      subset_columns = [col for col in df.columns if any(item in col for item in bettable_books)]

      df['highest_bettable_odds'] = df[subset_columns].max(axis=1)

      df['ev'] = ((1/df['average_market_odds'])*(100*df['highest_bettable_odds']-100)) - ((1-(1/df['average_market_odds'])) * 100)

      game_id_to_commence_time = self.market_odds.set_index('game_id')['commence_time'].to_dict()

      df['date'] = df['game_id'].map(game_id_to_commence_time)

      df['date'] = pd.to_datetime(df['date']).dt.date

      return df

    def send_texts(self, text_list):

      texter_instance = texter(text_list)
      texter_instance.send_batch_texts(text_list)


      








