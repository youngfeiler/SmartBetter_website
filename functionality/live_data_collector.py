import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class data_collector():
    def __init__(self, bettable_books, encoders, scaler, min_minutes_since_commence=-1000000, max_minutes_since_commence=2400, min_avg_odds=0, max_avg_odds = 10000, min_ev=10, max_ev=1000, min_best_odds = -1, max_best_odds = 1000, n_last_games_odds = 3, n_last_games_win_perc = 10, n_last_matchups_odds = 3, n_last_matchups_record = 20, n_obs_ago = 3):
        
        self.min_minutes_since_commence = min_minutes_since_commence
        self.max_minutes_since_commence = max_minutes_since_commence
        self.min_avg_odds = min_avg_odds
        self.max_avg_odds = max_avg_odds
        self.min_best_odds = min_best_odds
        self.max_best_odds = max_best_odds
        self.min_ev = min_ev
        self.max_ev = max_ev
        self.n_last_games_odds = n_last_games_odds
        self.n_last_games_win_perc = n_last_games_win_perc
        self.n_last_matchups_odds = n_last_matchups_odds
        self.n_last_matchups_record = n_last_matchups_record
        self.encoders = encoders
        self.scaler = scaler
        self.unique_teams = None
        self.bettable_books = [book + '_1_odds' for book in bettable_books]
    
  
    def format(self, df):
        self.df = df

        self.filter_by_params()

        data_point = self.format_for_nn()
        return data_point

    def make_average_market_odds(self):
        odds_columns = [x for x in self.df.columns if x.endswith('_odds')]
        odds_df = self.df[odds_columns]
        df_array = odds_df.values
        # Create mask for values greater than 0.5
        mask = df_array > 0.5

        # Apply mask and calculate row-wise average
        row_avg = np.nanmean(np.where(mask, df_array, np.nan), axis=1)

        self.df['average_market_odds'] = row_avg

    def filter_by_minutes_since_commence(self):
        self.df = self.df[self.df['minutes_since_commence'] >= self.min_minutes_since_commence]
        self.df = self.df[self.df['minutes_since_commence'] <= self.max_minutes_since_commence]

    def filter_by_average_market_odds(self):
        self.df = self.df[self.df['average_market_odds'] >= self.min_avg_odds]
        self.df = self.df[self.df['average_market_odds'] <= self.max_avg_odds]
    
    def filter_by_ev_thresh(self):

        self.df['highest_bettable_odds'] = self.df[self.bettable_books].max(axis=1)

        self.df['ev'] = ((1/self.df['average_market_odds'])*(100*self.df['highest_bettable_odds']-100)) - ((1-(1/self.df['average_market_odds'])) * 100)

        self.df = self.df[self.df['ev'] >= self.min_ev]
        self.df = self.df[self.df['ev'] <= self.max_ev]

        #self.df.drop('highest_bettable_odds', axis='columns')
        #self.df.drop('highest_bettable_book', axis='columns')
        self.df.drop('ev', axis='columns')

    def filter_by_best_odds(self):
        self.df = self.df[self.df['highest_bettable_odds'] >= self.min_best_odds]
        self.df = self.df[self.df['highest_bettable_odds'] <= self.max_best_odds]

    def filter_by_params(self):
        self.make_average_market_odds()
        self.filter_by_minutes_since_commence()
        self.filter_by_average_market_odds()
        self.filter_by_ev_thresh()
        self.filter_by_best_odds()

    def replace_bad_vals_for_split(self):
        self.df.loc[self.df['park_id'] == 'WIL02', 'park_id'] = 'LOS03'
        self.df.loc[self.df['park_id'] == 'DYE01', 'park_id'] = 'LOS03'
        # self.df.loc[self.df['park_id'] == 'BUF05', 'park_id'] = 'LOS03'
        # self.df.loc[self.df['park_id'] == 'DUN01', 'park_id'] = 'LOS03'

        self.df.loc[self.df['hour_of_start'] == 20, 'hour_of_start'] = 19
        self.df.loc[self.df['hour_of_start'] == 8, 'hour_of_start'] = 9

    def get_cat_cols(self):
    
        categorical_cols = ['home_away', 'team_1', 'hour_of_start', 'day_of_week', 'number_of_game_today', 'day_night', 'park_id', 'this_team_league', 'opponent_league']

        return categorical_cols
    
    def get_cont_cols(self):
        i = 0
        continuous_cols = [col for col in self.df.columns if '1_odds' in col]
        continuous_cols.append('highest_bettable_odds')
        continuous_cols.append('minutes_since_commence')
        continuous_cols.append('this_team_game_of_season')
        continuous_cols.append('opponent_game_of_season')
        continuous_cols.append('average_market_odds')

        return continuous_cols

    def format_for_nn(self):
        final_data_point = pd.DataFrame()

        # Replace some bad values same as we did in the creation of the model
        self.replace_bad_vals_for_split()
        
        #  DON'T TOUCH. ORDER IS VERY IMPORTANT
        column_order = ['barstool_1_odds', 'betclic_1_odds',  
                      'betfair_1_odds', 'betfred_1_odds', 'betmgm_1_odds',               'betonlineag_1_odds', 
                      'betrivers_1_odds', 'betus_1_odds', 'betway_1_odds', 'bovada_1_odds', 'casumo_1_odds', 'circasports_1_odds', 
                      'coral_1_odds', 'draftkings_1_odds', 'fanduel_1_odds', 'foxbet_1_odds', 'gtbets_1_odds', 'ladbrokes_1_odds', 
                      'lowvig_1_odds', 'marathonbet_1_odds', 'matchbook_1_odds', 'mrgreen_1_odds', 'mybookieag_1_odds', 'nordicbet_1_odds', 'onexbet_1_odds', 'paddypower_1_odds', 'pinnacle_1_odds', 'pointsbetus_1_odds', 'sport888_1_odds', 'sugarhouse_1_odds', 'superbook_1_odds', 'twinspires_1_odds', 'unibet_1_odds', 'unibet_eu_1_odds', 'unibet_uk_1_odds', 'unibet_us_1_odds', 'williamhill_1_odds', 'williamhill_us_1_odds', 'wynnbet_1_odds', 'highest_bettable_odds', 'minutes_since_commence', 'this_team_game_of_season', 'opponent_game_of_season', 'average_market_odds', 'home_away', 'team_1', 'hour_of_start', 'day_of_week', 'number_of_game_today', 'day_night', 'park_id', 'this_team_league', 'opponent_league']
        
        self.df = self.df[column_order]
        print('---------------------')
        print(self.df)
        print('---------------------')
        
        continuous_cols = self.get_cont_cols()
        categorical_cols = self.get_cat_cols()

        if len(self.df) > 0:
            continuous_df = self.df[continuous_cols]
            scaled_data = self.scaler.transform(continuous_df)
            
            # Create an empty DataFrame to store the encoded columns
            encoded_df = pd.DataFrame()

            # Iterate over the columns of the original DataFrame
            for col in categorical_cols:
                if col in self.encoders:
                    # Get the corresponding encoder from the dictionary
                    encoder = self.encoders[col]
                    self.df[col] = self.df[col].astype(encoder.categories_[0].dtype)

                    # Encode the column using the corresponding encoder
                    encoded_column = encoder.transform(self.df[col].values.reshape(-1, 1))

                    column_names = [f"{col}_{category}" for category in encoder.categories_[0]]

                    encoded_columns_df = pd.DataFrame(encoded_column, columns=column_names)

                    encoded_df = pd.concat([encoded_df, encoded_columns_df], axis=1)

            scaled_data_df = pd.DataFrame(scaled_data, columns=continuous_cols)
            final_df = pd.concat([scaled_data_df, encoded_df], axis=1)
            return final_df 
        
        return False

      