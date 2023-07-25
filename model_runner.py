import os
import time
import torch
import pickle
from .database import database
from .util import *

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

      self.amount_of_models = self.set_amount_of_models()

      self.check_amount_of_models()

      self.run()

      # for each model in dict
      # process data with model.dc
      # predict using model.model
      # update live results sheet (new method cause notification process too)

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
          with open(f'models/encoders/{strat_name}.pkl', 'rb') as f:
            loaded_encoder = pickle.load(f)
          with open(f'models/scalers/{strat_name}.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
          with open(f'models/params/{strat_name}.pkl', 'rb') as f:
            loaded_ordered_params_dict = pickle.load(f)
            loaded_params_dict = dict(loaded_ordered_params_dict)
          this_model_dict = {
            'model': loaded_model,
            'encoder': loaded_encoder,
            'scaler': loaded_scaler,
            'params': loaded_params_dict
            }
          
          self.model_storage[strat_name] = this_model_dict


  

    def run(self):
      # Do this whole process every 5 minutes
      database_instance = database()
      database_instance.update_winning_teams_data()

      database_instance.update_strategy_performance_files()

      # Up to this point is preliminarily tested.

      # Every 5 minutes, pull a df that looks like the historical odds df and keep it for indexing purposes later on in this function 

      # Find the game in the extra sheet

      # Combine

      # process 

      # for each model 


      i = 0
      while i < 10:

        # adds new models with which to predict 
        self.check_amount_of_models()

        # get market odds 
        market_odds_df = get_odds()

        combined_market_extra_df = preprocess(market_odds_df)

        # Find the game in the extra sheet


        # transform 

        for model_name, model_obj in self.model_storage.items():

          # feed 
          pass
          # if statement 
        print(f'Ran {i} times')
        time.time.sleep(10)
        i+=1
        

    




