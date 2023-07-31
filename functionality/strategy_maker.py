
# takes in a series of arguments 
# makes the proper dataset and augments it properly
# trains the model and tests all the confidence thresholds 


# tests the best test model on validate 


from .data_collector import data_collector
from .model import oddsNet
from .database import database
import pandas as pd
import pickle
import torch
from collections import OrderedDict


# for each model in the folder that contains all the models (no update function because it reads teh dir automatically): 
#    var = strat_maker.dc do the shit to the data
#    df = model.update_pred_results(var)

class strategy_maker():
    def __init__(self, name, bettable_books, architecture= 'silu', learning_rate= .001, weight_decay=False, num_epochs=100, batch_size = 2048, pos_weight=1, from_stacked=True,  min_minutes_since_commence=-1000000, max_minutes_since_commence=240, min_avg_odds=0, max_avg_odds = 10000, min_ev=10, max_ev=10000, max_best_odds=100):
        
        database_instance = database()

        name, 

        
        self.name = name

        self.params= OrderedDict({
            'min_minutes_since_commence':min_minutes_since_commence,
            'max_minutes_since_commence':max_minutes_since_commence,
            'min_ev':min_ev,
            'min_avg_odds':min_avg_odds,
            'max_avg_odds':max_avg_odds,
            'bettable_books': bettable_books
        })

        self.dc = data_collector(
            from_stacked=True,         
            min_minutes_since_commence=min_minutes_since_commence, max_minutes_since_commence=max_minutes_since_commence, 
            min_avg_odds= min_avg_odds, 
            max_avg_odds=max_avg_odds, 
            min_ev=min_ev, 
            max_best_odds=max_best_odds,
            bettable_books = bettable_books
            ) 
    

        self.model = oddsNet(name=name, scaler=self.dc.scaler, encoders=self.dc.encoders, train_data=self.dc.train_data, test_data=self.dc.test_data, val_data=self.dc.val_data, architecture=architecture, learning_rate=learning_rate, weight_decay=weight_decay, num_epochs=num_epochs, batch_size=batch_size, pos_weight=pos_weight)

        backtest_dict = self.model.backtest()

        self.dc.save_val_info(backtest_dict['indices'], self.name)

        self.model.save_model(self.params)

        database_instance.update_strategy_performance_files()
