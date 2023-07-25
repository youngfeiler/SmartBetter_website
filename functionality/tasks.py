from .strategy_maker import strategy_maker
from .model_runner import model_runner
from .database import database

from celery import Celery
import time
from collections import OrderedDict
import torch

celery = Celery('tasks', broker='redis://localhost:6379/0')


@celery.task
def make_strategy(name, min_ev, min_odds, max_odds, min_min_com, max_min_com, num_epochs):
  # TODO: Check if this strategy already exists

  strat_params_dict = OrderedDict({
            'min_minutes_since_commence':min_min_com,
            'max_minutes_since_commence':max_min_com,
            'min_ev':min_ev,
            'min_avg_odds':min_odds,
            'max_avg_odds':max_odds,
        })

  my_db = database()

  if my_db.check_if_strategy_exists_and_handle_duplicate(name, strat_params_dict):
    return

  strat_maker = strategy_maker(
    name=name,
    min_ev=min_ev,
    min_avg_odds=min_odds,
    max_avg_odds=max_odds,
    min_minutes_since_commence=min_min_com,
    max_minutes_since_commence=max_min_com,
    num_epochs=num_epochs,
  )

  
@celery.task
def start_model_runner():

  mr = model_runner()



  




  
