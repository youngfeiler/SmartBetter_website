from .strategy_maker import strategy_maker
from .model_runner import model_runner
from .database import database

# from celery import Celery
# from app import celery as my_celery
import time
from collections import OrderedDict
import torch

from celery import Celery

celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    task_serializer='json'
)

@celery.task
def make_strategy(name, min_ev, min_odds, max_odds, min_min_com, max_min_com, num_epochs, bettable_books):
  # TODO: Check if this strategy already exists

  strat_params_dict = OrderedDict({
            'min_minutes_since_commence':min_min_com,
            'max_minutes_since_commence':max_min_com,
            'min_ev':min_ev,
            'min_avg_odds':min_odds,
            'max_avg_odds':max_odds,
            'bettable_books': bettable_books
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
    bettable_books=bettable_books
  )

  
@celery.task
def start_model_runner():

  my_db = database()

  mr = model_runner()



  




  
