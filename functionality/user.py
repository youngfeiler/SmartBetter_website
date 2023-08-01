import pandas as pd

class User():
    def __init__(self, username):
        self.username = username
        self.password = ''

    def create_user(self, firstname, lastname, username, password, phone):
      df = pd.read_csv('users/login_info.csv')
      info_row = [firstname, lastname, self.username, password, phone]

      df.loc[len(df)] = info_row

      df.to_csv('users/login_info.csv', index=False)

      self.add_strategy_to_user(self.username, 'SmartBetter low risk demo strategy')


    def add_strategy_to_user(self, username, strategy_name):
      df = pd.read_csv('users/user_strategy_names.csv')
      info_row = [username, strategy_name, False]
      df.loc[len(df)] = info_row
      df.to_csv('users/user_strategy_names.csv', index=False)

    def get_strategies_associated_with_user(self):
      df = pd.read_csv('users/user_strategy_names.csv')
      df = df[df['username'] == self.username]
      strategies = df['strategy_name'].tolist()
      unique_strategies = list(set(strategies))
      return unique_strategies





