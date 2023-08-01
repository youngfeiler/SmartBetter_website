import pandas as pd
import requests

class result_updater():
    
  def __init__ (self):
      self.API_KEY = '02456682ed7b05ec7fd159a594d48339'
      self.SPORT = 'baseball_mlb'
      self.MARKETS = 'h2h'
      self.REGIONS = 'us,eu,uk'
      self.ODDS_FORMAT = 'decimal'
      self.DATE_FORMAT = 'iso'


  def pull_scores(self):
    self.scores_df = ''
    odds_response = requests.get(
      f'https://api.the-odds-api.com/v4/sports/{self.SPORT}/scores/',
      params={
          'api_key': self.API_KEY,
          'daysFrom': int(3)
      }
    )


    if odds_response.status_code != 200:
      print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

    else:
      odds_json = odds_response.json()

      df = pd.DataFrame.from_dict(odds_json)

      self.scores_df = df

      self.scores_json = odds_json

      return odds_json

  def update_results(self):
     try:
      scores_dict = self.pull_scores()
      df = pd.read_csv('mlb_data/scores.csv')
      for each in scores_dict:
            if each['completed'] == False:
                pass
            elif each['completed'] == True:
                df_list = []
                df_list.append(each['id'])
                df_list.append(each['sport_title'])
                df_list.append(each['commence_time'])
                df_list.append(each['home_team'])
                df_list.append(each['away_team'])
                df_list.append(each['scores'][0]['score'])
                df_list.append(each['scores'][1]['score'])
                if int(each['scores'][0]['score']) > int(each['scores'][1]['score']):
                    df_list.append(each['home_team'])
                elif int(each['scores'][0]['score']) < int(each['scores'][1]['score']):
                    df_list.append(each['away_team'])
            df.loc[len(df)] = df_list
      df_unique_game_id = df.drop_duplicates(subset=['game_id'])
      df_unique_game_id.to_csv('mlb_data/scores.csv', index=False)
      return True
     except:
        print("Live results couldn't be updated. Trying agiain in 5 min... ")
        return False




     



