from twilio.rest import Client
from .database import database
import pandas as pd

class texter():
    def __init__ (self, strat_list):
        self.account_sid = 'AC8c14921fcae1fead46c9a56bbd46d606'
        self.auth_token = 'xxx'
        self.twilio_phone_number = '+18573922435'
        self.client = Client(self.account_sid, self.auth_token)
 
    def send_batch_texts(self, strat_to_text_about_list):

        df = pd.read_csv('users/user_strategy_names.csv')

        text_allowed = df[df['text_alerts'] == True ]
        
        phone_nums = pd.read_csv('users/login_info.csv')

        merged_df = phone_nums.merge(text_allowed, on='username', how='inner')

        phone_nums_list = merged_df['phone'].to_list()

        merged_df_info = merged_df[['phone', 'strategy_name']]

        grouped_dict = merged_df_info.groupby('phone')['strategy_name'].agg(list).to_dict()

        for phone_number, strategy_list  in grouped_dict.items():

            for element in strategy_list:
                if element in strat_to_text_about_list:
                    if phone_number in phone_nums_list:
                        strategies_for_user = ''.join([f'\n"{element}"' for element in strategy_list])
                        message_body = f'SmartBetter Alert:\n\nNew info available for \none of your strategies \n\nAct now\n'
                        message = self.client.messages.create(
                                        body=message_body,
                                        from_=self.twilio_phone_number,
                                        to=phone_number
                                    )



