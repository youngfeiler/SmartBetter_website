from twilio.rest import Client
from .database import database
import pandas as pd

class texter():
    def __init__ (self, strat_list):
        self.account_sid = 'AC8c14921fcae1fead46c9a56bbd46d606'
        self.auth_token = 'a3bfa3894e91404fe1d53c6f2fb8246a'
        self.twilio_phone_number = '+18573922435'
        self.client = Client(self.account_sid, self.auth_token)
 
    def send_batch_texts(self, text_list):
        # read all the strategy names associated with users
        df = pd.read_csv('users/user_strategy_names.csv')

        # filter for where the texts are allowed
        text_allowed = df[df['text_alerts'] == True ]
        
        # Find the phone numbers associated with those allowed strategies
        phone_nums = pd.read_csv('users/login_info.csv')
        merged_df = phone_nums.merge(text_allowed, on='username', how='inner')
        merged_df_info = merged_df[['phone', 'strategy_name']]

        # Group each phone number to the 
        grouped_dict = merged_df_info.groupby('phone')['strategy_name'].agg(list).to_dict()

        # test this logic
        for phone_number, strategy_list  in grouped_dict.items():
            if phone_number in text_list:
                strategies_for_user = ''.join([f'\n"{element}"' for element in strategy_list])
                print(strategies_for_user)
                message_body = f'Smart Alert:\n\nNew info available for one of your strategies \n\nAct now:\n'
                message = Client.messages.create(
                                body=message_body,
                                from_=self.twilio_phone_number,
                                to=phone_number
                            )



