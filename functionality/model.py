import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import warnings
import pickle
import os
import sys
from collections import OrderedDict

#from catboost import CatBoostClassifier
#import shap as shap

warnings.filterwarnings("ignore")

# Want to define the model architecture, the training method, the testing methods, the validate methods. 
# draw out how you want this to be designed ig... Losing focus.... 

class oddsNet():
   def __init__(self, name, train_data, test_data, val_data, scaler, encoders, architecture, learning_rate= .001, weight_decay=False, num_epochs=10, batch_size = 2048, pos_weight=1, sort_criteria='ev', iterations=1000):
      self.name = name
      self.num_trees = iterations
      self.architecture = architecture
      self.learning_rate = learning_rate

      if self.architecture == 'sigmoid':
         self.model = torch.nn.Sequential(   
            torch.nn.Linear(train_data.tensors[0].shape[1],256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256,256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256,128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128,64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64,16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16,1)
         )
         if weight_decay != False:
            self.weight_decay = weight_decay
            self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
         elif not weight_decay:
            self.weight_decay = 'False'
            self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate)
      elif architecture == 'relu': 
         self.model = torch.nn.Sequential(   
            torch.nn.Linear(train_data.tensors[0].shape[1],256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1)
         )
         if weight_decay != False:
            self.weight_decay = weight_decay
            self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
         elif not weight_decay:
            self.weight_decay = 'False'
            self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate)
      elif architecture == 'silu':
         self.model = torch.nn.Sequential(   
            torch.nn.Linear(train_data.tensors[0].shape[1],256),
            torch.nn.SiLU(),
            torch.nn.Linear(256,256),
            torch.nn.SiLU(),
            torch.nn.Linear(256,256),
            torch.nn.SiLU(),
            torch.nn.Linear(256,128),
            torch.nn.SiLU(),
            torch.nn.Linear(128,64),
            torch.nn.SiLU(),
            torch.nn.Linear(64,1)
         )

         if weight_decay != False:
            self.weight_decay = weight_decay
            self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
         elif not weight_decay:
            self.weight_decay = 'False'
            self.optimizer = torch.optim.Adam( self.model.parameters(), lr=self.learning_rate)
      elif architecture == 'catboost':
          self.model = CatBoostClassifier(iterations=self.num_trees, learning_rate=self.learning_rate)
          # Todo: 91 categorical vars
          # store the amount of columns that are categorical in a dc variable 
      
      self.num_epochs = num_epochs
      print(f'num epochs: {self.num_epochs}')
      self.batch_size = batch_size
      self.pos_weight = pos_weight
      self.scaler = scaler
      self.encoders = encoders
      self.sort_criteria = sort_criteria
      self.seed = 42
      random.seed(self.seed)
      np.random.seed(self.seed)
      torch.manual_seed(self.seed)
      self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.batch_size, shuffle = False)
      self.test_loader = torch.utils.data.DataLoader(test_data, batch_size = self.batch_size, shuffle = True)
      self.val_loader = torch.utils.data.DataLoader(val_data, batch_size = 200000, shuffle = True)
      self.val_dataset = val_data


   def scoring_function(self, pred, label):
      return nn.functional.binary_cross_entropy_with_logits(pred, label)
    
   def train(self):
      if self.architecture == 'catboost':
         # Extract the features and labels from the DataLoader
         train_features = []
         train_labels = []

         for batch in self.train_loader:
            batch_features, batch_labels = batch
            train_features.append(batch_features.numpy())
            train_labels.append(batch_labels.numpy())

         test_features = []
         test_labels = []

         for batch in self.test_loader:
            batch_features, batch_labels = batch
            test_features.append(batch_features.numpy())
            test_labels.append(batch_labels.numpy())

         train_features = np.concatenate(train_features, axis=0)
         train_labels = np.concatenate(train_labels, axis=0)

         test_features = np.concatenate(test_features, axis=0)
         test_labels = np.concatenate(test_labels, axis=0)

         # Create a DataFrame from the features and labels
         train_df = pd.DataFrame(train_features)  # Replace column='' names with your actual feature names
         train_df['target'] = train_labels

         test_df = pd.DataFrame(test_features)  # Replace column='' names with your actual feature names
         test_df['target'] = test_labels

         # Select all columns except the last column
         X_train = train_df.iloc[:, :-1]
         y_train = train_df['target']

         self.X_test = test_df.iloc[:, :-1]
         self.y_test = test_df['target']

         self.model.fit(X_train, y_train)

         y_pred = self.model.predict_proba(self.X_test)[:, 1]
         roc_auc = roc_auc_score(self.y_test, y_pred)
         self.auc = roc_auc

         return 


      # Initializes a list that will contain our batch losses for an individual epoch
      epoch_losses = []
      
      # Defines how we want to step through each batch in the epoch
      for batch in self.train_loader:
         
         # Resets the gradient to zero
         self.optimizer.zero_grad()

         # Prepare the input and output tensors for the current batch
         batchX = torch.tensor(batch[0], dtype=torch.float32)
         batchY = torch.tensor(batch[1], dtype=torch.float32)
         
         # Forward pass
         y_pred = self.model.forward(batchX)
         batchY = batchY.unsqueeze(1)  # Reshape to (batch_size, 1)
         
         # Compute the loss with weighted BCEWithLogitsLoss
         pos_weight = torch.tensor([self.pos_weight])  # higher weight for positive class
         
         criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
         #criterion = nn.BCEWithLogitsLoss()

         loss = criterion(y_pred, batchY)
         
         # Store the loss for this batch in the list
         epoch_losses.append(loss.detach().clone())

         # Compute the gradient of the error with respect to the model parameters
         loss.mean().backward()

         # update the model parameters
         self.optimizer.step()
         
      all_epoch_loss = torch.tensor(epoch_losses)
      epoch_loss = torch.mean(all_epoch_loss)
      
      return epoch_loss
   
   def tune_model_test(self):
    # New list for all of the batch predicitons 
    all_predictions = []

    # New list for all of the batch targets 
    all_targets = []

    # New list for each batch
    all_batchx = []

    # Put model in evaluation mode
    self.model.eval()

    # Loop over the test data
    for batch in self.test_loader:
        # Prepare the input and output tensors for the current batch
        batchX = torch.tensor(batch[0], dtype=torch.float32)
        batchY = torch.tensor(batch[1], dtype=torch.float32)

        # Make predictions
        predictions = self.model(batchX)

        all_predictions.append(predictions)

        all_targets.append(batchY)

        all_batchx.append(batchX)

    #auc = calc_tpr_fpr_auc(all_predictions, all_targets, all_batchx)
    stats = self.calc_stats_full_train(all_predictions, all_targets, all_batchx)

    # Return the results
    return stats
   
   def calc_stats_full_train(self, list_predictions, list_targets, list_batchx):
   # We have fully trained our model. now we're finding the best thresholds to use for this model 

    # Define the thresholds we're testing based on the model architecture
      thresholds = []

    # Define some lists whose values we want to see change across thresholds
      tprs = []
      fprs = []
      precisions = []
      amounts_of_bets = []
      evs_per_bet = []

    # Gets data regarding the whole run 
      predictions = torch.cat(list_predictions, dim=0)
      targets = torch.cat(list_targets, dim=0)
      x_vals = torch.cat(list_batchx, dim=0)

      if self.architecture == 'sigmoid':
        for value in range(1, 101):
          threshold = value / 100.0
          thresholds.append(threshold)
      else:
       # Find the minimum and maximum values
        min_value = torch.min(predictions)
        max_value = torch.max(predictions)
        # Calculate the range and step size
        value_range = max_value - min_value
        step_size = value_range / 100.0
        # Step through the range
        for i in range(101):
            threshold = min_value + i * step_size
            thresholds.append(threshold)
    

      for threshold in thresholds:
        # Defines some variables 

        thresh_predictions = torch.where(predictions > threshold, 1, 0)

        thresh_predictions = thresh_predictions.squeeze()
        
        # Splits our sets 
        true_pos = x_vals[(thresh_predictions == 1) & (targets == 1)] # True positives
        false_pos = x_vals[(thresh_predictions == 1) & (targets == 0)] # False positives
        true_neg = x_vals[(thresh_predictions == 0) & (targets == 0)] # True negatives
        false_neg = x_vals[(thresh_predictions == 0) & (targets == 1)] # False negatives

        # Gets info about our sets

        amount_of_correct_pos_preds = true_pos.shape[0]
        amount_of_incorrect_pos_preds = false_pos.shape[0]
        amount_of_bets = amount_of_correct_pos_preds + amount_of_incorrect_pos_preds
        
        # Unscale our true_pos set
        unscaled_true_pos = self.scaler.inverse_transform(true_pos[:, :44]) if len(true_pos) > 0 and hasattr(self.scaler, 'scale_') else np.array([])
        
        if amount_of_bets > 0:
            if amount_of_correct_pos_preds > 0:
               average_best_odds = np.mean(unscaled_true_pos[:, 39])
            else:
               average_best_odds = 0
            gross_rev = average_best_odds*100*amount_of_correct_pos_preds

            net_p_l = gross_rev - (100*amount_of_correct_pos_preds) - (100*amount_of_incorrect_pos_preds)

            ev_per_bet = net_p_l/amount_of_bets

            tpr = true_pos.shape[0] / (true_pos.shape[0] + false_neg.shape[0])

            fpr = false_pos.shape[0] / (false_pos.shape[0] + true_neg.shape[0])

            precision = true_pos.shape[0] / (true_pos.shape[0] + false_pos.shape[0])

        elif amount_of_bets == 0:
            net_p_l = 0
            ev_per_bet = 0
            tpr = 0
            fpr = 0
            precision = 0
            

        tprs.append(tpr)

        fprs.append(fpr)

        precisions.append(precision)

        amounts_of_bets.append(amount_of_bets)

        evs_per_bet.append(ev_per_bet)
            
      auc = self.calculate_auc(tprs, fprs)

      info_list = self.find_best_thresholds(thresholds, evs_per_bet, amounts_of_bets, precisions, auc)
    
      return info_list

   def calculate_auc(self, tpr, fpr):
    sorted_indices = np.argsort(fpr)  # Sort based on FPR
    sorted_tpr = np.array(tpr)[sorted_indices]
    sorted_fpr = np.array(fpr)[sorted_indices]

    # Calculate the AUC using the sorted TPR and FPR arrays
    auc = np.trapz(sorted_tpr, sorted_fpr)

    self.auc = auc

    return self.auc

   def find_best_thresholds(self, my_thresholds, input_evs_per_bet, input_amounts_of_bets, input_precisions, input_auc):
      
         
      my_list = []

      thresholds_new = np.array([t.detach().numpy() for t in my_thresholds])
      evs_per_bet = np.array(input_evs_per_bet)
      amounts_of_bets = np.array(input_amounts_of_bets)
      precisions = np.array(input_precisions)

      # TODO: find this amount 
      min_bets = 232
      max_bets = 1548

      # Select the subset that fits our bet frequency criteria
      filtered_amounts_of_bets = amounts_of_bets[(amounts_of_bets > min_bets) & (amounts_of_bets < max_bets)]
      filtered_evs = evs_per_bet[(amounts_of_bets > min_bets) & (amounts_of_bets < max_bets)]
      filtered_precisions = precisions[(amounts_of_bets > min_bets) & (amounts_of_bets < max_bets)]
      filtered_thresholds = thresholds_new[(amounts_of_bets > min_bets) & (amounts_of_bets < max_bets)]
      # Now sort by precisions
      if self.sort_criteria =='ev':
         sorted_indices = np.argsort(filtered_evs)[::-1]
      elif self.sort_criteria =='precision':
         sorted_indices = np.argsort(filtered_precisions)[::-1]
      
      sorted_filtered_precisions = filtered_precisions[sorted_indices]
      sorted_filtered_evs = filtered_evs[sorted_indices]
      sorted_filtered_thresholds = filtered_thresholds[sorted_indices]
      sorted_filtered_amounts = filtered_amounts_of_bets[sorted_indices]

      # Now select the best 5
      sorted_filtered_precisions_best = sorted_filtered_precisions[:3]
      sorted_filtered_evs_best = sorted_filtered_evs[:3]
      sorted_filtered_thresholds_best = sorted_filtered_thresholds[:3]
      sorted_filtered_amounts_best = sorted_filtered_amounts[:3]

      my_list.append(input_auc)

      for i in range(len(sorted_filtered_precisions_best)):
         my_list.append(sorted_filtered_thresholds_best[i])
         my_list.append(sorted_filtered_amounts_best[i])
         my_list.append(sorted_filtered_evs_best[i])
         my_list.append(sorted_filtered_precisions_best[i])

      return my_list

   def backtest(self):

      if self.architecture == 'catboost':
         self.train()
         return self.auc
         

      # TODO: reinitialize everything 
      best_loss = float('inf')
      patience = 3
      eps_wo_improv = 0

      for epoch in range(int(self.num_epochs)):

         ep_result = self.train()

         if ep_result < best_loss:
            best_loss = ep_result
            eps_wo_improv = 0
         else:
            eps_wo_improv +=1

         if eps_wo_improv >= patience:
            break

         print(f'Training {self.name} epoch #{epoch}')

      if self.architecture == 'catboost':
         # Calculate the ROC AUC score
         y_pred = self.model.predict_proba(self.X_test)[:, 1]
         roc_auc = roc_auc_score(self.y_test, y_pred)


         return roc_auc
      # Gets the testdata
      info_list = self.tune_model_test()

      column_names = ['auc', 'thresh_1', 'tbp_1', 'ev_1', 'prec_1', 'thresh_2', 'tbp_2', 'ev_2', 'prec_2', 'thresh_3', 'tbp_3', 'ev_3', 'prec_3']

      try:
         info_df = pd.DataFrame(info_list).T

         info_df.columns = column_names

         self.best_thresh = float(info_df['thresh_1'])

         hyper_params  = self.make_hp_df()

         new_df = pd.concat([hyper_params, info_df], axis = 1)


         new_df = 0
         
      except:
         print('couldnt make the info sheet... sorry... ')

      return self.apply_best_ev_model_to_validate()

   def apply_best_ev_model_to_validate(self):

      indices_above_threshold = []

      # New list for all of the batch predicitons 
      all_predictions = []

      # New list for all of the batch targets 
      all_targets = []

      # New list for each batch
      all_batchx = []

      # Put model in evaluation mode
      self.model.eval()

      for batch in self.val_loader:
         # Prepare the input and output tensors for the current batch
         batchX = torch.tensor(batch[0], dtype=torch.float32)
         batchY = torch.tensor(batch[1], dtype=torch.float32)

         # Make predictions
         predictions = self.model(batchX)

         # Apply the thresholding and convert predictions to 0 or 1
         batch_thresh_predictions = torch.where(predictions > self.best_thresh, 1, 0)

         # Find indices where predictions are greater than the threshold
         batch_indices_above_thresh = torch.nonzero(batch_thresh_predictions, as_tuple=False)[:, 0]

         # Add the indices from this batch to the list
         indices_above_threshold.extend(batch_indices_above_thresh.numpy())

         all_predictions.append(predictions)

         all_targets.append(batchY)

         all_batchx.append(batchX)

      predictions = torch.cat(all_predictions, dim=0)
      targets = torch.cat(all_targets, dim=0)
      x_vals = torch.cat(all_batchx, dim=0)

      thresh_predictions = torch.where(predictions > self.best_thresh, 1, 0)

      thresh_predictions = thresh_predictions.squeeze()
        
      # Splits our sets 
      true_pos = x_vals[(thresh_predictions == 1) & (targets == 1)] # True positives
      false_pos = x_vals[(thresh_predictions == 1) & (targets == 0)] # False positives
      true_neg = x_vals[(thresh_predictions == 0) & (targets == 0)] # True negatives
      false_neg = x_vals[(thresh_predictions == 0) & (targets == 1)] # False negatives

      # Gets info about our sets
      amount_of_correct_pos_preds = true_pos.shape[0]
      amount_of_incorrect_pos_preds = false_pos.shape[0]
      amount_of_bets = amount_of_correct_pos_preds + amount_of_incorrect_pos_preds
        
      # Unscale our true_pos set
      unscaled_true_pos = self.scaler.inverse_transform(true_pos[:, :44]) if len(true_pos) > 0 and hasattr(self.scaler, 'scale_') else np.array([])
      
      if amount_of_bets > 0:
         if amount_of_correct_pos_preds > 0:
            average_best_odds = np.mean(unscaled_true_pos[:, 39])
            average_market_odds = np.mean(unscaled_true_pos[:, 39])

         else:
            average_best_odds = 0

         gross_rev = average_best_odds*100*amount_of_correct_pos_preds

         net_p_l = gross_rev - (100*amount_of_correct_pos_preds) - (100*amount_of_incorrect_pos_preds)

         ev_per_bet = net_p_l/amount_of_bets

         tpr = true_pos.shape[0] / (true_pos.shape[0] + false_neg.shape[0])

         fpr = false_pos.shape[0] / (false_pos.shape[0] + true_neg.shape[0])

         precision = true_pos.shape[0] / (true_pos.shape[0] + false_pos.shape[0])

      elif amount_of_bets == 0:
         gross_rev = 0
         net_p_l = 0
         ev_per_bet = 0
         tpr = 0
         fpr = 0
         precision = 0
            
      stats = []
      stats.append(ev_per_bet)
      stats.append(tpr)
      stats.append(fpr)
      stats.append(precision)

      return {'stats': stats,
              'indices': indices_above_threshold
              }

   def make_hp_df(self):
      headers = ['architecture', 'learning_rate', 'weight_decay', 'num_epochs', 'batch_size', 'pos_weight']
      
      values=[[self.architecture, self.learning_rate, self.weight_decay, self.num_epochs, self.batch_size, self.pos_weight]]
   
      df = pd.DataFrame(values, columns=headers)

      return df

   def save_model(self, params_dict):

      params_dict['pred_thresh'] = self.best_thresh
       
      # saves the arch and weights
      torch.save(self.model, f'models/model_objs/{self.name}.pth')

      # saves the encoders
      with open(f'models/encoders/{self.name}.pkl', 'wb') as file:
         pickle.dump(self.encoders, file)


      # saves the scaler using pickle
      with open(f'models/scalers/{self.name}.pkl', 'wb') as file:
         pickle.dump(self.scaler, file)


      # Saves the params dictionary to a file using pickle
      with open(f"models/params/{self.name}.pkl", "wb") as file:
         pickle.dump(params_dict, file)   
