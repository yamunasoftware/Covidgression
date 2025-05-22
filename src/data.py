### DATA IMPORTS ###

from regression import *
import pandas as pd
import sys
import matplotlib.pyplot as plt

### REGRESSION RUNNING ###

# Checks Arguments:
if len(sys.argv) == 2:
  if '.csv' in sys.argv[1]:
    try:
      # Read Datasets:
      data_orig = pd.read_csv("dataset.csv")
      test_data_orig = pd.read_csv(sys.argv[1])

      # Define Model Features:
      model_features = ['People_tested'] 
      model_output = 'Deaths'

      # Generate Usable Datasets:
      data, max, min = get_dataset(data_orig, model_output)
      test_data, test_max, test_min = get_dataset(test_data_orig, model_output)

      # Get Feature Matrices from Data:
      feature_matrix, output = get_numpy_data(data, model_features, model_output)
      test_feature_matrix, test_output = get_numpy_data(test_data, model_features, model_output)

      # Initialize and Run Regression Model Training:
      initial_weights = [0.0, 0.0]
      step_size = 0.0001
      tolerance = 0.001
      weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
      
      # Get Normalized Prediction Based on Test Data:
      predictions = predict_output(test_feature_matrix, weights)
      normalized_predictions = normalize_series(max, min, predictions, normal=False)

      # Format Test Data for Plotting
      i = 0
      days = []
      for day in test_data_orig.itertuples():
        i += 1
        days.append(i)
      test_plot_data = test_data_orig[['Deaths']].copy()
      test_plot_data['Days'] = days

      # Format Prediction Data for Plotting:
      days.clear()
      for prediction in normalized_predictions:
        i += 1
        days.append(i)
      prediction_data = pd.DataFrame({'Days': days, 'Deaths': normalized_predictions})
      
      # Plot Predictions and Test Data:
      fig = plt.figure()
      for frame, label in zip([test_plot_data, prediction_data], ['Test Data', 'Predictions']):
        plt.plot(frame['Days'], frame['Deaths'], label=label)
      plt.legend()
      plt.savefig('deaths.png', bbox_inches='tight')
      plt.close()
    
    except Exception as e:
      print('Invalid CSV Error')
  
  else:
    print('Invalid CSV File')

else:
  print('Invalid Arguments Length')