### REGRESSION IMPORTS ###

import pandas as pd
import numpy as np

### REGRESSION FUNCTIONS ###

# Regression Training Function:
def regression_gradient_descent(feature_matrix, labels, initial_weights, step_size, tolerance):
  gradient_magnitude = tolerance 
  weights = np.array(initial_weights)

  while gradient_magnitude >= tolerance:
    gradient = weight_derivative(weights, feature_matrix, labels)
    weights -= step_size * gradient
    gradient_magnitude = np.linalg.norm(gradient)
  return (weights)

# Weight Derivative Function:
def weight_derivative(weights, feature_matrix, labels):
  predictions = np.dot(feature_matrix, weights)
  residuals = (predictions - labels)

  derivative = np.dot(feature_matrix.T, residuals)
  return derivative/(len(labels))

# Output Prediction Function:
def predict_output(feature_matrix, weights):
  return np.dot(feature_matrix, weights)

# Normalization of Series Function:
def normalize_series(max, min, series, normal=True):
  normalized = []
  for data in series:
    if normal:
      normalized.append((data - min)/(max - min))
    else:
      normalized.append((data * (max - min)) + min)
  return normalized

# Normalization Function:
def normalize(max, min, data, normal=True):
  if normal:
    return (data - min)/(max - min)
  else:
    return (data * (max - min)) + min

### DATA FUNCTIONS ###

# Gets the Numpy Data from Dataframe:
def get_numpy_data(data, features, output):
  data_frame = data.copy()
  data_frame['constant'] = 1

  features = ['constant'] + features 
  features_frame = data_frame[features]

  feature_matrix = features_frame.to_numpy()
  output_array = data_frame[output]

  output_array = output_array.to_numpy()
  return feature_matrix, output_array

# Get Dataset Function:
def get_dataset(data_orig, model_output):
  data = data_orig.copy()
  dates = data['date']
  date_format = [pd.to_datetime(d) for d in dates]

  data_list = data.columns.values.tolist()
  for i in data_list[-11:]:
    data[[i]] = normalize(data_orig[i].max(), data_orig[i].min(), data_orig[[i]], normal=True)

  X = date_format
  day_numbers = []
  for i in range(1, len(X) + 1):
    day_numbers.append([i])

  data['Days'] = pd.DataFrame(day_numbers, columns = ['Days'])
  data["Days"] = data["Days"].astype(int)

  max = data_orig[model_output].max()
  min = data_orig[model_output].min()
  return data, max, min