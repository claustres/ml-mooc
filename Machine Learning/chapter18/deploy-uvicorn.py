import pickle
import pandas as pd
from joblib import dump, load
import uvicorn
from fastapi import FastAPI

app = FastAPI()

class Data(dict):
  # The model sata is shared by all samples
  variables = load('variables.joblib')
  categories = load('categories.joblib')
  features = load('features.joblib')
  dummies = load('dummies.joblib')
  model = load('model.joblib')
  
  def __init__(self, values):
      # Start by defining required variables
      for variable in self.variables:
          self[variable] = values[variable]

  def predict(self):
    # Create a dataframe from variables so that we can get dummies
    self.data = pd.DataFrame(self, columns = self.variables, index=[0])
    self.data = pd.get_dummies(self.data, columns = self.categories)
    # Ensure we have all required dummies as the sampel does not contains all possible categorical values
    self.data = self.data.reindex(columns = self.dummies, fill_value=0)
    # As we predict only a single sample return the value directly
    return self.model.predict(self.data[self.features])[0]

@app.post('/predict')
def predict(payload: dict):
  data = Data(payload)
  return { 'y': data.predict() }
