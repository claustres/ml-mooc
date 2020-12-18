from flask import Flask, request, render_template, jsonify
from predict import Data

app = Flask('app')

@app.route('/')
def form():
  return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
  data = Data(request.json)
  return jsonify({ 'y': data.predict() })

@app.route('/predict-form', methods=['POST'])
def predictForm():
  data = Data(request.form)
  return render_template('predict.html',y=data.predict())

app.run(port=8000)