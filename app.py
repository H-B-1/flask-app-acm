from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Add a route for the root URL
@app.route('/', methods=['GET'])
def home():
    return "Flask app is running. Use '/predict' endpoint to make predictions."

# Load the model
model = joblib.load('cookie_decision_tree_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)

        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
