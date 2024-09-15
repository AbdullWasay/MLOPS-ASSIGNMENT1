from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import traceback

app = Flask(__name__)
model = joblib.load('model_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)
        df = pd.DataFrame([data])
        print("DataFrame:", df)
        prediction = model.predict(df)[0]
        print("Prediction:", prediction)
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        print("Error occurred:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

if __name__ == '__main__':
    app.run(debug=True)