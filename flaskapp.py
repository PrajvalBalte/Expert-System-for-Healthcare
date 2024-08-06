from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    features = np.array([[data['Pregnancies'], data['Glucose'], data['BloodPressure'], data['SkinThickness'], 
                          data['Insulin'], data['BMI'], data['DiabetesPedigreeFunction'], data['Age']]])
    
    # Transform features
    features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features)
    
    # Render result
    result = 'likely to have diabetes.' if prediction[0] == 1 else 'unlikely to have diabetes.'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
