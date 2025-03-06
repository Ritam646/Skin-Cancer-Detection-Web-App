from flask import Flask, request, render_template
import numpy as np
import joblib
app = Flask(__name__)
model = joblib.load("model/skin_cancer_model.pkl")
@app.route('/')
def home():
    return render_template('index.html') 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4']),
            float(request.form['feature5']),
            float(request.form['feature6']),
            float(request.form['feature7']),
            float(request.form['feature8']),
            float(request.form['feature9']),
            float(request.form['feature10'])
        ]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        confidence = model.predict_proba(features).max() * 100
        result = "Malignant" if prediction[0] == 1 else "Benign"
        return render_template("result.html", result=result, confidence=round(confidence, 2))
    except Exception as e:
        return f"Error in prediction: {str(e)}"
if __name__ == '__main__':
    app.run(debug=True)