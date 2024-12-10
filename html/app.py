from flask import Flask, render_template, request
from joblib import load
from mlp_model import mlp
import numpy as np

app = Flask(__name__)

# Load your pre-trained machine learning model here
model = load('mlp_model.joblib') 
scaler = load('scaler.joblib')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("predicting...")
    if request.method == "POST":
        # Get user input from the form
        fields = ["size", "weight", "sweetness", "crunchiness", "juiciness", "ripeness", "acidity"]

        input_data = np.array([float(request.form.get(field)) for field in fields])
        

        input_data = input_data.reshape(1,-1)
        print("input data\n")
        print(input_data)

        print("input data shape")
        print(input_data.shape)


        # Preprocess the input data for your model (if needed)
        preprocessed_data = scaler.transform(input_data)

        print("data has been preprocessed")
        print("preprocessed data: \n")
        print(preprocessed_data)

        # Make prediction using your model
        prediction = model.predict(input_data)  # Assuming a list input

        print("prediction: \n")
        print(prediction)

        # Format the prediction for display
        predicted_class = prediction[0]  # Assuming single class output
        print("predicted class: \n")
        print(predicted_class)

        return render_template("results.html", prediction=predicted_class)

    else:
        return "Something went wrong. Please try again."


if __name__ == "__main__":
    app.run(debug=True, port=5500)