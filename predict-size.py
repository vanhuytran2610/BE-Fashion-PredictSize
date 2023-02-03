from flask import Flask, jsonify, request, Response
import pickle
import json

app = Flask(__name__)

@app.route("/predict-size", methods=["POST"])
def predict():
        weight = request.json["weight"]
        age = request.json["age"]
        height = request.json["height"]

        file_model = "model.pkl"
        load_model = pickle.load(open(file_model, 'rb'))

        size = ["XXS","S","M","L","XL","XXL","XXXL"]
        prediction = size[load_model.predict([[weight, age, height]])[0]]

        if prediction:
            serve_model = {
                "Status": "OK",
                "Cloth Size": prediction
            }
            return jsonify(serve_model)
        else:
            error = {
                "Status": "Error",
                "Message": "Can not predict cloth-size, please re-enter"
            }
            return jsonify(error)

if __name__ == '__main__':
    app.run(debug=True)

    
