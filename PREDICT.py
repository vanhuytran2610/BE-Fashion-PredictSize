from flask import Flask, jsonify, request
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict-size", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        weight = data["weight"]
        height = data["height"]

        with open("D:\FashionApp\Predict-Size\BE-Fashion-PredictSize\model.pkl", "rb") as file: 
            load_model = pickle.load(file)

        size = ["XS", "S", "M", "L", "XL", "XXL"]
        probabilities = load_model.predict_proba([[weight, height]])[0]
        sorted_indexes = probabilities.argsort()[::-1][:2]  # Get the indices of the top 2 probabilities
        predictions = [{"size": size[i], "probability": probabilities[i]} for i in sorted_indexes]

        serve_model = {
            "status": 200,
            "predictions": predictions
        }
        return jsonify(serve_model)

    except KeyError:
        error = {
            "status": 400,
            "message": "Invalid request payload"
        }
        return jsonify(error)

    except Exception as e:
        error = {
            "status": 500,
            "message": "An error occurred while processing the request"
        }
        return jsonify(error)

if __name__ == '__main__':
    app.run(debug=True)
