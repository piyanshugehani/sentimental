from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

# Set of stopwords
STOPWORDS = set(stopwords.words("english"))

# Initialize Flask app with CORS
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get the JSON data from the request
    input_text = data.get('text')  # Extract the 'text' field from the JSON
    print(input_text)
    # Load the models and scalers from the Models folder
    try:
        predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
        scaler = pickle.load(open("Models/scaler.pkl", "rb"))
        cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))
    except Exception as e:
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)
            print("pred:",predictions)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")

            return response

        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            print(predicted_sentiment)

            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    
    # Step 1: Preprocess the text
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)

    # Debugging: Print preprocessed text
    print("Preprocessed Text:", corpus)
    
    # Step 2: Transform using CountVectorizer
    X_prediction = cv.transform(corpus).toarray()
    print("CountVectorizer Output:", X_prediction)
    
    # Step 3: Scale the features
    X_prediction_scl = scaler.transform(X_prediction)
    print("Scaled Features:", X_prediction_scl)

    # Step 4: Predict the probability and choose the class
    y_predictions = predictor.predict_proba(X_prediction_scl)
    print("Predicted Probabilities:", y_predictions)
    
    # Step 5: Get the class with the highest probability
    y_predictions = y_predictions.argmax(axis=1)[0]
    print("Predicted Class:", y_predictions)

    return "Positive" if y_predictions == 1 else "Negative"

    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]


    return "Positive" if y_predictions == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    graph.seek(0)  # Ensure the pointer is at the beginning of the BytesIO object for reading
    return graph


def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
