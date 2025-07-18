from flask import Flask, render_template, request
import joblib

# Load saved model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        message = request.form["message"]
        clean = message.lower()
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        result = "Spam" if pred == 1 else "Not Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)