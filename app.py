from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import string

app = Flask(__name__)
CORS(app)  # Allow mobile app / frontend access

# -----------------------------
# Load Model + Vectorizer
# -----------------------------
print("Loading model and vectorizer...")
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: model.pkl or vectorizer.pkl not found!")


# -----------------------------
# Preprocess Function
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text


# -----------------------------
# Home (GET)
# -----------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Spam Detector API is running",
        "routes": {
            "/": "Home",
            "/health": "Check API status",
            "/predict": "POST request with JSON: {'message': 'text here'}"
        }
    })


# -----------------------------
# Health Check (GET)
# -----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "server": "Spam Detector API"})


# -----------------------------
# Predict (POST only)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({"error": "Please send JSON like {'message': 'your text'}"}), 400

        message = data['message']
        cleaned = preprocess_text(message)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        confidence = float(model.predict_proba(vectorized).max())

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "message": message
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# If user sends GET to /predict → Show helpful message
# -----------------------------
@app.route('/predict', methods=['GET'])
def predict_wrong_method():
    return jsonify({
        "error": "Use POST method for /predict",
        "example": {"message": "your text here"}
    }), 405


# -----------------------------
# Custom 404 Handler
# -----------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Route not found",
        "valid_routes": ["/", "/health", "/predict"]
    }), 404


# -----------------------------
# Start Server
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
