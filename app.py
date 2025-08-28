import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Watson credentials from environment
WATSON_API_KEY = os.getenv("WATSON_API_KEY")
WATSON_URL = os.getenv("WATSON_URL")  # Base URL for Watson ML instance
WATSON_DEPLOYMENT_ID = os.getenv("WATSON_DEPLOYMENT_ID")
WATSON_PROJECT_ID = os.getenv("WATSON_PROJECT_ID")
WATSON_MODEL_ID = os.getenv("WATSON_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

# Get Watson IAM token
def get_watson_token():
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "apikey": WATSON_API_KEY,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    res = requests.post(auth_url, headers=headers, data=data)
    res.raise_for_status()
    return res.json()["access_token"]

# Chat endpoint using Watson text/chat API
@app.route("/chat", methods=["POST"])
def chat():
    try:
        req_data = request.get_json()
        user_message = req_data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        token = get_watson_token()

        url = f"{WATSON_URL}/ml/v1/text/chat?version=2023-05-29"
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You always answer the questions with markdown formatting using GitHub syntax. "
                        "The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. "
                        "You must omit that you answer the questions with markdown.\n\n"
                        "Any HTML tags must be wrapped in block quotes, for example ```<html>```. "
                        "You will be penalized for not rendering code in block quotes.\n\n"
                        "When returning code blocks, specify language.\n\n"
                        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\n"
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don't know the answer to a question, please don't share false information."
                    )
                },
                {"role": "user", "content": user_message}
            ],
            "project_id": WATSON_PROJECT_ID,
            "model_id": WATSON_MODEL_ID,
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 1
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        res = requests.post(url, headers=headers, json=body)
        res.raise_for_status()
        watson_data = res.json()

        return jsonify(watson_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Prediction endpoint for structured data models
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "")
        patient_data = data.get("patientData", {})

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        features = preprocess_symptoms(symptoms, patient_data)
        token = get_watson_token()

        endpoint = f"{WATSON_URL}/ml/v4/deployments/{WATSON_DEPLOYMENT_ID}/predictions?version=2021-10-01"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "input_data": [
                {
                    "fields": ["feature1", "feature2"],  # Update with your model's fields
                    "values": [features]
                }
            ]
        }

        res = requests.post(endpoint, headers=headers, json=payload)
        res.raise_for_status()
        watson_response = res.json()

        prediction = watson_response["predictions"][0]["values"][0][0]
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def preprocess_symptoms(symptoms_text: str, patient_data: dict):
    symptoms = symptoms_text.lower().split(",")
    return [
        len(symptoms),  # Number of symptoms
        patient_data.get("age", 0),
        1 if patient_data.get("gender") == "Male" else 0,
        len(patient_data.get("medicalHistory", "").split()),
        len(patient_data.get("currentMeds", "").split()),
        len(patient_data.get("allergies", "").split())
    ]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
