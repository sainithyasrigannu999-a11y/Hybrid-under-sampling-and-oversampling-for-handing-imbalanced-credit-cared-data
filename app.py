import random
import time
import uuid
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Global state for the dashboard
transaction_history = []
stats = {"total": 0, "blocked": 0, "safe": 0}
simulation_active = False

# Attempt to load the model (ensure you ran model_train.py first)
try:
    model = pickle.load(open('upi_model.pkl', 'rb'))
except:
    model = None
    print("Warning: upi_model.pkl not found. Running in Rule-only mode.")

def evaluate_transaction(amount, hour, velocity):
    """
    Hybrid Decision Engine:
    Combines ML Probability with Business Logic (Middleware Layer)
    """
    reasons = []
    base_prob = 0.1 # Default baseline
    
    # 1. ML Model Prediction (if available)
    if model:
        features = np.array([[amount, hour, velocity]])
        base_prob = model.predict_proba(features)[0][1]

    # 2. Heuristic Rules & XAI Reasoning
    if hour < 5 or hour > 23:
        base_prob += 0.35
        reasons.append("Midnight/Early morning transaction (High-risk window).")
    
    if velocity > 7:
        base_prob += 0.45
        reasons.append("High transaction frequency (Possible bot activity).")
        
    if amount > 80000:
        base_prob += 0.25
        reasons.append("Transaction amount exceeds standard individual thresholds.")

    # Final Risk Calculation
    risk_score = min(base_prob, 1.0)
    is_fraud = risk_score > 0.5
    
    # Update Dashboard Stats
    stats["total"] += 1
    if is_fraud:
        stats["blocked"] += 1
    else:
        stats["safe"] += 1

    return {
        "id": str(uuid.uuid4())[:8].upper(),
        "amount": f"{amount:,}",
        "hour": f"{hour:02d}:00",
        "velocity": velocity,
        "risk": f"{risk_score:.0%}",
        "status": "BLOCKED" if is_fraud else "APPROVED",
        "color": "danger" if is_fraud else "success",
        "time": time.strftime("%H:%M:%S"),
        "details": " | ".join(reasons) if reasons else "Pattern matches normal user behavior."
    }

@app.route('/')
def home():
    return render_template('index.html', history=transaction_history, stats=stats, sim_on=simulation_active)

@app.route('/predict', methods=['POST'])
def predict():
    amount = float(request.form.get('amount', 0))
    hour = int(request.form.get('hour', 0))
    velocity = int(request.form.get('velocity', 0))
    
    res = evaluate_transaction(amount, hour, velocity)
    transaction_history.insert(0, res)
    return jsonify(res)

@app.route('/toggle_sim')
def toggle_sim():
    global simulation_active
    simulation_active = not simulation_active
    
    if simulation_active:
        # Generate 5 random transactions to simulate live traffic
        for _ in range(5):
            is_evil = random.random() > 0.7
            res = evaluate_transaction(
                random.randint(30000, 99000) if is_evil else random.randint(50, 5000),
                random.randint(0, 4) if is_evil else random.randint(9, 21),
                random.randint(8, 15) if is_evil else random.randint(1, 3)
            )
            transaction_history.insert(0, res)
    return jsonify({"active": simulation_active})

@app.route('/clear')
def clear():
    global transaction_history, stats
    transaction_history = []
    stats = {"total": 0, "blocked": 0, "safe": 0}
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)