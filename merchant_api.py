from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# 1) Load your artifacts
scaler    = joblib.load('scaler.pkl')
rf_model  = joblib.load('random_forest_model.pkl')
gb_model  = joblib.load('gradient_boosting_model.pkl')
meta      = joblib.load('model_metadata.pkl')

FEATURE_ORDER = meta['features']
RF_WEIGHT     = meta['rf_weight']
GB_WEIGHT     = meta['gb_weight']

app = FastAPI(title="Merchant Trust API")

# 2) Define the input schema
class MerchantData(BaseModel):
    quality_return_rate: float
    defect_rate: float
    authenticity_score: float
    quality_sentiment: float
    on_time_delivery_rate: float
    shipping_accuracy: float
    order_fulfillment_rate: float
    inventory_accuracy: float
    avg_rating_normalized: float
    review_sentiment: float
    positive_review_ratio: float
    review_authenticity: float
    response_time_score: float
    query_resolution_rate: float
    service_satisfaction: float
    proactive_communication: float

# 3) Helper to get letter grade
def letter_grade(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Moderate"
    return ""  # hide below 40

# 4) Prediction endpoint
@app.post("/predict_merchant")
def predict_merchant(data: MerchantData):
    # build feature vector
    raw = [getattr(data, feat) for feat in FEATURE_ORDER]
    X   = np.array(raw, dtype=np.float32).reshape(1, -1)
    
    # scale + predict
    Xs   = scaler.transform(X)
    rf_p  = rf_model.predict(Xs)[0]
    gb_p  = gb_model.predict(Xs)[0]
    
    # ensemble
    pred = RF_WEIGHT * rf_p + GB_WEIGHT * gb_p

    # Precompute the severe‐quality flag
    severe_quality = (data.quality_return_rate > 0.15 and data.defect_rate > 0.1)

    # 1) Heavy penalty
    if severe_quality:
        pred -= 30

    # 2) Tiered penalties for other metrics

    # Return‐rate tiers
    if data.quality_return_rate > 0.10:
        pred -= 20   # severe
    elif data.quality_return_rate > 0.05:
        pred -= 10   # moderate

    # On‐time delivery tiers
    if data.on_time_delivery_rate < 0.80:
        pred -= 20   # severe
    elif data.on_time_delivery_rate < 0.90:
        pred -= 10   # moderate

    # Average rating tiers
    if data.avg_rating_normalized < 2.5:
        pred -= 15   # very low ratings
    elif data.avg_rating_normalized < 3.5:
        pred -= 5    # somewhat low ratings

    # Shipping accuracy tiers
    if data.shipping_accuracy < 0.80:
        pred -= 15
    elif data.shipping_accuracy < 0.90:
        pred -= 5

    # Review authenticity tiers
    if data.review_authenticity < 0.40:
        pred -= 15
    elif data.review_authenticity < 0.60:
        pred -= 5

    # Response time tiers
    if data.response_time_score < 0.50:
        pred -= 10
    elif data.response_time_score < 0.70:
        pred -= 5

    # Service satisfaction tiers
    if data.service_satisfaction < 0.50:
        pred -= 10
    elif data.service_satisfaction < 0.70:
        pred -= 5

    # 3) Ensure floor for severe cases
    if severe_quality:
        pred = max(pred, 10)

    # 4) Clip & grade
    score = float(np.clip(pred, 0, 100))
    grade = letter_grade(score)

    return {
        "trust_score": round(score, 2),
        "grade": grade
    }
