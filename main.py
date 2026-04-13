from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import logging
import time


model = joblib.load('cancer_pipeline.pkl')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='cancer prediction API')

class CancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

@app.get("/")
def home():
    return {'message': 'cancer API running'}

@app.post("/predict")
def predict(data: CancerInput):
    start_time = time.time()
    row = {
        "mean_radius": data.mean_radius,
        "mean_texture": data.mean_texture,
        "mean_perimeter": data.mean_perimeter,
        "mean_area": data.mean_area,
        "mean_smoothness": data.mean_smoothness,
        "mean_compactness": data.mean_compactness,
        "mean_concavity": data.mean_concavity,
        "mean_concave_points": data.mean_concave_points,
        "mean_symmetry": data.mean_symmetry,
        "mean_fractal_dimension": data.mean_fractal_dimension,
        "radius_error": data.radius_error,
        "texture_error": data.texture_error,
        "perimeter_error": data.perimeter_error,
        "area_error": data.area_error,
        "smoothness_error": data.smoothness_error,
        "compactness_error": data.compactness_error,
        "concavity_error": data.concavity_error,
        "concave_points_error": data.concave_points_error,
        "symmetry_error": data.symmetry_error,
        "fractal_dimension_error": data.fractal_dimension_error,
        "worst_radius": data.worst_radius,
        "worst_texture": data.worst_texture,
        "worst_perimeter": data.worst_perimeter,
        "worst_area": data.worst_area,
        "worst_smoothness": data.worst_smoothness,
        "worst_compactness": data.worst_compactness,
        "worst_concavity": data.worst_concavity,
        "worst_concave_points": data.worst_concave_points,
        "worst_symmetry": data.worst_symmetry,
        "worst_fractal_dimension": data.worst_fractal_dimension,
        
        # "mean radius": data.mean_radius,
        # "mean texture": data.mean_texture,
        # "mean perimeter": data.mean_perimeter,
        # "mean area": data.mean_area,
        # "mean smoothness": data.mean_smoothness,
        # "mean compactness": data.mean_compactness,
        # "mean concavity": data.mean_concavity,
        # "mean concave points": data.mean_concave_points,
        # "mean symmetry": data.mean_symmetry,
        # "mean fractal dimension": data.mean_fractal_dimension,
        # "radius error": data.radius_error,
        # "texture error": data.texture_error,
        # "perimeter error": data.perimeter_error,
        # "area error": data.area_error,
        # "smoothness error": data.smoothness_error,
        # "compactness error": data.compactness_error,
        # "concavity error": data.concavity_error,
        # "concave points error": data.concave_points_error,
        # "symmetry error": data.symmetry_error,
        # "fractal dimension error": data.fractal_dimension_error,
        # "worst radius": data.worst_radius,
        # "worst texture": data.worst_texture,
        # "worst perimeter": data.worst_perimeter,
        # "worst area": data.worst_area,
        # "worst smoothness": data.worst_smoothness,
        # "worst compactness": data.worst_compactness,
        # "worst concavity": data.worst_concavity,
        # "worst concave points": data.worst_concave_points,
        # "worst symmetry": data.worst_symmetry,
        # "worst fractal dimension": data.worst_fractal_dimension,
    }
    df = pd.DataFrame([row])

    logger.info(f"mean_radius input= {df["mean_radius"].iloc[0]}")
    logger.info(f"worst_area input: {df["worst_area"].iloc[0]}")



    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    latency = time.time() - start_time

    logger.info(f"prediction: {int(pred)},probability: {float(prob):.6f}")
    logger.info(f"Inference latency: {latency:.4f} seconds")

    return {
        'prediction': int(pred),
        'probability': float(prob)
    }




