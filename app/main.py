from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as exc:
    missing = MODEL_PATH if not MODEL_PATH.exists() else VECTORIZER_PATH
    raise RuntimeError(f"Required artifact missing: {missing}") from exc

LABELS = {0: "Not Liked", 1: "Liked"}

app = FastAPI(title="Restaurant Review Sentiment", version="1.0.0")
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "app" / "static"),
    name="static",
)
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")


class ReviewPayload(BaseModel):
    review: str = Field(..., min_length=3, max_length=2000)


class PredictionResponse(BaseModel):
    review: str
    prediction: Literal["Liked", "Not Liked"]
    confidence: float


def predict_label(text: str) -> PredictionResponse:
    transformed = vectorizer.transform([text])
    class_prob = model.predict_proba(transformed)[0]
    predicted_idx = int(class_prob.argmax())
    confidence = float(class_prob[predicted_idx])
    return PredictionResponse(
        review=text,
        prediction=LABELS[predicted_idx],
        confidence=round(confidence, 4),
    )


@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None},
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_from_form(request: Request, review: str = Form(...)):
    try:
        result = predict_label(review)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result},
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict_from_api(payload: ReviewPayload):
    return predict_label(payload.review)


if __name__ == "__main__":
    import uvicorn
    import sys
    # Add project root to sys.path so 'app.main' module can be found by uvicorn
    sys.path.append(str(PROJECT_ROOT))
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
