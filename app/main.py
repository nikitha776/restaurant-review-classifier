from pathlib import Path
from typing import Literal, Dict, List, Optional
import sys
import re
import joblib
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure app.text_utils can be imported
sys.path.append(str(PROJECT_ROOT))

from app.text_utils import preprocess_text

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


class AspectSentence(BaseModel):
    sentence: str
    prediction: Literal["Liked", "Not Liked"]
    confidence: Optional[float]


class PredictionResponse(BaseModel):
    review: str
    prediction: Literal["Liked", "Not Liked"]
    confidence: Optional[float]
    aspects: Dict[str, List[AspectSentence]]


# Aspect Keywords & Regex Configuration
aspects_keywords = {
    'food': ['food','taste','flavor','dish','meal','menu','tasteful','tasty'],
    'service': ['service','staff','waiter','waitress','server','host','manager'],
    'speed': ['quick','slow','speed','time','wait','waited'],
    'hygiene': ['hygiene','clean','dirty','sanitary','unclean','hygienic','cleanliness','neat'],
    'ambience': ['ambience','ambiance','atmosphere','music','decor','lighting'],
    'price': ['price','cost','expensive','cheap','value','worth','money'],
}

aspects_regex = {
    asp: re.compile(r'\b(' + r'|'.join(map(re.escape, kws)) + r')\b', flags=re.IGNORECASE)
    for asp, kws in aspects_keywords.items()
}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def predict_sentiment(text: str):
    """Helper to predict sentiment for a given text snippet."""
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    
    pred_idx = int(model.predict(vec)[0])
    
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(vec)[0]
            confidence = float(probs.max())
        except Exception:
            confidence = None
            
    return pred_idx, confidence


def analyze_review(text: str) -> PredictionResponse:
    # 1. Overall Prediction
    pred_idx, confidence = predict_sentiment(text)
    
    # 2. Aspect Extraction
    aspect_results = {}
    sentences = split_sentences(text)
    
    for s in sentences:
        for asp, rx in aspects_regex.items():
            if rx.search(s):
                # Predict sentiment for this specific sentence
                # Use the overall prediction as fallback if this fails (though usually it won't)
                try:
                    s_pred_idx, s_conf = predict_sentiment(s)
                except Exception:
                    s_pred_idx = pred_idx
                    s_conf = None
                
                if asp not in aspect_results:
                    aspect_results[asp] = []
                
                aspect_results[asp].append(AspectSentence(
                    sentence=s,
                    prediction=LABELS[s_pred_idx],
                    confidence=round(s_conf, 4) if s_conf is not None else None
                ))

    return PredictionResponse(
        review=text,
        prediction=LABELS[pred_idx],
        confidence=round(confidence, 4) if confidence is not None else None,
        aspects=aspect_results
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
        result = analyze_review(review)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result},
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict_from_api(payload: ReviewPayload):
    return analyze_review(payload.review)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
