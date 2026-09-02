"""
AegisAI — FastAPI Backend

Provides REST API endpoints for predicting phishing in URLs, text messages,
and QR codes, as well as providing SHAP explanations.

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

from typing import Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.ml.url_model import predict_url
from src.ml.text_model import predict_text
from src.qr.classifier import process_qr
from src.explainability.url_explainer import explain_url
from config.settings import SETTINGS

app = FastAPI(
    title="AegisAI API",
    description="Machine-learning-based phishing detection API for URLs, Text, and QR codes.",
    version="1.0.0"
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class URLRequest(BaseModel):
    url: str = Field(..., description="The URL string to classify or explain.", min_length=1)

class TextRequest(BaseModel):
    text: str = Field(..., description="The text message (SMS/Email) to classify.", min_length=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health Check")
async def health_check() -> dict[str, str]:
    """Verify that the API server is running."""
    return {"status": "ok", "message": "AegisAI API is running"}


@app.post("/predict/url", summary="Predict URL Phishing")
async def api_predict_url(request: URLRequest) -> dict[str, Any]:
    """
    Classify a URL as phishing or legitimate.
    
    IMPORTANT: The URL is treated purely as string data. 
    The API will NOT visit or download the URL.
    """
    try:
        result = predict_url(request.url)
        if result.get("verdict") == "invalid":
            raise HTTPException(status_code=400, detail=result.get("error", "Invalid URL."))
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/predict/text", summary="Predict Text/SMS/Email Phishing")
async def api_predict_text(request: TextRequest) -> dict[str, Any]:
    """Classify a text string as phishing (spam) or legitimate (ham)."""
    try:
        result = predict_text(request.text)
        if result.get("verdict") == "invalid":
            raise HTTPException(status_code=400, detail=result.get("error", "Invalid text."))
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/predict/qr", summary="Predict QR Code Phishing")
async def api_predict_qr(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Detect and decode a QR code from an uploaded image.
    If the payload is a URL, it is safely routed to the URL phishing classifier.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Read file bytes securely
        image_bytes = await file.read()
        if len(image_bytes) > SETTINGS.MAX_QR_IMAGE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Image size exceeds maximum limit.")
            
        result = process_qr(image_bytes)
        
        if result.get("verdict") == "error":
             raise HTTPException(status_code=400, detail=result.get("risk_indicators", ["Error processing QR payload."])[0])
             
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Dependent model not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/explain/url", summary="Explain URL Phishing Prediction")
async def api_explain_url(request: URLRequest) -> dict[str, Any]:
    """
    Generate a human-readable SHAP explanation for a URL prediction.
    """
    try:
        result = explain_url(request.url)
        if result.get("prediction") == "error":
            # Determine if it's a model error or a bad request error
            if "Empty or invalid URL" in result.get("error", ""):
                raise HTTPException(status_code=400, detail=result["error"])
            raise HTTPException(status_code=503, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
