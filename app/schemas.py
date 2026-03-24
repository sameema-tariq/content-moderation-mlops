"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    """Input schema — raw text to be classified."""

    text: str = Field(..., min_length=1, description="Raw text to classify")


class PredictionResponse(BaseModel):
    """Output schema — predicted label and model confidence."""

    label: Literal["spam", "ham"] = Field(..., description="Predicted label: spam or ham")
    confidence: float = Field(..., description="Model confidence score (0-1)")
