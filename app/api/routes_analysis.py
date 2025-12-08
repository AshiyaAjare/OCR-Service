# app/api/routes_analysis.py

"""
API routes for house vs broker analysis.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.database import get_db
from app.models.schemas import (
    HouseVsBrokerAnalysisResponse,
    HouseVsBrokerAnalysisItem,
)
from app.services.house_vs_broker_service import analyze_house_vs_broker


router = APIRouter(prefix="/api/v1/house-vs-broker", tags=["analysis"])


class CompanyIdRequest(BaseModel):
    """Request body for company ID(s)"""
    company_ids: List[int] = Field(..., description="List of company IDs to analyze", min_items=1)


@router.get(
    "/",
    response_model=HouseVsBrokerAnalysisResponse,
    summary="House vs Broker Analysis (GET - Query Parameters)",
    description=(
        "Perform house vs broker analysis for one or more companies. "
        "Accepts company_id as a query parameter (single ID or comma-separated list)."
    ),
)
def get_house_vs_broker_analysis(
    company_id: Optional[str] = Query(
        None,
        description=(
            "Company ID(s) to analyze. Can be a single integer or comma-separated list of integers. "
            "Example: ?company_id=1 or ?company_id=1,2,3"
        ),
    ),
    db: Session = Depends(get_db),
):
    """
    GET endpoint for house vs broker analysis.
    
    Supports both single company_id and multiple company_ids via query parameter.
    Query parameters are always strings in FastAPI, so we parse them here.
    """
    if company_id is None:
        raise HTTPException(
            status_code=400,
            detail="company_id query parameter is required"
        )
    
    # Parse company_id(s) - handle comma-separated string
    company_ids = []
    try:
        # Split by comma and convert to integers
        company_ids = [int(id_str.strip()) for id_str in company_id.split(",") if id_str.strip()]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid company_id format: {company_id}. Expected integer(s) or comma-separated integers."
        )
    
    if not company_ids:
        raise HTTPException(
            status_code=400,
            detail="At least one valid company_id is required"
        )
    
    # Perform analysis
    results = analyze_house_vs_broker(db, company_ids)
    
    # Convert to response format
    response_items = [
        HouseVsBrokerAnalysisItem(**result) for result in results
    ]
    
    return HouseVsBrokerAnalysisResponse(
        results=response_items,
        meta={"count": len(response_items)}
    )


@router.post(
    "/",
    response_model=HouseVsBrokerAnalysisResponse,
    summary="House vs Broker Analysis (POST - Request Body)",
    description=(
        "Perform house vs broker analysis for one or more companies. "
        "Accepts company_ids as a list in the request body."
    ),
)
def post_house_vs_broker_analysis(
    request: CompanyIdRequest,
    db: Session = Depends(get_db),
):
    """
    POST endpoint for house vs broker analysis.
    
    Accepts a JSON body with a list of company_ids.
    """
    if not request.company_ids:
        raise HTTPException(
            status_code=400,
            detail="company_ids list cannot be empty"
        )
    
    # Perform analysis
    results = analyze_house_vs_broker(db, request.company_ids)
    
    # Convert to response format
    response_items = [
        HouseVsBrokerAnalysisItem(**result) for result in results
    ]
    
    return HouseVsBrokerAnalysisResponse(
        results=response_items,
        meta={"count": len(response_items)}
    )

