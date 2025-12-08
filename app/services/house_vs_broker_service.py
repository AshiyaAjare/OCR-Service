# app/services/house_vs_broker_service.py

"""
Service for performing house vs broker analysis.

This service compares house valuations (fv_cmp from usermanagement_companylisting)
with broker estimates (broker_estimate from ocr_extracted_document) by matching
companies via company_name.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.db_models import (
    CompanyListingModel,
    CompanyModel,
    ExtractedDocumentModel,
)


def normalize_company_name(name: Optional[str]) -> Optional[str]:
    """
    Normalize company name for matching (case-insensitive, trimmed).
    
    Args:
        name: Company name string
        
    Returns:
        Normalized company name or None if input is None/empty
    """
    if not name:
        return None
    return name.strip().lower()


def get_latest_broker_estimate(
    db: Session,
    company_name: str,
) -> Optional[float]:
    """
    Get the latest broker estimate for a company by matching company_name.
    
    Strategy: If multiple OCR entries exist for the same company_name,
    we select the one with the most recent report_date (or created_at if report_date is null).
    If all have null report_date, we use the most recent created_at.
    
    Args:
        db: Database session
        company_name: Company name to match (will be normalized)
        
    Returns:
        Latest broker_estimate value or None if no match found
    """
    normalized_name = normalize_company_name(company_name)
    if not normalized_name:
        return None
    
    # Query for documents matching the company name (case-insensitive)
    # Order by report_date DESC (nulls last), then created_at DESC
    query = (
        db.query(ExtractedDocumentModel)
        .filter(
            func.lower(func.trim(ExtractedDocumentModel.company_name)) == normalized_name
        )
        .filter(ExtractedDocumentModel.broker_estimate.isnot(None))
        .order_by(
            ExtractedDocumentModel.report_date.desc().nullslast(),
            ExtractedDocumentModel.created_at.desc()
        )
    )
    
    latest_doc = query.first()
    if latest_doc and latest_doc.broker_estimate is not None:
        return float(latest_doc.broker_estimate)
    
    return None


def analyze_house_vs_broker(
    db: Session,
    company_ids: List[int],
) -> List[Dict[str, Any]]:
    """
    Perform house vs broker analysis for given company IDs.
    
    Data flow:
    1. Get fv_cmp (house value) from usermanagement_companylisting for given company_id(s)
    2. Join with usermanagement_company to get company_name
    3. Match company_name with ocr_extracted_document to get broker_estimate
    4. Calculate difference and percentage_difference
    
    Args:
        db: Database session
        company_ids: List of company IDs to analyze
        
    Returns:
        List of analysis results, each containing:
        - company_id
        - company_name
        - house_value (fv_cmp)
        - broker_estimate (from OCR, latest if multiple exist)
        - difference (broker_estimate - house_value)
        - percentage_difference ((broker_estimate - house_value) / house_value * 100)
    """
    if not company_ids:
        return []
    
    # Query company listings with company info
    # Filter by company_id and exclude deleted records
    listings = (
        db.query(CompanyListingModel, CompanyModel)
        .join(CompanyModel, CompanyListingModel.company_id == CompanyModel.company_id)
        .filter(CompanyListingModel.company_id.in_(company_ids))
        .filter(CompanyListingModel.deleted == False)
        .filter(CompanyModel.deleted == False)
        .all()
    )
    
    results = []
    
    for listing, company in listings:
        house_value = None
        if listing.fv_cmp is not None:
            house_value = float(listing.fv_cmp)
        
        company_name = company.company_name
        
        # Get broker estimate by matching company_name
        broker_estimate = get_latest_broker_estimate(db, company_name)
        
        # Calculate difference and percentage_difference
        difference = None
        percentage_difference = None
        
        if house_value is not None and broker_estimate is not None:
            difference = broker_estimate - house_value
            if house_value != 0:
                percentage_difference = (difference / house_value) * 100
        
        results.append({
            "company_id": company.company_id,
            "company_name": company_name,
            "house_value": house_value,
            "broker_estimate": broker_estimate,
            "difference": difference,
            "percentage_difference": percentage_difference,
        })
    
    return results

