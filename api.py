from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging
from db_connector import DatabaseConnector
from water_leakage_analysis import WaterLeakageAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Water Leakage Detection API",
    description="API to check for water leakage in specific devices using 7 analysis methods.",
    version="1.0.0"
)

from fastapi.security import APIKeyHeader

# Security
API_TOKEN = os.environ.get("API_TOKEN", "default-secret-token")
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=True)

async def verify_token(token: str = Depends(api_key_header)):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API Token")
    return token

class LeakCheckResponse(BaseModel):
    device_name: str
    is_leaking: bool
    risk_score: int
    details: Dict[str, Any]

@app.get("/")
def read_root():
    return {"status": "online", "service": "Water Leakage Detection API"}

@app.get("/leak-check/{device_name}", response_model=LeakCheckResponse)
def check_leak(device_name: str, token: str = Depends(verify_token)):
    """
    Check for leaks in a specific device.
    """
    logger.info(f"Received leak check request for device: {device_name}")
    
    try:
        # 1. Fetch data
        connector = DatabaseConnector()
        df = connector.fetch_data_for_device(device_name)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for device {device_name} in the last 7 days")
        
        # 2. Run Analysis
        analyzer = WaterLeakageAnalyzer(dataframe=df)
        success = analyzer.run_all_analyses()
        
        if not success:
            raise HTTPException(status_code=500, detail="Analysis failed to execute")
            
        # 3. Get Results
        results = analyzer.get_results_summary()
        
        return {
            "device_name": device_name,
            "is_leaking": results['is_leaking'],
            "risk_score": results['risk_score'],
            "details": results['details']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
