import os
import time
import schedule
import logging
from datetime import datetime
from db_connector import DatabaseConnector
from water_leakage_analysis import WaterLeakageAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("leakage_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

def run_analysis():
    """
    Main function to run the water leakage analysis.
    Fetches data from DB, runs analysis, and saves results.
    """
    logger.info("Starting nightly water leakage analysis...")
    
    try:
        # 1. Connect to DB and fetch data
        connector = DatabaseConnector()
        df = connector.fetch_last_7_days_data()
        
        if df.empty:
            logger.warning("No data fetched from database. Aborting analysis.")
            return
        
        # 2. Initialize Analyzer with fetched data
        analyzer = WaterLeakageAnalyzer(dataframe=df)
        
        # 3. Run Analysis
        success = analyzer.run_all_analyses()
        
        if success:
            logger.info("Analysis completed successfully.")
            
            # 4. Save results (The analyzer already generates a report and plots)
            # We can also explicitly save the leakage indicators to a CSV if needed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create reports directory if it doesn't exist
            os.makedirs("nightly_reports", exist_ok=True)
            
            # Generate comprehensive report dataframe
            leak_df = analyzer.generate_comprehensive_report()
            
            if not leak_df.empty:
                report_file = f"nightly_reports/leakage_report_{timestamp}.csv"
                leak_df.to_csv(report_file, index=False)
                logger.info(f"Leakage report saved to {report_file}")
            else:
                logger.info("No significant leakage indicators found to save.")
                
            # Move visualizations to nightly folder
            # if os.path.exists("water_leakage_analysis.png"):
            #     os.rename("water_leakage_analysis.png", f"nightly_reports/visualization_{timestamp}.png")
                
        else:
            logger.error("Analysis failed during execution.")
            
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")

def main():
    """
    Entry point. Checks for scheduling mode or runs once.
    """
    schedule_mode = os.environ.get('SCHEDULE_MODE', 'false').lower() == 'true'
    
    if schedule_mode:
        logger.info("Starting in SCHEDULED mode. Will run every night at 02:00.")
        # Schedule to run every day at 02:00 AM
        schedule.every().day.at("01:00").do(run_analysis)
        
        # Also run once immediately on startup to verify
        logger.info("Running initial check on startup...")
        run_analysis()
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        logger.info("Starting in RUN-ONCE mode.")
        run_analysis()

if __name__ == "__main__":
    main()
