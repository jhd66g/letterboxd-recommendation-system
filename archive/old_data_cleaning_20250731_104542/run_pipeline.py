#!/usr/bin/env python3
"""
Simple Pipeline Runner

Runs the complete Letterboxd data pipeline according to your specifications.
This script replaces the complex previous versions with a straightforward approach.
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete streamlined pipeline"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info("🚀 Starting streamlined Letterboxd data pipeline")
    logger.info(f"📁 Data directory: {data_dir}")
    
    try:
        # Import the pipeline module
        from streamlined_pipeline import main as run_pipeline
        success = run_pipeline()
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
            
            # Run the demo
            logger.info("🧪 Running demo...")
            try:
                # Change to parent directory to run demo
                parent_dir = os.path.dirname(os.path.dirname(__file__))
                os.chdir(parent_dir)
                
                # Run the demo
                import subprocess
                result = subprocess.run([sys.executable, 'streamlined_demo.py'], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Demo completed successfully!")
                    logger.info("Demo output:")
                    logger.info(result.stdout)
                else:
                    logger.error("❌ Demo failed:")
                    logger.error(result.stderr)
                    
            except Exception as e:
                logger.error(f"⚠️  Could not run demo: {e}")
        else:
            logger.error("❌ Pipeline failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
