"""
Main entry point for the material analysis framework.
"""
from src.framework import SimpleReActFramework
from config.settings import API_KEY
from src.utils.logger import setup_logger
import argparse

logger = setup_logger("material_analysis.main")

def get_user_query():
    parser = argparse.ArgumentParser(description="Material Analysis Framework")
    parser.add_argument("--query", type=str, help="Query to analyze")
    parser.add_argument("--web", action="store_true", help="Start web interface")
    args = parser.parse_args()
    
    if args.web:
        from app import app
        app.run(debug=True)
        return None
    elif args.query:
        return args.query
    else:
        return input("Input your analysis query: ")

def main():
    logger.info("Starting material analysis application")
    
    query = get_user_query()
    
    if query is None:
        return
    
    react_framework = SimpleReActFramework(openai_api_key=API_KEY)
    result = react_framework.run(query)
    
    print(result)
    logger.info("Application execution completed")

if __name__ == "__main__":
    main()