"""
SimpleReActFramework implementation for material analysis.
"""
import traceback
from typing import List, Optional
import os

from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

from src.tools.material_tools import MaterialTools
from src.utils.logger import setup_logger
from config.settings import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, AGENT_TYPE, MEMORY_KEY

logger = setup_logger("material_analysis.framework")

class SimpleReActFramework:

    def __init__(self, openai_api_key: str, base_url: str = DEFAULT_BASE_URL, 
                 model_name: str = DEFAULT_MODEL_NAME, temperature: float = DEFAULT_TEMPERATURE,
                 max_history_length: int = 10):
        logger.info("Initializing SimpleReActFramework...")
        
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            base_url=base_url,
            temperature=temperature,
            model_name=model_name
        )
        
        self.memory = ConversationBufferMemory(memory_key=MEMORY_KEY)
        self.max_history_length = max_history_length
        self.material_tools = MaterialTools()
        self.material_tools.set_openai_client(openai_api_key)
        self.tools = self._setup_tools()
        
        self.agent = self._setup_agent(handle_parsing_errors=True)
        
        logger.info("SimpleReActFramework initialization completed")
    
    def _setup_tools(self) -> List[Tool]:
        tools = [
            Tool(
                name="model_predict",
                func=self.material_tools.model_predict,
                description="Predict material properties using the trained model. Parameter: data_path (str): Path to the input data CSV file, example: ./data/test.csv. The model will be loaded from ./model/model.cbm. Results will be saved as [input_filename]_predict.csv in the same directory as the input file."
            ),
            Tool(
                name="criterion_match",
                func=self.material_tools.criterion_match,
                description="Filter materials containing rare earth elements from a CSV file. Parameter: data_path (str): Path to the input data CSV file, example: ./data/test.csv. Results will be saved as [input_filename]_rare_earth.csv in the same directory as the input file."
            ),
            Tool(
                name="model_train",
                func=self.material_tools.model_train,
                description="Train a CatBoost model using a single data file. Parameter: data_path (str): Path to the input data CSV file, example: ./data/train.csv. The model will be automatically saved to ./model/model.cbm. Returns training metrics including accuracy, precision, recall, and F1 score."
            ),
            Tool(
                name="information_extract",
                func=self.material_tools.information_extract,
                description="Extract material information from all literature in a directory sequentially. Parameter: input_dir (str): Path to the directory containing PDF files, without quotes. Example: ./data/literature/."
            ),
            Tool(
                name="importance_analysis",
                func=self.material_tools.importance_analysis,
                description="Analyze feature importance using novel SHAP values. Parameter: data_path (str): Path to the input data CSV file, example: ./data/test.csv. The file should contain either MAGPIE features or a 'Substance' column for automatic conversion."
            ),
            Tool(
                name="feature_engineering",
                func=self.material_tools.feature_engineering,
                description="Perform feature engineering on material data. Parameter: data_path (str): Path to the input data file, exmaple: data/train.csv. The function will generate statistics and weighted features, saving them in the same directory as the input file."
            )
        ]
        return tools

    def _setup_agent(self, handle_parsing_errors: bool):
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=eval(f"AgentType.{AGENT_TYPE}"),
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=handle_parsing_errors
        )
        return agent

    def clear_history(self):
        self.memory.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[dict]:
        return self.memory.chat_memory.messages

    def run(self, query: str, clear_history: bool = False) -> str:
        if clear_history:
            self.clear_history()
            
        logger.info(f"Executing query: {query}")
        try:
            response = self.agent.run(query)
            logger.info("Query execution successful")
            return response
        except Exception as e:
            error_msg = f"Agent execution error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return error_msg