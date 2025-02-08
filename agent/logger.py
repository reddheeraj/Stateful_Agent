import logging
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

class AgentLogger:
    _loggers = {}

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        os.makedirs('logs', exist_ok=True)
        
        if agent_id not in AgentLogger._loggers:
            self._init_logger()
            AgentLogger._loggers[agent_id] = self.logger
        else:
            self.logger = AgentLogger._loggers[agent_id]

    def _init_logger(self):
        """Initialize logger only once per agent_id"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True, parents=True)

        # Create new logger instance
        self.logger = logging.getLogger(f"Agent-{self.agent_id}")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers = []

        # Create and configure handler
        log_file = log_dir / f"{self.agent_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def log(self, activity_type: str, details: dict):
        log_message = f"[{activity_type}] {str(details)}"
        self.logger.info(log_message)

    def log_activity(self, activity_type: str, details: Dict[str, Any]):
        log_entry = f"[{activity_type}] {details}"
        self.logger.info(log_entry)