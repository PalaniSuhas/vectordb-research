"""Structured logging utility."""
import sys
from typing import Any
from datetime import datetime


class BenchmarkLogger:
    """Structured logger for benchmark operations."""
    
    @staticmethod
    def info(message: str) -> None:
        """Log info message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] INFO: {message}", file=sys.stdout)
    
    @staticmethod
    def error(message: str) -> None:
        """Log error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ERROR: {message}", file=sys.stderr)
    
    @staticmethod
    def metric(name: str, value: Any) -> None:
        """Log metric."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] METRIC: {name} = {value}", file=sys.stdout)
    
    @staticmethod
    def stage(name: str) -> None:
        """Log stage start."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] STAGE: {name}", file=sys.stdout)
        print("=" * 60, file=sys.stdout)


logger = BenchmarkLogger()