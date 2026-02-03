"""Timer utility for precise benchmarking."""
import time
from typing import Optional
from contextlib import contextmanager


class Timer:
    """Precise timer for benchmarking operations."""
    
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: float = 0.0
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        if self.start_time is None:
            raise RuntimeError("Timer was never started")
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return self.elapsed_ms
    
    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms


@contextmanager
def time_operation(operation_name: str):
    """Context manager for timing operations."""
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        print(f"[TIMER] {operation_name}: {elapsed:.2f}ms")