import sys
import traceback
from typing import Optional


class CustomException(Exception):
    """Custom exception with detailed error tracking."""
    
    def __init__(
        self,
        error_message: str,
        error_detail: Optional[sys.exc_info] = None
    ):
        super().__init__(error_message)
        self.error_message = error_message
        
        if error_detail:
            _, _, exc_tb = error_detail
            if exc_tb:
                self.file_name = exc_tb.tb_frame.f_code.co_filename
                self.line_number = exc_tb.tb_lineno
                self.function_name = exc_tb.tb_frame.f_code.co_name
            else:
                self.file_name = "Unknown"
                self.line_number = 0
                self.function_name = "Unknown"
        else:
            self.file_name = "Unknown"
            self.line_number = 0
            self.function_name = "Unknown"
    
    def __str__(self) -> str:
        return (
            f"Error in [{self.file_name}] "
            f"function [{self.function_name}] "
            f"line [{self.line_number}]: {self.error_message}"
        )
    
    def get_detailed_traceback(self) -> str:
        """Get full traceback for logging."""
        return traceback.format_exc()
