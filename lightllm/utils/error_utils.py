class ServerBusyError(Exception):
    """Custom exception for server busy/overload situations"""

    def __init__(self, message="Server is busy, please try again later", status_code=503):
        """
        Initialize the ServerBusyError

        Args:
            message (str): Error message to display
            status_code (int): HTTP status code (default 503 Service Unavailable)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code  # HTTP 503 Service Unavailable

    def __str__(self):
        """String representation of the error"""
        return f"{self.message} (Status code: {self.status_code})"

class KVMoveTimeoutError(ServerBusyError):
    """KV移动超时错误，用于触发重试机制"""
    def __init__(self, message="KV move timeout, please try again later", status_code=503):
        super().__init__(message, status_code)