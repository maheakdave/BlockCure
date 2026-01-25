import structlog
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import uuid
import time

def configure_logging()->None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer() 
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

def get_logger()->structlog.BoundLogger:
    logger = structlog.get_logger()
    return logger

class StructlogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, dispatch = None):
        super().__init__(app, dispatch)
        self.logger = get_logger()

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else "unknown",
        )

        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            process_time = time.perf_counter() - start_time
            
            self.logger.info(
                "request_finished",
                status_code=response.status_code,
                duration=f"{process_time:.4f}s"
            )
            return response

        except Exception as e:
            
            process_time = time.perf_counter() - start_time
            self.logger.exception(
                "request_failed",
                error=str(e),
                duration=f"{process_time:.4f}s"
            )
            raise
