"""
Logging Setup
=============
Structured logging with Loguru.
Every event in the system gets logged — trades, signals, errors, risk events.
"""

import sys
import os
import re
import traceback
from pathlib import Path
from loguru import logger

from src.utils.config import PROJECT_ROOT

SECRET_PATTERNS = [
    re.compile(r"(api[-_]?key|secret|password|token|auth|key|pwd)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.\/]+)['\"]?", re.IGNORECASE),
    re.compile(r"Bearer\s+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE),
    re.compile(r"X-API-Key\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]+)['\"]?", re.IGNORECASE),
]

def redact_secrets(message: str) -> str:
    """Strip credentials or sensitive parameters from log messages."""
    if not isinstance(message, str):
        return message
        
    sensitive_values: list[str] = []
    for env_name in [
        "API_ACCESS_KEY",
        "METALPRICE_API_KEY",
        "GOLD_API_KEY",
        "ALPHAVANTAGE_API_KEY",
        "CLERK_SECRET_KEY",
        "QUESTDB_PASSWORD"
    ]:
        val = os.getenv(env_name)
        if val and len(val) > 4:
            sensitive_values.append(val)
            
    sensitive_values.append("medallion_secret_key")
    
    sorted_secrets: list[str] = sorted(set(sensitive_values), key=len, reverse=True)
    for val in sorted_secrets:
        message = message.replace(val, "[REDACTED]")
        
    for pattern in SECRET_PATTERNS:
        def replace_match(match: re.Match[str]) -> str:
            full_match = match.group(0)
            if len(match.groups()) >= 2:
                secret_val = match.group(2)
                if secret_val:
                    return full_match.replace(secret_val, "[REDACTED]")
            elif len(match.groups()) == 1:
                secret_val = match.group(1)
                if secret_val:
                    return full_match.replace(secret_val, "[REDACTED]")
            return full_match
            
        message = pattern.sub(replace_match, message)
        
    return message

def format_and_redact_exception(exception_info) -> str:
    """Format the exception info and redact any sensitive parameters."""
    if not exception_info:
        return ""
    exc_type, exc_value, exc_tb = exception_info
    lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    traceback_str = "".join(lines)
    redacted = redact_secrets(traceback_str)
    # Escape any angle brackets so loguru doesn't parse them as color tags
    return redacted.replace("<", "\\<").replace(">", "\\>")

def console_formatter(record):
    redacted_message = redact_secrets(record["message"])
    escaped_message = redacted_message.replace("<", "\\<").replace(">", "\\>").replace("{", "{{").replace("}", "}}")
    
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        f"<level>{escaped_message}</level>\n"
    )
    
    if record["exception"]:
        redacted_tb = format_and_redact_exception(record["exception"])
        escaped_tb = redacted_tb.replace("{", "{{").replace("}", "}}")
        fmt += f"<red>{escaped_tb}</red>"
        
    return fmt

def file_formatter(record):
    redacted_message = redact_secrets(record["message"])
    escaped_message = redacted_message.replace("<", "\\<").replace(">", "\\>").replace("{", "{{").replace("}", "}}")
    
    fmt = f"{{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{module}}:{{function}}:{{line}} | {escaped_message}\n"
    
    if record["exception"]:
        redacted_tb = format_and_redact_exception(record["exception"])
        escaped_tb = redacted_tb.replace("{", "{{").replace("}", "}}")
        fmt += escaped_tb
        
    return fmt

def trade_formatter(record):
    redacted_message = redact_secrets(record["message"])
    escaped_message = redacted_message.replace("<", "\\<").replace(">", "\\>").replace("{", "{{").replace("}", "}}")
    
    return f"{{time:YYYY-MM-DD HH:mm:ss.SSS}} | {escaped_message}\n"

def setup_logger(
    level: str = "INFO",
    log_file: str = "logs/medallion.log",
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure the global logger.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file relative to project root.
        rotation: When to rotate (e.g., "100 MB", "1 day").
        retention: How long to keep old logs.
    """
    # Remove default handler
    logger.remove()

    # Console handler (colorized, human-readable)
    logger.add(
        sys.stderr,
        level=level,
        format=console_formatter,
        colorize=True,
    )

    # File handler (structured, for post-mortem analysis)
    log_path = PROJECT_ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level="DEBUG",  # File always captures everything
        format=file_formatter,
        rotation=rotation,
        retention=retention,
        compression="gz",
        enqueue=True,  # Thread-safe
    )

    # Trade-specific log (separate file for trade audit trail)
    trade_log_path = PROJECT_ROOT / "logs" / "trades.log"
    logger.add(
        str(trade_log_path),
        level="INFO",
        format=trade_formatter,
        filter=lambda record: "trade" in record["extra"],
        rotation="1 day",
        retention="1 year",
    )

    logger.info(f"Logger initialized | level={level} | file={log_path}")

    # Silence noisy standard library loggers
    import logging
    logging.getLogger("tvDatafeed").setLevel(logging.CRITICAL)
    logging.getLogger("tvDatafeed.main").setLevel(logging.CRITICAL)
    logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
    logging.getLogger("hmmlearn.base").setLevel(logging.CRITICAL)
    logging.getLogger("hmmlearn.hmm").setLevel(logging.CRITICAL)

def get_trade_logger():
    """Get a logger specifically for trade events."""
    return logger.bind(trade=True)

def get_risk_logger():
    """Get a logger specifically for risk events."""
    return logger.bind(risk=True)
