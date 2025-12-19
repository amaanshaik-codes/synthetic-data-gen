"""
Operational logs data generator.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthdata.config import (
    BusinessContextConfig,
    BusinessSizeConfig,
    DataQualityConfig,
    Industry,
    Difficulty,
)
from synthdata.generators.base import BaseGenerator


class OperationalLogGenerator(BaseGenerator):
    """Generator for operational/system logs data."""
    
    def __init__(
        self,
        business_context: BusinessContextConfig,
        business_size: BusinessSizeConfig,
        data_quality: DataQualityConfig,
        seed: Optional[int] = None,
        difficulty: Difficulty = Difficulty.MEDIUM,
    ):
        super().__init__(business_context, business_size, data_quality, seed)
        
        self.difficulty = difficulty
        self.event_types = self._get_event_types()
        self.services = self._get_services()
        self.log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def _get_event_types(self) -> Dict[str, List[str]]:
        """Get event types based on industry."""
        common_events = {
            "System": [
                "startup", "shutdown", "restart", "config_change",
                "health_check", "heartbeat", "maintenance"
            ],
            "Security": [
                "login", "logout", "login_failed", "password_change",
                "permission_change", "access_denied", "suspicious_activity"
            ],
            "API": [
                "request", "response", "timeout", "rate_limit",
                "authentication", "validation_error"
            ],
            "Database": [
                "query", "insert", "update", "delete", "connection",
                "pool_exhausted", "slow_query", "deadlock"
            ],
            "Application": [
                "user_action", "page_view", "feature_used",
                "error", "exception", "crash"
            ],
        }
        
        industry_events = {
            Industry.ECOMMERCE: {
                "Commerce": [
                    "cart_add", "cart_remove", "checkout_start", "checkout_complete",
                    "payment_process", "payment_success", "payment_failed",
                    "order_created", "order_updated", "order_shipped"
                ],
            },
            Industry.FINTECH: {
                "Transaction": [
                    "transfer_initiated", "transfer_completed", "transfer_failed",
                    "fraud_check", "fraud_alert", "kyc_verification",
                    "balance_update", "limit_exceeded"
                ],
            },
            Industry.SAAS: {
                "Subscription": [
                    "trial_start", "trial_end", "subscription_created",
                    "subscription_upgraded", "subscription_cancelled",
                    "usage_limit_warning", "usage_limit_exceeded"
                ],
            },
        }
        
        events = common_events.copy()
        events.update(industry_events.get(self.business_context.industry, {}))
        
        return events
    
    def _get_services(self) -> List[str]:
        """Get service names."""
        return [
            "api-gateway", "auth-service", "user-service", "payment-service",
            "order-service", "notification-service", "analytics-service",
            "search-service", "recommendation-engine", "cdn", "database",
            "cache-service", "queue-worker", "scheduler", "webhook-handler"
        ]
    
    def _generate_log_message(
        self,
        event_type: str,
        event: str,
        log_level: str,
        service: str,
    ) -> str:
        """Generate a log message."""
        templates = {
            "INFO": [
                f"[{service}] {event} completed successfully",
                f"[{service}] Processing {event} request",
                f"[{service}] {event_type}/{event} - OK",
            ],
            "WARNING": [
                f"[{service}] {event} took longer than expected",
                f"[{service}] Retrying {event} (attempt 2/3)",
                f"[{service}] High latency detected for {event}",
            ],
            "ERROR": [
                f"[{service}] {event} failed: Connection timeout",
                f"[{service}] Error processing {event}: Invalid data",
                f"[{service}] {event_type}/{event} - FAILED",
            ],
            "CRITICAL": [
                f"[{service}] CRITICAL: {event} service unavailable",
                f"[{service}] FATAL ERROR in {event}",
                f"[{service}] System failure during {event}",
            ],
            "DEBUG": [
                f"[{service}] {event} called with params: {{...}}",
                f"[{service}] Entering {event} handler",
                f"[{service}] {event} response: {{status: 200}}",
            ],
        }
        
        message_list = templates.get(log_level, templates["INFO"])
        return random.choice(message_list)
    
    def _generate_metadata(
        self,
        event: str,
        log_level: str,
    ) -> Dict[str, Any]:
        """Generate event metadata."""
        metadata = {
            "request_id": self.generate_id("req_", 16).lower(),
            "trace_id": self.generate_id("trace_", 24).lower(),
            "duration_ms": random.randint(1, 5000) if log_level != "DEBUG" else random.randint(1, 100),
        }
        
        if log_level in ["ERROR", "CRITICAL"]:
            metadata["error_code"] = f"E{random.randint(1000, 9999)}"
            metadata["stack_trace"] = "at Function.process (line:42)" if random.random() > 0.5 else None
        
        if event in ["login", "logout", "login_failed"]:
            metadata["user_id"] = self.generate_id("usr_", 8).lower()
            metadata["ip_address"] = self.faker.ipv4()
        
        if event in ["query", "slow_query"]:
            metadata["query_type"] = random.choice(["SELECT", "INSERT", "UPDATE", "DELETE"])
            metadata["table"] = random.choice(["users", "orders", "products", "transactions"])
            metadata["rows_affected"] = random.randint(0, 10000)
        
        return metadata
    
    def _inject_chaos(self, logs: List[Dict]) -> List[Dict]:
        """Inject chaos for chaotic difficulty level."""
        if self.difficulty != Difficulty.CHAOTIC:
            return logs
        
        chaotic_logs = []
        
        for log in logs:
            # Randomly corrupt some entries
            if random.random() < 0.05:
                # Truncated log
                log["message"] = log["message"][:random.randint(10, 30)] + "..."
            
            if random.random() < 0.03:
                # Wrong timestamp format
                log["timestamp"] = log["timestamp"].strftime("%d/%m/%Y %H:%M")
            
            if random.random() < 0.02:
                # Null values in unexpected places
                log["service"] = None
            
            if random.random() < 0.04:
                # Duplicate with slight variation
                chaotic_logs.append(log.copy())
            
            if random.random() < 0.02:
                # Invalid JSON in metadata
                log["metadata"] = "{{invalid json}}"
            
            chaotic_logs.append(log)
        
        return chaotic_logs
    
    def generate(
        self,
        num_rows: Optional[int] = None,
        user_ids: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Generate operational log data."""
        if num_rows is None:
            # Estimate logs based on business size
            logs_per_day = self.business_size.daily_transactions * 10  # ~10 logs per transaction
            total_days = self.business_context.time_span_months * 30
            num_rows = min(logs_per_day * total_days, 1000000)  # Cap at 1M for performance
        
        logs = []
        
        for i in range(num_rows):
            log_id = self.generate_id("log_", 12).lower()
            
            # Timestamp with realistic distribution (more during business hours)
            timestamp = self.random_date()
            hour = random.choices(
                range(24),
                weights=[1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 10, 10, 10, 10, 10, 8, 6, 5, 4, 3, 2, 2, 1, 1],
            )[0]
            timestamp = timestamp.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
                microsecond=random.randint(0, 999999),
            )
            
            # Event type and event
            event_type = random.choice(list(self.event_types.keys()))
            event = random.choice(self.event_types[event_type])
            
            # Log level (weighted towards INFO)
            level_weights = [0.05, 0.70, 0.15, 0.08, 0.02]
            log_level = self.weighted_choice(self.log_levels, level_weights)
            
            # Service
            service = random.choice(self.services)
            
            # User ID if provided
            user_id = random.choice(user_ids) if user_ids and random.random() > 0.3 else None
            
            # Generate metadata
            metadata = self._generate_metadata(event, log_level)
            
            log_entry = {
                "log_id": log_id,
                "timestamp": timestamp,
                "log_level": log_level,
                "service": service,
                "event_type": event_type,
                "event": event,
                "message": self._generate_log_message(event_type, event, log_level, service),
                "user_id": user_id,
                "session_id": self.generate_id("ses_", 10).lower() if user_id else None,
                "request_id": metadata.get("request_id"),
                "trace_id": metadata.get("trace_id"),
                "duration_ms": metadata.get("duration_ms"),
                "status_code": self._generate_status_code(log_level),
                "ip_address": metadata.get("ip_address", self.faker.ipv4() if random.random() > 0.7 else None),
                "user_agent": self._generate_user_agent() if random.random() > 0.5 else None,
                "environment": random.choice(["production", "production", "production", "staging", "development"]),
                "version": f"v{random.randint(1, 5)}.{random.randint(0, 20)}.{random.randint(0, 100)}",
                "metadata": str(metadata) if random.random() > 0.3 else None,
            }
            
            logs.append(log_entry)
        
        # Apply chaos if needed
        logs = self._inject_chaos(logs)
        
        df = pd.DataFrame(logs)
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df
    
    def _generate_status_code(self, log_level: str) -> int:
        """Generate HTTP status code based on log level."""
        if log_level == "INFO":
            return random.choice([200, 201, 204])
        elif log_level == "WARNING":
            return random.choice([301, 302, 304, 400, 401])
        elif log_level == "ERROR":
            return random.choice([400, 401, 403, 404, 422, 429])
        elif log_level == "CRITICAL":
            return random.choice([500, 502, 503, 504])
        else:
            return 200
    
    def _generate_user_agent(self) -> str:
        """Generate a user agent string."""
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Mobile/15E148",
            "Mozilla/5.0 (Linux; Android 14) Chrome/120.0.0.0 Mobile",
            "api-client/2.0",
            "PostmanRuntime/7.35.0",
            "python-requests/2.31.0",
        ]
        return random.choice(agents)
    
    def get_schema(self) -> Dict[str, str]:
        """Get the schema for operational log data."""
        return {
            "log_id": "string",
            "timestamp": "datetime",
            "log_level": "category",
            "service": "category",
            "event_type": "category",
            "event": "category",
            "message": "text",
            "user_id": "string",
            "session_id": "string",
            "request_id": "string",
            "trace_id": "string",
            "duration_ms": "integer",
            "status_code": "integer",
            "ip_address": "string",
            "user_agent": "string",
            "environment": "category",
            "version": "string",
            "metadata": "json",
        }
    
    def get_column_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each column."""
        return {
            "log_id": "Unique identifier for the log entry",
            "timestamp": "When the event occurred",
            "log_level": "Severity level of the log",
            "service": "Service that generated the log",
            "event_type": "Category of the event",
            "event": "Specific event name",
            "message": "Human-readable log message",
            "user_id": "Associated user ID (if applicable)",
            "session_id": "User session identifier",
            "request_id": "Request identifier for tracing",
            "trace_id": "Distributed trace identifier",
            "duration_ms": "Duration of the operation in milliseconds",
            "status_code": "HTTP status code (if applicable)",
            "ip_address": "Client IP address",
            "user_agent": "Client user agent string",
            "environment": "Deployment environment",
            "version": "Application version",
            "metadata": "Additional metadata as JSON",
        }
