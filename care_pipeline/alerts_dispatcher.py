"""
alerts_dispatcher.py
Enterprise Webhook Alert Dispatcher.
Pushes critical care transition notifications and operational threshold breaches
to external webhooks (Slack, Discord, MS Teams, or custom local HTTP sinks)
with fail-safe, non-blocking error handling.
"""

import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {
    "INFO": 0x38BDF8,      # Sky Blue
    "OPTIMAL": 0x34D399,   # Emerald Green
    "WARNING": 0xFBBF24,   # Amber
    "STAGNATION": 0xFB923C,# Orange
    "CRITICAL": 0xEF4444   # Rose Red
}

def dispatch_webhook_alert(
    webhook_url: str,
    alert_level: str,
    message: str,
    metric_details: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 3.0
) -> Dict[str, Any]:
    """
    Dispatches a structured webhook alert to Slack, Discord, or generic HTTP sinks.
    
    Guarantees:
    - Never raises unhandled exceptions to crash the caller.
    - Times out gracefully if endpoint is unresponsive.
    - Automatically formats payload based on destination platform.
    
    Returns a status dict: {"success": bool, "status_code": int, "error": Optional[str]}
    """
    if not webhook_url or not webhook_url.strip():
        return {"success": False, "status_code": 400, "error": "Empty or missing webhook URL"}

    alert_level_upper = alert_level.upper()
    color_int = SEVERITY_COLORS.get(alert_level_upper, 0x64748B)
    
    # 1. Format Slack Payload
    # Slack accepts { "text": "...", "attachments": [...] }
    slack_payload = {
        "text": f"*{alert_level_upper} ALERT*: {message}",
        "attachments": [
            {
                "color": f"#{color_int:06x}",
                "fields": [
                    {"title": k.replace('_', ' ').title(), "value": str(v), "short": True}
                    for k, v in (metric_details or {}).items()
                ]
            }
        ]
    }
    
    # 2. Format Discord Embed Payload
    discord_payload = {
        "content": f"🚨 **UAC Pipeline Alert** [{alert_level_upper}]",
        "embeds": [
            {
                "title": "Care Transition Operational Notification",
                "description": message,
                "color": color_int,
                "fields": [
                    {"name": k.replace('_', ' ').title(), "value": str(v), "inline": True}
                    for k, v in (metric_details or {}).items()
                ]
            }
        ]
    }

    # Detect platform or send universal fallback
    url_lower = webhook_url.lower()
    if "discord.com" in url_lower:
        payload = discord_payload
    else:
        payload = slack_payload

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout_sec
        )
        success = (200 <= response.status_code < 300)
        return {
            "success": success,
            "status_code": response.status_code,
            "error": None if success else f"HTTP {response.status_code}: {response.text[:100]}"
        }
    except requests.exceptions.Timeout:
        logger.warning(f"Webhook dispatch timed out after {timeout_sec}s for URL: {webhook_url}")
        return {"success": False, "status_code": 408, "error": f"Request timed out after {timeout_sec}s"}
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to dispatch webhook alert: {e}")
        return {"success": False, "status_code": 500, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in dispatch_webhook_alert: {e}")
        return {"success": False, "status_code": 500, "error": f"Internal error: {str(e)}"}
