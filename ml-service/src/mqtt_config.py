"""
mqtt_config.py - Environment configuration and TLS client factory for HiveMQ Cloud.

Phase 1: Telemetry Simulator & MQTT Infrastructure
Security Rule: NEVER hardcode credentials. NEVER print secrets to logs.
"""

import os
import ssl
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import paho.mqtt.client as mqtt

logger = logging.getLogger("MachineMind.MQTTConfig")

# Automatically search for .env in project root or backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv_paths = [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "backend" / ".env",
]
for p in dotenv_paths:
    if p.exists():
        load_dotenv(dotenv_path=p, override=False)


class MQTTConfig:
    """
    Centralized configuration container for HiveMQ Cloud MQTT connection.
    Loads settings from environment variables safely with placeholders.
    """

    def __init__(self, env_file: Optional[Path] = None):
        if env_file and env_file.exists():
            load_dotenv(dotenv_path=env_file, override=True)

        self.broker_host: str = os.getenv("MQTT_BROKER_HOST", "").strip()
        self.broker_port: int = int(os.getenv("MQTT_BROKER_PORT", "8883"))
        self.username: str = os.getenv("MQTT_USERNAME", "").strip()
        self.password: str = os.getenv("MQTT_PASSWORD", "").strip()
        self.topic_prefix: str = os.getenv("MQTT_TOPIC_PREFIX", "machinemind/v1").strip("/")
        self.machine_id: str = os.getenv("MQTT_MACHINE_ID", "ims_set2_rig").strip()
        self.qos: int = int(os.getenv("MQTT_QOS", "1"))
        
        use_tls_str = os.getenv("MQTT_USE_TLS", "true").lower()
        self.use_tls: bool = use_tls_str in ("1", "true", "yes", "on")

    def is_configured(self) -> bool:
        """Checks if minimum required connection parameters are set and non-placeholder."""
        is_valid_host = bool(self.broker_host and not self.broker_host.startswith("your-"))
        is_valid_user = bool(self.username and not self.username.startswith("your_"))
        return is_valid_host and is_valid_user and bool(self.password)

    def get_masked_summary(self) -> Dict[str, Any]:
        """Returns safe connection summary with masked credentials for logging."""
        masked_user = f"{self.username[:2]}***" if len(self.username) > 2 else "***"
        return {
            "broker_host": self.broker_host or "<NOT_CONFIGURED>",
            "broker_port": self.broker_port,
            "username": masked_user,
            "use_tls": self.use_tls,
            "topic_prefix": self.topic_prefix,
            "machine_id": self.machine_id,
            "qos": self.qos,
            "configured": self.is_configured(),
        }

    def get_telemetry_topic(self, machine_id: Optional[str] = None) -> str:
        """Constructs telemetry topic: {topic_prefix}/telemetry/{machine_id}."""
        target_m_id = machine_id or self.machine_id
        return f"{self.topic_prefix}/telemetry/{target_m_id}"

    def get_status_topic(self, machine_id: Optional[str] = None) -> str:
        """Constructs status topic: {topic_prefix}/status/{machine_id}."""
        target_m_id = machine_id or self.machine_id
        return f"{self.topic_prefix}/status/{target_m_id}"


def create_mqtt_client(
    config: MQTTConfig,
    client_id: str,
    clean_session: bool = True
) -> mqtt.Client:
    """
    Creates and returns a configured paho-mqtt Client with HiveMQ Cloud TLS and auth settings.

    Parameters
    ----------
    config : MQTTConfig
        Configuration object containing broker endpoint and auth credentials.
    client_id : str
        Unique client identifier for MQTT session.
    clean_session : bool, default True
        If True, broker clears session state on disconnect.

    Returns
    -------
    mqtt.Client
        Configured Paho MQTT client instance ready for connect().
    """
    # Initialize Paho MQTT client using CallbackAPIVersion.VERSION2 if available
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=clean_session,
        )
    except AttributeError:
        # Fallback for older paho-mqtt releases
        client = mqtt.Client(
            client_id=client_id,
            clean_session=clean_session,
        )

    # Configure username/password authentication if provided
    if config.username and config.password:
        client.username_pw_set(config.username, config.password)

    # Configure SSL/TLS for HiveMQ Cloud
    if config.use_tls:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = True
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED
        client.tls_set_context(ssl_ctx)

    # Set default reconnect backoff bounds (min 1 sec, max 120 sec)
    client.reconnect_delay_set(min_delay=1, max_delay=120)

    return client
