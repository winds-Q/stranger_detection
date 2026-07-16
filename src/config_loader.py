import copy
import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
    },
    "recognition": {
        "tolerance": 0.5,
        "model": "hog",
        "stranger_tolerance": 0.5,
        "stranger_retention_seconds": 3600,
    },
    "alert": {
        "enabled": True,
        "cooldown_seconds": 180,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "",
        "sender_password": "",
        "receiver_email": "",
    },
    "snapshot": {
        "save_to": "./snapshots/",
        "max_snapshots": 100,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = config_path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.isfile(self._config_path):
            logger.warning("配置文件 %s 不存在，使用默认配置", self._config_path)
            return copy.deepcopy(DEFAULT_CONFIG)

        with open(self._config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        merged = _deep_merge(DEFAULT_CONFIG, user_config)
        logger.info("配置文件已加载: %s", self._config_path)
        return merged

    @property
    def camera(self) -> Dict[str, Any]:
        return self._data["camera"]

    @property
    def recognition(self) -> Dict[str, Any]:
        return self._data["recognition"]

    @property
    def alert(self) -> Dict[str, Any]:
        return self._data["alert"]

    @property
    def snapshot(self) -> Dict[str, Any]:
        return self._data["snapshot"]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
