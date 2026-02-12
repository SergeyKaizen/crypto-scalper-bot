# src/core/config.py
"""
Файл загрузки и валидации конфигурации проекта.

Логика:
1. Загружает default.yaml — базовые настройки
2. Добавляет hardware-профиль (phone_tiny / colab / server)
3. Добавляет trading_mode (conservative / balanced / aggressive / custom)
4. Валидирует ключевые параметры (max_coins > 0, seq_len >= 20 и т.д.)
5. Автоматически определяет окружение, если --hardware не указан
6. Возвращает единый dict config, который используют все модули

Использование:
    from src.core.config import load_config
    config = load_config(hardware_profile="phone_tiny", trading_mode="balanced")
"""

import yaml
from pathlib import Path
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(
    hardware_profile: Optional[str] = None,
    trading_mode: str = "balanced"
) -> Dict[str, Any]:
    """
    Основная функция загрузки конфига.
    
    Args:
        hardware_profile: "phone_tiny" / "colab" / "server" (или None — автоопределение)
        trading_mode: "conservative" / "balanced" / "aggressive" / "custom"
    
    Returns:
        Dict со всеми настройками (merged)
    """
    base_path = Path(__file__).parent.parent.parent / "config"

    # 1. Загружаем базовый конфиг
    default_path = base_path / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Определяем hardware-профиль (если не указан — автоопределение)
    if hardware_profile is None:
        hardware_profile = _detect_hardware()
        logger.info("Hardware profile auto-detected: %s", hardware_profile)

    hw_path = base_path / "hardware" / f"{hardware_profile}.yaml"
    if not hw_path.exists():
        logger.warning("Hardware profile not found: %s. Using default settings.", hw_path)
    else:
        with open(hw_path, "r", encoding="utf-8") as f:
            hw_config = yaml.safe_load(f)
        _deep_update(config, hw_config)
        logger.info("Loaded hardware profile: %s", hardware_profile)

    # 3. Загружаем trading mode
    mode_path = base_path / "trading_modes" / f"{trading_mode}.yaml"
    if not mode_path.exists():
        logger.warning("Trading mode not found: %s. Using default trading settings.", mode_path)
    else:
        with open(mode_path, "r", encoding="utf-8") as f:
            mode_config = yaml.safe_load(f)
        _deep_update(config, mode_config)
        logger.info("Loaded trading mode: %s", trading_mode)

    # 4. Валидация ключевых параметров
    _validate_config(config)

    # 5. Дополнительная адаптация под телефон (если low_power_mode)
    if config.get("low_power_mode", False):
        logger.info("Low power mode activated - additional optimizations applied")
        config["quiet_mode"] = config.get("quiet_mode", False)  # можно принудительно выключить
        config["pr"]["recalc_interval_minutes"] = 10  # реже пересчёт PR

    return config


def _detect_hardware() -> str:
    """Автоматическое определение окружения (очень упрощённо)"""
    import platform
    import psutil

    system = platform.system().lower()
    cpu_count = psutil.cpu_count(logical=False)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    has_gpu = "cuda" in torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else False

    if "android" in system or "linux" in system and "arm" in platform.machine():
        return "phone_tiny"
    elif "google" in platform.node().lower() or ram_gb < 30:
        return "colab"
    else:
        return "server"


def _deep_update(target: Dict, source: Dict) -> None:
    """Глубокое слияние словарей (source переписывает target)"""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _validate_config(config: Dict) -> None:
    """Валидация ключевых параметров — чтобы бот не запустился с ошибками"""
    errors = []

    if config.get("max_coins", 0) < 1:
        errors.append("max_coins должен быть >= 1")
    if config.get("seq_len", 0) < 20:
        errors.append("seq_len должен быть >= 20 (для GRU)")
    if config.get("pr", {}).get("analysis_period_candles", 0) < 100:
        errors.append("pr.analysis_period_candles должен быть >= 100")
    if config.get("risk", {}).get("default_risk_pct", 0) <= 0:
        errors.append("default_risk_pct должен быть > 0")
    if config.get("prob_threshold", 0) < 0.5 or config.get("prob_threshold", 0) > 0.95:
        errors.append("prob_threshold должен быть в диапазоне 0.5–0.95")

    if errors:
        raise ValueError("Конфигурация невалидна:\n" + "\n".join(errors))


# Пример использования (для тестов)
if __name__ == "__main__":
    config = load_config("phone_tiny", "balanced")
    print("Loaded config keys:", list(config.keys()))
    print("max_coins:", config["max_coins"])
    print("seq_len:", config["seq_len"])
    print("prob_threshold:", config["prob_threshold"])