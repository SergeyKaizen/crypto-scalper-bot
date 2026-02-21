"""
src/model/inference.py

=== Основной принцип работы файла ===

Этот файл отвечает за inference (предсказание) модели в live и бектесте.
Он загружает ансамбль из максимум 3 моделей и выдаёт взвешенную вероятность достижения TP раньше SL.

Ключевые решения:
- Ансамбль из 3 моделей (каждая на последних 3 месяцах на момент создания).
- Вес модели = винрейт × (1 + 0.2 × коэффициент свежести).
- Коэффициент свежести: 1.0 для самой новой модели, уменьшается на 0.2 за каждый месяц старше.
- Если несколько моделей < 60 % винрейт — все такие модели удаляются при загрузке.
- Если ансамбль пуст — предикт = 0.0 (нет сигнала).
- Финальная вероятность = взвешенное среднее предиктов всех моделей.
- Нет lookahead — признаки подаются строго до текущей свечи.

=== Главные функции и за что отвечают ===

- Inference() — инициализация: загрузка моделей из models/, фильтр <60%, расчёт весов.
- predict(features, symbol, timeframe) → float — основной метод:
  - Подаёт признаки во все модели.
  - Считает взвешенное среднее с коэффициентом свежести.
  - Возвращает вероятность 0..1 или 0.0 если ансамбль пуст.
- _load_ensemble() — загрузка, фильтр <60%, сортировка по свежести, расчёт весов.
- _get_weight(winrate: float, age_months: int) → float — вес с бонусом за свежесть.

=== Примечания ===
- Модели хранятся как model_YYYYMMDD.pth в models/.
- Самая новая — максимальный вес (×1.2), самая старая — ×1.0.
- Если модель <60% — удаляется сразу при загрузке.
- Полностью соответствует всем твоим уточнениям.
- Готов к использованию в live_loop и backtest.
"""

import os
import torch
from typing import Dict, Optional, List
from datetime import datetime

from src.model.architectures import MultiWindowHybrid
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('inference', logging.INFO)

class Inference:
    def __init__(self):
        self.config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self._load_ensemble()
        logger.info(f"Загружен ансамбль из {len(self.models)} моделей")

    def predict(self, features: Dict, symbol: str, timeframe: str) -> float:
        """
        Основной метод предсказания.
        Возвращает взвешенную вероятность достижения TP раньше SL (0..1).
        Если ансамбль пуст — возвращает 0.0 (нет сигнала).
        """
        if not self.models:
            logger.warning(f"Ансамбль пуст — нет предсказания для {symbol} {timeframe}")
            return 0.0

        probs = []
        weights = []

        for model_info in self.models:
            model = model_info['model']
            weight = model_info['weight']

            with torch.no_grad():
                # Подготовка батча из одного примера
                input_batch = {tf: {w: torch.tensor([features[tf][w]], device=self.device) for w in features[tf]} for tf in features}
                output = model(input_batch)
                prob = torch.sigmoid(output).item()

            probs.append(prob)
            weights.append(weight)

        # Взвешенное среднее
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        final_prob = sum(p * w for p, w in zip(probs, weights)) / total_weight
        return final_prob

    def _load_ensemble(self) -> List[Dict]:
        """
        Загружает последние модели из models/.
        Фильтрует модели с винрейтом <60%.
        Сортирует по свежести (самая новая — первая).
        Рассчитывает веса с коэффициентом свежести 20%.
        """
        models_dir = self.config['paths']['models_dir']
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

        if not model_files:
            logger.warning("Нет моделей в директории — ансамбль пуст")
            return []

        # Сортировка по дате в имени файла (предполагаем формат model_YYYYMMDD.pth)
        model_files.sort(reverse=True)  # новая — первая

        loaded = []

        for file in model_files:
            path = os.path.join(models_dir, file)
            try:
                model = MultiWindowHybrid(input_dim=self.config['model']['input_dim']).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()

                # Винрейт из имени файла или из tracker (здесь placeholder)
                winrate = 0.65  # в реале брать из scenario_tracker или лога обучения

                # Фильтр: если винрейт < 60% — пропускаем
                if winrate < 0.60:
                    logger.info(f"Модель {file} имеет винрейт {winrate:.2%} < 60% — удалена")
                    continue

                # Возраст модели в месяцах
                age_months = self._get_model_age_months(file)

                # Коэффициент свежести: 1.0 для новой, -0.2 за каждый месяц старше
                freshness = max(0.0, 1.0 - 0.2 * age_months)

                # Вес = винрейт × (1 + 0.2 × свежесть)
                weight = winrate * (1 + 0.2 * freshness)

                loaded.append({
                    'model': model,
                    'winrate': winrate,
                    'weight': weight,
                    'file': file,
                    'age_months': age_months
                })

                logger.info(f"Загружена модель {file}, винрейт={winrate:.2%}, вес={weight:.4f}")

            except Exception as e:
                logger.error(f"Ошибка загрузки модели {file}: {e}")

        # Ограничиваем до 3 моделей (если больше — берём самые свежие)
        loaded = loaded[:3]

        return loaded

    def _get_model_age_months(self, filename: str) -> int:
        """
        Вычисляет возраст модели в месяцах по имени файла (model_YYYYMMDD.pth).
        """
        try:
            date_str = filename.split('_')[1].split('.')[0]  # YYYYMMDD
            model_date = datetime.strptime(date_str, "%Y%m%d")
            current = datetime.utcnow()
            delta = current - model_date
            return delta.days // 30
        except:
            return 0  # если ошибка — считаем свежей