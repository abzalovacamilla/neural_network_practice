"""Класс DatasetManager: загрузка, разбиение и нормализация данных."""
import csv
from pathlib import Path

import numpy as np

from .exceptions import (DataNotLoadedError, DatasetFileError,
                         InvalidInputError, MismatchedDataError)

NORMALIZATION_METHODS = ("standard", "minmax")


class DatasetManager:
    """Готовит данные из CSV-файла для обучения сети.

    Формат файла: первая строка — заголовки, последний столбец —
    целевая переменная (ответ), остальные столбцы — признаки.
    """

    def __init__(self):
        """Создаёт пустой менеджер без данных."""
        self._reset()

    def _reset(self):
        """Очищает все загруженные данные."""
        self.source_path = None
        self.feature_names = []
        self.target_name = ""
        self.features = None
        self.targets = None
        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None
        self.normalization = None
        self._shift = None
        self._scale = None

    @property
    def is_loaded(self):
        """True, если данные уже загружены."""
        return self.features is not None

    @property
    def is_split(self):
        """True, если выборка уже разбита на train/test."""
        return self.train_features is not None

    @property
    def feature_count(self):
        """Количество признаков (столбцов без целевого)."""
        return len(self.feature_names)

    def load_csv(self, path):
        """Загружает данные из CSV-файла.

        Args:
            path: путь к CSV-файлу.

        Returns:
            int: количество загруженных строк.

        Raises:
            DatasetFileError: если файла нет, он пустой или
                содержит нечисловые значения.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise DatasetFileError(f"файл не найден: {file_path}")
        with file_path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if not header or len(header) < 2:
                raise DatasetFileError(
                    "в файле должна быть строка заголовков "
                    "и минимум два столбца"
                )
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != len(header):
                    raise DatasetFileError(
                        f"строка {line_number}: ожидалось "
                        f"{len(header)} значений, найдено {len(row)}"
                    )
                try:
                    rows.append([float(value) for value in row])
                except ValueError as error:
                    raise DatasetFileError(
                        f"строка {line_number}: нечисловое значение"
                    ) from error
        if len(rows) < 2:
            raise DatasetFileError("в файле меньше двух строк данных")
        table = np.array(rows)
        self._reset()
        self.source_path = file_path
        self.feature_names = header[:-1]
        self.target_name = header[-1]
        self.features = table[:, :-1]
        self.targets = table[:, -1]
        return table.shape[0]

    def split(self, test_ratio=0.2, seed=None):
        """Перемешивает примеры и делит их на train и test.

        Args:
            test_ratio: доля тестовой выборки (от 0 до 1).
            seed: зерно генератора для воспроизводимости.

        Returns:
            tuple[int, int]: размеры обучающей и тестовой выборок.

        Raises:
            DataNotLoadedError: если данные ещё не загружены.
            InvalidInputError: если доля вне диапазона (0, 1).
        """
        if not self.is_loaded:
            raise DataNotLoadedError("сначала загрузите CSV-файл")
        if not 0 < test_ratio < 1:
            raise InvalidInputError("доля теста должна быть от 0 до 1")
        sample_count = self.features.shape[0]
        test_count = max(1, int(round(sample_count * test_ratio)))
        test_count = min(test_count, sample_count - 1)
        order = np.random.default_rng(seed).permutation(sample_count)
        test_index = order[:test_count]
        train_index = order[test_count:]
        self.train_features = self.features[train_index]
        self.train_targets = self.targets[train_index]
        self.test_features = self.features[test_index]
        self.test_targets = self.targets[test_index]
        self.normalization = None
        return len(train_index), len(test_index)

    def normalize(self, method="standard"):
        """Нормализует признаки обучающей и тестовой выборок.

        Параметры (среднее и разброс либо минимум и максимум)
        считаются только по обучающей выборке, чтобы сеть
        «не подглядывала» в тест.

        standard: x' = (x - среднее) / стандартное_отклонение
        minmax:   x' = (x - минимум) / (максимум - минимум)

        Args:
            method: 'standard' или 'minmax'.

        Raises:
            DataNotLoadedError: если выборка ещё не разбита.
            InvalidInputError: если метод неизвестен.
        """
        if not self.is_split:
            raise DataNotLoadedError("сначала разбейте выборку")
        if method not in NORMALIZATION_METHODS:
            raise InvalidInputError(
                f"метод должен быть одним из: "
                f"{', '.join(NORMALIZATION_METHODS)}"
            )
        if self.normalization is not None:
            return
        if method == "standard":
            shift = self.train_features.mean(axis=0)
            scale = self.train_features.std(axis=0)
        else:
            shift = self.train_features.min(axis=0)
            scale = self.train_features.max(axis=0) - shift
        scale[scale == 0] = 1.0
        self._shift = shift
        self._scale = scale
        self.normalization = method
        self.train_features = (self.train_features - shift) / scale
        self.test_features = (self.test_features - shift) / scale

    def transform(self, features):
        """Нормализует новые примеры теми же параметрами.

        Нужен для предсказания: признаки, введённые вручную,
        должны пройти то же преобразование, что и обучающие.

        Args:
            features: матрица признаков (примеры, признаки).

        Returns:
            numpy.ndarray: нормализованная матрица.

        Raises:
            MismatchedDataError: если число признаков не совпадает.
        """
        features = np.atleast_2d(np.asarray(features, dtype=float))
        if self.is_loaded and features.shape[1] != self.feature_count:
            raise MismatchedDataError(
                f"ожидалось {self.feature_count} признака(ов), "
                f"получено {features.shape[1]}"
            )
        if self.normalization is None:
            return features
        return (features - self._shift) / self._scale

    def summary(self):
        """Возвращает строку с краткой информацией о данных."""
        if not self.is_loaded:
            return "данные не загружены"
        classes, counts = np.unique(self.targets, return_counts=True)
        balance = ", ".join(
            f"{label:g}: {count}" for label, count in zip(classes, counts)
        )
        return (
            f"файл: {self.source_path}\n"
            f"  примеров: {self.features.shape[0]}, "
            f"признаков: {self.feature_count}\n"
            f"  признаки: {', '.join(self.feature_names)}\n"
            f"  целевая переменная: {self.target_name} ({balance})"
        )
