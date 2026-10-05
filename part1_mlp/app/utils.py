"""Вспомогательные функции: разбор ввода пользователя и график.

Функции разбора не спрашивают значение повторно, а выбрасывают
собственное исключение. Меню перехватывает его и выводит
понятное сообщение, после чего пользователь пробует снова.
"""
import csv
from pathlib import Path

import numpy as np

from .exceptions import (DatasetFileError, InvalidInputError,
                         InvalidLayerSizeError)


def parse_int(text, field_name, min_value=1):
    """Преобразует строку в целое число не меньше min_value.

    Args:
        text: строка, введённая пользователем.
        field_name: название поля для сообщения об ошибке.
        min_value: минимально допустимое значение.

    Returns:
        int: полученное число.

    Raises:
        InvalidInputError: если строка не число или число мало.
    """
    try:
        value = int(text)
    except ValueError as error:
        raise InvalidInputError(
            f"{field_name}: ожидалось целое число, получено '{text}'"
        ) from error
    if value < min_value:
        raise InvalidInputError(
            f"{field_name}: значение должно быть не меньше {min_value}"
        )
    return value


def parse_float(text, field_name, min_value=None, max_value=None):
    """Преобразует строку в дробное число в границах (min, max).

    Границы не включаются: значение должно быть строго больше
    min_value и строго меньше max_value.

    Args:
        text: строка, введённая пользователем.
        field_name: название поля для сообщения об ошибке.
        min_value: нижняя граница или None.
        max_value: верхняя граница или None.

    Returns:
        float: полученное число.

    Raises:
        InvalidInputError: если строка не число или вне границ.
    """
    try:
        value = float(text.replace(",", "."))
    except ValueError as error:
        raise InvalidInputError(
            f"{field_name}: ожидалось число, получено '{text}'"
        ) from error
    if min_value is not None and value <= min_value:
        raise InvalidInputError(
            f"{field_name}: значение должно быть больше {min_value}"
        )
    if max_value is not None and value >= max_value:
        raise InvalidInputError(
            f"{field_name}: значение должно быть меньше {max_value}"
        )
    return value


def parse_layer_sizes(text):
    """Разбирает строку с размерами слоёв, например '4 8 1'.

    Args:
        text: числа через пробел или запятую.

    Returns:
        list[int]: размеры слоёв от входного к выходному.

    Raises:
        InvalidLayerSizeError: если слоёв меньше двух или
            размер слоя не является положительным целым числом.
    """
    parts = text.replace(",", " ").split()
    if len(parts) < 2:
        raise InvalidLayerSizeError(
            "нужно минимум два числа: входной и выходной слой"
        )
    sizes = []
    for part in parts:
        if not part.isdigit() or int(part) == 0:
            raise InvalidLayerSizeError(
                f"размер слоя '{part}' должен быть целым числом больше 0"
            )
        sizes.append(int(part))
    return sizes


def parse_vector(text):
    """Разбирает строку с признаками одного примера.

    Args:
        text: числа через пробел или точку с запятой,
            например '5.5 80 7 4'.

    Returns:
        numpy.ndarray: массив формы (1, количество_признаков).

    Raises:
        InvalidInputError: если среди значений есть не числа.
    """
    parts = text.replace(";", " ").split()
    if not parts:
        raise InvalidInputError("не введено ни одного значения")
    try:
        values = [float(part.replace(",", ".")) for part in parts]
    except ValueError as error:
        raise InvalidInputError(
            f"все значения должны быть числами: '{text}'"
        ) from error
    return np.array([values])


def read_feature_file(path):
    """Читает CSV-файл с признаками для предсказания.

    Первая строка файла считается заголовком и пропускается.
    Все остальные столбцы считаются признаками.

    Args:
        path: путь к CSV-файлу.

    Returns:
        numpy.ndarray: матрица формы (примеры, признаки).

    Raises:
        DatasetFileError: если файл не найден или содержит
            нечисловые значения.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetFileError(f"файл не найден: {file_path}")
    rows = []
    with file_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                rows.append([float(value) for value in row])
            except ValueError as error:
                raise DatasetFileError(
                    f"строка {line_number}: нечисловое значение"
                ) from error
    if not rows:
        raise DatasetFileError(f"в файле нет данных: {file_path}")
    return np.array(rows)


def plot_loss(loss_history, output_path, show=True):
    """Строит график ошибки по эпохам и сохраняет его в PNG.

    Args:
        loss_history: список значений ошибки по эпохам.
        output_path: путь для сохранения картинки.
        show: открыть ли окно с графиком.

    Returns:
        pathlib.Path: путь к сохранённому файлу.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(loss_history) + 1)
    axis.plot(epochs, loss_history, color="#1f77b4", linewidth=2)
    axis.set_title("Ошибка (MSE) на обучающей выборке")
    axis.set_xlabel("Эпоха")
    axis.set_ylabel("MSE")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    if show:
        plt.show()
    plt.close(figure)
    return output_path
