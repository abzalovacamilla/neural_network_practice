"""Собственные классы исключений для части 1.

Все исключения наследуются от общего базового класса
NeuralNetworkError. Благодаря этому в меню можно перехватить
любую ошибку приложения одним блоком except и показать
пользователю понятное сообщение.
"""


class NeuralNetworkError(Exception):
    """Базовое исключение для всех ошибок приложения."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Размеры слоёв сети заданы неверно."""


class InvalidActivationError(NeuralNetworkError):
    """Указана неизвестная функция активации."""


class MismatchedDataError(NeuralNetworkError):
    """Размерность данных не совпадает с архитектурой сети."""


class DatasetFileError(NeuralNetworkError):
    """CSV-файл не найден или имеет неверный формат."""


class DataNotLoadedError(NeuralNetworkError):
    """Действие требует загруженных данных, а их нет."""


class NetworkNotCreatedError(NeuralNetworkError):
    """Действие требует созданной сети, а её нет."""


class WeightsFileError(NeuralNetworkError):
    """Файл весов не найден или повреждён."""


class InvalidInputError(NeuralNetworkError):
    """Пользователь ввёл некорректное значение."""
