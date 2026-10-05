"""Многослойный перцептрон: функции активации, слой и сеть.

Архитектура построена на композиции: сеть (NeuralNetwork)
состоит из слоёв (DenseLayer), а каждый слой содержит объект
функции активации. Функции активации построены на наследовании
от общего базового класса Activation.
"""
import json
from pathlib import Path

import numpy as np

from .exceptions import (InvalidActivationError, InvalidInputError,
                         InvalidLayerSizeError, MismatchedDataError,
                         WeightsFileError)


class Activation:
    """Базовый класс функции активации.

    Наследники переопределяют два метода: forward (значение
    функции) и derivative (производная).
    """

    name = "base"

    def forward(self, values):
        """Вычисляет значение функции активации.

        Args:
            values: взвешенная сумма входов слоя (numpy-массив).

        Returns:
            numpy.ndarray: выход функции активации.
        """
        raise NotImplementedError

    def derivative(self, activated):
        """Вычисляет производную через уже посчитанный выход.

        Args:
            activated: выход функции активации (numpy-массив).

        Returns:
            numpy.ndarray: значение производной.
        """
        raise NotImplementedError


class Sigmoid(Activation):
    """Сигмоида: f(z) = 1 / (1 + e^(-z)), выход от 0 до 1."""

    name = "sigmoid"

    def forward(self, values):
        """Возвращает сигмоиду от values."""
        clipped = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))

    def derivative(self, activated):
        """Возвращает f'(z) = f(z) * (1 - f(z))."""
        return activated * (1.0 - activated)


class ReLU(Activation):
    """ReLU: f(z) = max(0, z)."""

    name = "relu"

    def forward(self, values):
        """Возвращает max(0, values)."""
        return np.maximum(0.0, values)

    def derivative(self, activated):
        """Возвращает 1 там, где выход больше нуля, иначе 0."""
        return (activated > 0).astype(float)


ACTIVATIONS = {"sigmoid": Sigmoid, "relu": ReLU}


def get_activation(name):
    """Создаёт объект функции активации по её названию.

    Args:
        name: 'sigmoid' или 'relu'.

    Returns:
        Activation: объект функции активации.

    Raises:
        InvalidActivationError: если название неизвестно.
    """
    activation_class = ACTIVATIONS.get(name.strip().lower())
    if activation_class is None:
        available = ", ".join(ACTIVATIONS)
        raise InvalidActivationError(
            f"неизвестная функция активации '{name}', "
            f"доступны: {available}"
        )
    return activation_class()


class DenseLayer:
    """Полносвязный слой: каждый вход связан с каждым нейроном."""

    def __init__(self, input_size, output_size, activation, rng):
        """Создаёт слой и заполняет веса случайными числами.

        Args:
            input_size: количество входов слоя.
            output_size: количество нейронов слоя.
            activation: объект функции активации.
            rng: генератор случайных чисел numpy.
        """
        scale = np.sqrt(1.0 / input_size)
        self.weights = rng.normal(0.0, scale, (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """Прямой проход: a = f(X · W + b).

        Args:
            inputs: матрица входов формы (примеры, входы).

        Returns:
            numpy.ndarray: выходы слоя формы (примеры, нейроны).
        """
        self.inputs = inputs
        weighted_sum = inputs @ self.weights + self.biases
        self.outputs = self.activation.forward(weighted_sum)
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        """Обратный проход и шаг градиентного спуска.

        Args:
            output_gradient: производная ошибки по выходу слоя.
            learning_rate: скорость обучения.

        Returns:
            numpy.ndarray: производная ошибки по входу слоя,
            которую получит предыдущий слой.
        """
        batch_size = self.inputs.shape[0]
        delta = output_gradient * self.activation.derivative(self.outputs)
        weights_gradient = self.inputs.T @ delta / batch_size
        biases_gradient = delta.mean(axis=0, keepdims=True)
        input_gradient = delta @ self.weights.T
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        return input_gradient


class NeuralNetwork:
    """Полносвязная нейронная сеть (многослойный перцептрон).

    Скрытые слои используют выбранную функцию активации,
    выходной слой всегда использует сигмоиду, чтобы выход
    находился в диапазоне от 0 до 1.
    """

    def __init__(self, layer_sizes, activation="sigmoid", seed=None):
        """Создаёт сеть заданной архитектуры.

        Args:
            layer_sizes: список размеров слоёв, например [4, 8, 1]:
                4 входа, 8 нейронов в скрытом слое, 1 выход.
            activation: функция активации скрытых слоёв.
            seed: зерно генератора для воспроизводимости.

        Raises:
            InvalidLayerSizeError: если слоёв меньше двух или
                размер слоя меньше единицы.
            InvalidActivationError: если активация неизвестна.
        """
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError(
                "нужно минимум два слоя: входной и выходной"
            )
        if any(int(size) < 1 for size in layer_sizes):
            raise InvalidLayerSizeError(
                "размер каждого слоя должен быть больше 0"
            )
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.activation_name = get_activation(activation).name
        self.loss_history = []
        rng = np.random.default_rng(seed)
        self.layers = []
        layer_pairs = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        last_index = len(self.layer_sizes) - 2
        for index, (inputs, outputs) in enumerate(layer_pairs):
            if index == last_index:
                layer_activation = Sigmoid()
            else:
                layer_activation = get_activation(activation)
            self.layers.append(
                DenseLayer(inputs, outputs, layer_activation, rng)
            )

    @property
    def input_size(self):
        """Количество входов сети (признаков)."""
        return self.layer_sizes[0]

    @property
    def output_size(self):
        """Количество выходов сети."""
        return self.layer_sizes[-1]

    def describe(self):
        """Возвращает краткое текстовое описание архитектуры."""
        sizes = " -> ".join(str(size) for size in self.layer_sizes)
        return f"слои: {sizes}, активация скрытых слоёв: " \
               f"{self.activation_name}"

    def set_activation(self, name):
        """Меняет функцию активации всех скрытых слоёв.

        Args:
            name: 'sigmoid' или 'relu'.
        """
        self.activation_name = get_activation(name).name
        for layer in self.layers[:-1]:
            layer.activation = get_activation(name)

    def _check_features(self, features):
        """Проверяет, что число признаков совпадает с входами сети."""
        features = np.atleast_2d(np.asarray(features, dtype=float))
        if features.shape[1] != self.input_size:
            raise MismatchedDataError(
                f"сеть ожидает {self.input_size} признака(ов), "
                f"а получено {features.shape[1]}"
            )
        return features

    def prepare_targets(self, targets):
        """Приводит ответы к форме (примеры, выходы сети).

        При одном выходе ответы используются как есть (0 или 1).
        При нескольких выходах номер класса превращается в
        вектор one-hot: класс 2 из 3 -> [0, 0, 1].

        Args:
            targets: одномерный массив ответов.

        Returns:
            numpy.ndarray: матрица ответов.

        Raises:
            MismatchedDataError: если номер класса не помещается
                в число выходов сети.
        """
        targets = np.asarray(targets, dtype=float).reshape(-1)
        if self.output_size == 1:
            return targets.reshape(-1, 1)
        labels = targets.astype(int)
        if labels.min() < 0 or labels.max() >= self.output_size:
            raise MismatchedDataError(
                f"номера классов должны быть от 0 до "
                f"{self.output_size - 1}"
            )
        one_hot = np.zeros((labels.size, self.output_size))
        one_hot[np.arange(labels.size), labels] = 1.0
        return one_hot

    def forward(self, features):
        """Прямой проход через все слои сети.

        Args:
            features: матрица признаков (примеры, признаки).

        Returns:
            numpy.ndarray: выходы сети.
        """
        signal = self._check_features(features)
        for layer in self.layers:
            signal = layer.forward(signal)
        return signal

    def backward(self, predictions, targets, learning_rate):
        """Обратное распространение ошибки от выхода ко входу.

        Ошибка E = 1/2 * сумма (y_pred - y)^2, поэтому её
        производная по выходу сети равна (y_pred - y).

        Args:
            predictions: выходы сети после прямого прохода.
            targets: правильные ответы той же формы.
            learning_rate: скорость обучения.
        """
        gradient = predictions - targets
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    @staticmethod
    def mse(predictions, targets):
        """Среднеквадратичная ошибка (MSE)."""
        return float(np.mean((predictions - targets) ** 2))

    def train(self, features, targets, learning_rate=0.1, epochs=100,
              batch_size=16, on_epoch_end=None, seed=None):
        """Обучает сеть мини-батчевым градиентным спуском.

        Args:
            features: матрица признаков обучающей выборки.
            targets: одномерный массив правильных ответов.
            learning_rate: скорость обучения (больше 0).
            epochs: количество эпох (проходов по выборке).
            batch_size: размер батча; 1 — обучение по одному
                примеру.
            on_epoch_end: функция f(эпоха, ошибка), которую сеть
                вызывает после каждой эпохи (например, для печати).
            seed: зерно для перемешивания примеров.

        Returns:
            list[float]: история ошибки по эпохам.

        Raises:
            InvalidInputError: если параметры обучения неверны.
            MismatchedDataError: если размерности не совпадают.
        """
        if learning_rate <= 0 or epochs < 1 or batch_size < 1:
            raise InvalidInputError(
                "скорость обучения, число эпох и размер батча "
                "должны быть больше 0"
            )
        features = self._check_features(features)
        target_matrix = self.prepare_targets(targets)
        if features.shape[0] != target_matrix.shape[0]:
            raise MismatchedDataError(
                "число примеров и число ответов не совпадает"
            )
        rng = np.random.default_rng(seed)
        sample_count = features.shape[0]
        for epoch in range(1, epochs + 1):
            order = rng.permutation(sample_count)
            for start in range(0, sample_count, batch_size):
                batch_index = order[start:start + batch_size]
                batch_features = features[batch_index]
                batch_targets = target_matrix[batch_index]
                predictions = self.forward(batch_features)
                self.backward(predictions, batch_targets, learning_rate)
            epoch_loss = self.mse(self.forward(features), target_matrix)
            self.loss_history.append(epoch_loss)
            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_loss)
        return self.loss_history

    def predict(self, features):
        """Возвращает выходы сети (вероятности от 0 до 1)."""
        return self.forward(features)

    def predict_classes(self, features):
        """Возвращает номера предсказанных классов.

        При одном выходе класс 1, если выход не меньше 0.5.
        При нескольких выходах — номер выхода с наибольшим
        значением.
        """
        outputs = self.predict(features)
        if self.output_size == 1:
            return (outputs >= 0.5).astype(int).reshape(-1)
        return np.argmax(outputs, axis=1)

    def accuracy(self, features, targets):
        """Доля правильно угаданных классов (от 0 до 1)."""
        predicted = self.predict_classes(features)
        actual = np.asarray(targets).reshape(-1).astype(int)
        return float(np.mean(predicted == actual))

    def save(self, path):
        """Сохраняет архитектуру и веса сети в JSON-файл.

        Args:
            path: путь к файлу.

        Returns:
            pathlib.Path: путь к сохранённому файлу.

        Raises:
            WeightsFileError: если файл не удалось записать.
        """
        file_path = Path(path)
        data = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "weights": [layer.weights.tolist() for layer in self.layers],
            "biases": [layer.biases.tolist() for layer in self.layers],
            "loss_history": self.loss_history,
        }
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as weights_file:
                json.dump(data, weights_file, ensure_ascii=False)
        except OSError as error:
            raise WeightsFileError(
                f"не удалось записать файл {file_path}: {error}"
            ) from error
        return file_path

    @classmethod
    def load(cls, path):
        """Создаёт сеть из JSON-файла, сохранённого методом save.

        Args:
            path: путь к файлу.

        Returns:
            NeuralNetwork: сеть с загруженными весами.

        Raises:
            WeightsFileError: если файла нет или он повреждён.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise WeightsFileError(f"файл не найден: {file_path}")
        try:
            with file_path.open(encoding="utf-8") as weights_file:
                data = json.load(weights_file)
            network = cls(data["layer_sizes"], data["activation"])
            for layer, weights, biases in zip(
                    network.layers, data["weights"], data["biases"]):
                weights = np.array(weights)
                biases = np.array(biases)
                if (weights.shape != layer.weights.shape
                        or biases.shape != layer.biases.shape):
                    raise WeightsFileError(
                        "размеры весов не совпадают с архитектурой"
                    )
                layer.weights = weights
                layer.biases = biases
            network.loss_history = list(data.get("loss_history", []))
        except (json.JSONDecodeError, KeyError, TypeError,
                ValueError) as error:
            raise WeightsFileError(
                f"файл весов повреждён: {file_path}"
            ) from error
        return network
