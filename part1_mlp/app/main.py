"""Консольное меню для работы с нейронной сетью.

Запуск из корня проекта:

    python -m part1_mlp.app.main
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from .exceptions import (DataNotLoadedError, InvalidInputError,
                         NetworkNotCreatedError, NeuralNetworkError)
from .manager import DatasetManager
from .models import ACTIVATIONS, NeuralNetwork
from .utils import (parse_float, parse_int, parse_layer_sizes,
                    parse_vector, plot_loss, read_feature_file)

MENU = """
========== МНОГОСЛОЙНЫЙ ПЕРЦЕПТРОН ==========
1. Создать сеть
2. Загрузить данные из CSV
3. Обучить сеть
4. Сделать предсказание
5. Сохранить / загрузить веса
6. Показать график ошибки
7. Выход
============================================="""


def ask(prompt, default=None):
    """Запрашивает строку у пользователя.

    Args:
        prompt: текст вопроса.
        default: значение, если пользователь нажал Enter.

    Returns:
        str: введённая строка или значение по умолчанию.
    """
    hint = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{hint}: ").strip()
    if not answer and default is not None:
        return str(default)
    return answer


class Application:
    """Связывает меню, сеть и менеджер данных."""

    def __init__(self, data_dir, output_dir, dataset_file, seed):
        """Сохраняет настройки и создаёт пустое состояние.

        Args:
            data_dir: папка с входными данными.
            output_dir: папка для результатов (веса, графики).
            dataset_file: имя CSV-файла по умолчанию.
            seed: зерно генератора для воспроизводимости.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.dataset_file = dataset_file
        self.seed = seed
        self.network = None
        self.dataset = DatasetManager()
        self.actions = {
            "1": self.create_network,
            "2": self.load_data,
            "3": self.train_network,
            "4": self.predict,
            "5": self.save_or_load_weights,
            "6": self.show_loss_plot,
        }

    def require_network(self):
        """Проверяет, что сеть создана.

        Raises:
            NetworkNotCreatedError: если сети нет.
        """
        if self.network is None:
            raise NetworkNotCreatedError(
                "сначала создайте сеть (пункт 1) или загрузите веса "
                "(пункт 5)"
            )

    def create_network(self):
        """Пункт 1: создаёт сеть или меняет функцию активации."""
        if self.network is not None:
            print(f"Текущая сеть: {self.network.describe()}")
            choice = ask("1 — создать новую сеть, "
                         "2 — сменить функцию активации", "1")
            if choice == "2":
                name = ask(f"Функция активации ({'/'.join(ACTIVATIONS)})")
                self.network.set_activation(name)
                print(f"Готово: {self.network.describe()}")
                return
        if self.dataset.is_loaded:
            print(f"В данных {self.dataset.feature_count} признака(ов): "
                  f"столько же должно быть входов сети.")
        sizes = parse_layer_sizes(
            ask("Размеры слоёв через пробел (вход скрытые выход), "
                "например 4 8 1")
        )
        activation = ask(
            f"Функция активации скрытых слоёв ({'/'.join(ACTIVATIONS)})",
            "sigmoid",
        )
        self.network = NeuralNetwork(sizes, activation, seed=self.seed)
        print(f"Сеть создана: {self.network.describe()}")

    def load_data(self):
        """Пункт 2: загружает CSV, делит и нормализует выборку."""
        default_path = self.data_dir / self.dataset_file
        path = ask("Путь к CSV-файлу", default_path)
        row_count = self.dataset.load_csv(path)
        print(f"Загружено строк: {row_count}")
        print(self.dataset.summary())
        test_ratio = parse_float(
            ask("Доля тестовой выборки (от 0 до 1)", 0.2),
            "доля теста", 0, 1,
        )
        method = ask("Нормализация (standard/minmax)", "standard")
        train_count, test_count = self.dataset.split(test_ratio, self.seed)
        self.dataset.normalize(method)
        print(f"Обучающая выборка: {train_count}, тестовая: {test_count}, "
              f"нормализация: {method}")

    def train_network(self):
        """Пункт 3: обучает сеть и выводит ошибку по эпохам."""
        self.require_network()
        if not self.dataset.is_split:
            raise DataNotLoadedError("сначала загрузите данные (пункт 2)")
        learning_rate = parse_float(
            ask("Скорость обучения", 0.5), "скорость обучения", 0
        )
        epochs = parse_int(ask("Число эпох", 200), "число эпох")
        batch_size = parse_int(
            ask("Размер батча (1 — по одному примеру)", 16), "размер батча"
        )
        report_every = max(1, epochs // 10)

        def report(epoch, loss):
            """Печатает ошибку каждые report_every эпох."""
            if epoch == 1 or epoch % report_every == 0:
                print(f"  эпоха {epoch:>5}/{epochs}  ошибка MSE = "
                      f"{loss:.5f}")

        print("Обучение...")
        self.network.train(
            self.dataset.train_features, self.dataset.train_targets,
            learning_rate, epochs, batch_size,
            on_epoch_end=report, seed=self.seed,
        )
        train_accuracy = self.network.accuracy(
            self.dataset.train_features, self.dataset.train_targets
        )
        test_accuracy = self.network.accuracy(
            self.dataset.test_features, self.dataset.test_targets
        )
        print(f"Точность на обучающей выборке: {train_accuracy:.1%}")
        print(f"Точность на тестовой выборке:  {test_accuracy:.1%}")

    def predict(self):
        """Пункт 4: предсказание для введённых или тестовых примеров."""
        self.require_network()
        choice = ask("1 — ввести вручную, 2 — из CSV-файла, "
                     "3 — примеры из тестовой выборки", "1")
        if choice == "1":
            names = ", ".join(self.dataset.feature_names) or "признаки"
            features = parse_vector(ask(f"Введите через пробел: {names}"))
            self.print_predictions(self.dataset.transform(features))
        elif choice == "2":
            features = read_feature_file(ask("Путь к CSV-файлу"))
            self.print_predictions(self.dataset.transform(features))
        elif choice == "3":
            if not self.dataset.is_split:
                raise DataNotLoadedError("сначала загрузите данные")
            count = parse_int(ask("Сколько примеров показать", 5),
                              "количество примеров")
            features = self.dataset.test_features[:count]
            actual = self.dataset.test_targets[:count]
            self.print_predictions(features, actual)
        else:
            raise InvalidInputError(f"нет варианта '{choice}'")

    def print_predictions(self, features, actual=None):
        """Печатает выход сети и предсказанный класс.

        Args:
            features: нормализованные признаки.
            actual: правильные ответы или None.
        """
        outputs = self.network.predict(features)
        classes = self.network.predict_classes(features)
        for index, (output, predicted) in enumerate(zip(outputs, classes)):
            probabilities = ", ".join(f"{value:.3f}" for value in output)
            line = (f"  пример {index + 1}: выход сети [{probabilities}] "
                    f"-> класс {predicted}")
            if actual is not None:
                line += f" (правильный ответ: {int(actual[index])})"
            print(line)

    def save_or_load_weights(self):
        """Пункт 5: сохраняет веса в JSON или загружает их."""
        default_path = self.output_dir / "weights.json"
        choice = ask("1 — сохранить веса, 2 — загрузить веса", "1")
        if choice == "1":
            self.require_network()
            saved = self.network.save(ask("Путь к файлу", default_path))
            print(f"Веса сохранены: {saved}")
        elif choice == "2":
            self.network = NeuralNetwork.load(
                ask("Путь к файлу", default_path)
            )
            print(f"Веса загружены. {self.network.describe()}")
        else:
            raise InvalidInputError(f"нет варианта '{choice}'")

    def show_loss_plot(self):
        """Пункт 6: строит график ошибки и сохраняет его в PNG."""
        self.require_network()
        if not self.network.loss_history:
            raise InvalidInputError("сеть ещё не обучалась (пункт 3)")
        path = plot_loss(self.network.loss_history,
                         self.output_dir / "loss_plot.png")
        print(f"График сохранён: {path}")

    def run(self):
        """Главный цикл меню: работает до выбора пункта 7."""
        while True:
            print(MENU)
            choice = input("Выберите пункт: ").strip()
            if choice == "7":
                print("До свидания!")
                break
            action = self.actions.get(choice)
            if action is None:
                print("Ошибка: введите число от 1 до 7")
                continue
            try:
                action()
            except NeuralNetworkError as error:
                print(f"Ошибка: {error}")


def main():
    """Читает настройки из окружения и запускает меню."""
    load_dotenv()
    app = Application(
        data_dir=os.getenv("DATA_PATH", "data/input"),
        output_dir=os.getenv("OUTPUT_DIR", "data/output"),
        dataset_file=os.getenv("DATASET_FILE", "students.csv"),
        seed=int(os.getenv("RANDOM_SEED", "42")),
    )
    try:
        app.run()
    except (KeyboardInterrupt, EOFError):
        print("\nРабота прервана пользователем")


if __name__ == "__main__":
    main()
