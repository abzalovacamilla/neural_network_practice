"""Генератор тестового набора данных «Сдаст ли студент экзамен».

Создаёт CSV-файл с признаками студентов и целевым столбцом
passed (1 — сдал, 0 — не сдал). Запуск из корня проекта:

    python -m part1_mlp.app.generate_data
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

HEADER = ["study_hours", "attendance", "sleep_hours",
          "previous_grade", "passed"]


def generate_students(row_count, seed):
    """Генерирует таблицу студентов.

    Итоговый балл растёт с часами подготовки, посещаемостью
    и прошлой оценкой, а также зависит от сна: лучше всего
    около 7,5 часа. Студент сдал, если балл выше медианы.

    Args:
        row_count: количество строк.
        seed: зерно генератора случайных чисел.

    Returns:
        numpy.ndarray: таблица формы (row_count, 5).
    """
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 10, row_count).round(1)
    attendance = rng.uniform(30, 100, row_count).round(0)
    sleep_hours = rng.uniform(4, 10, row_count).round(1)
    previous_grade = rng.integers(2, 6, row_count)
    score = (0.5 * study_hours + 0.04 * attendance
             - 0.6 * np.abs(sleep_hours - 7.5)
             + 0.8 * previous_grade
             + rng.normal(0, 0.5, row_count))
    passed = (score > np.median(score)).astype(int)
    return np.column_stack(
        [study_hours, attendance, sleep_hours, previous_grade, passed]
    )


def save_csv(table, output_path):
    """Сохраняет таблицу в CSV с заголовком.

    Args:
        table: таблица значений.
        output_path: путь к файлу.

    Returns:
        pathlib.Path: путь к сохранённому файлу.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(HEADER)
        for row in table:
            writer.writerow([f"{value:g}" for value in row])
    return output_path


def main():
    """Разбирает аргументы командной строки и создаёт файл."""
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data/input")
    parser = argparse.ArgumentParser(
        description="Генерация тестового CSV для части 1"
    )
    parser.add_argument("--rows", type=int, default=300,
                        help="количество строк (по умолчанию 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="зерно генератора (по умолчанию 42)")
    parser.add_argument("--output",
                        default=str(Path(data_path) / "students.csv"),
                        help="путь к CSV-файлу")
    args = parser.parse_args()
    if args.rows < 10:
        parser.error("--rows должно быть не меньше 10")
    table = generate_students(args.rows, args.seed)
    saved_path = save_csv(table, args.output)
    print(f"Создан файл {saved_path}: {args.rows} строк, "
          f"сдали {int(table[:, -1].sum())}")


if __name__ == "__main__":
    main()
