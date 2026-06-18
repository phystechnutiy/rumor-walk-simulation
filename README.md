# Rumor Walk Simulation

## Идея проекта

Проект реализует компьютерную модель распространения слухов в обществе, используя аналогию со случайными блужданиями из термодинамики. Слух распространяется через мобильных агентов, которые перемещаются по графу социальной сети и передают информацию при контактах. Модель опирается на классические подходы Maki-Thompson и Daley-Kendall.

## Архитектура

| Модуль | Описание |
|---|---|
| `agent.py` | Агент с одним из четырёх состояний: ignorant / spreader / stifler / cooperator |
| `graph.py` | Граф социальной сети (Эрдёша-Реньи) |
| `interaction_model.py` | Правила взаимодействия агентов |
| `simulator.py` | Движок симуляции с поддержкой Monte-Carlo |
| `phase_analyzer.py` | Анализ фазовых переходов и критических значений |
| `dashboard.py` | Интерактивная визуализация через Plotly Dash |
| `config/config.py` | Конфигурация симуляции через dataclass |

## Состояния агента

```
ignorant  →  spreader  →  stifler
                ↓
           cooperator
```

- **ignorant** - не знает слух
- **spreader** - активно распространяет слух
- **stifler** - знает слух, но больше не распространяет
- **cooperator** - знает слух, не распространяет и снижает вероятность заражения ignorant

## Параметры модели

| Параметр | Описание | По умолчанию |
|---|---|---|
| `spread_prob` | Вероятность, что spreader заразит ignorant | `0.3` |
| `stifle_prob` | Вероятность, что spreader станет stifler | `0.3` |
| `cooperate_prob` | Вероятность, что ignorant станет cooperator | `0.1` |
| `n` | Количество агентов и узлов графа | `10` |
| `p` | Вероятность ребра в графе Эрдёша-Реньи | `0.5` |
| `seed` | Seed для воспроизводимости | `0` |
| `n_runs` | Количество Monte-Carlo прогонов | `5` |
| `param_start` | Начальное значение при переборе | `0.1` |
| `param_step` | Шаг перебора | `0.1` |
| `sizes` | Размеры сетей для фазового анализа | `[10, 50]` |
| `dashboard` | Запустить визуализацию | `false` |

## Реализованные фичи

### Модель cooperator
Расширение классической модели Maki-Thompson новым состоянием агента. Cooperator — агент, который знает слух, не распространяет его сам и при контакте с ignorant может переводить его напрямую в cooperator, минуя стадию spreader. Это позволяет моделировать «иммунизацию» — сценарии, когда часть общества сознательно блокирует распространение слуха.

### Параллельный Monte-Carlo
`Simulator.run_monte_carlo(n_runs)` запускает независимые прогоны через `ProcessPoolExecutor`, используя все доступные ядра процессора. Каждый прогон получает свой `np.random.default_rng()` без seed, что гарантирует статистическую независимость выборок.

### Фазовый анализ
`PhaseAnalyzer` перебирает значения `spread_prob` от `param_start` с шагом `param_step` и для каждого значения запускает Monte-Carlo симуляцию. Реализованы два метода нахождения критической точки:
- `find_param_crit` - первое ненулевое значение финального охвата
- `find_inflection_point` - точка максимального прироста охвата (точка перегиба кривой)

### Live Dashboard
Интерактивная анимация на Plotly Dash: граф с агентами, раскрашенными по состоянию, кнопка Play и слайдер по тикам. Позволяет наблюдать движение агентов и распространение слуха в реальном времени.

### Конфигурация через Hydra
Все параметры симуляции управляются через `conf/config.yaml` и могут быть переопределены из командной строки без изменения кода.

## Зависимости

| Библиотека | Версия | Использование |
|---|-------|---|
| `numpy` | 2.4.4 | Генератор случайных чисел |
| `plotly` | 6.7.0 | Визуализация |
| `dash` | 4.1.0 | Интерактивный дашборд |
| `networkx` | 3.6.1 | Layout графа |
| `hydra-core` | 1.3.2 | CLI и конфигурация |
| `matplotlib` | 3.10.9 | Графики профилирования |
| `pytest` | 9.0.3 | Тестирование |
| `pytest-cov` | 7.1.0 | Покрытие тестами |

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

### Режимы

#### Monte-Carlo симуляция

Запускает `n_runs` независимых прогонов и печатает историю состояний агентов по шагам.

```bash
python3 main.py mode=monte_carlo
```

#### Monte-Carlo с визуализацией

Запускает один прогон и открывает интерактивный дашборд в браузере. Дашборд показывает граф с агентами, раскрашенными по состоянию, и анимацию их движения по тикам.

```bash
python3 main.py mode=monte_carlo dashboard=true
```

#### Поиск критических значений

Перебирает значения `spread_prob` от `param_start` с шагом `param_step` для каждого размера сети из `sizes`. Находит критическое значение, при котором слух начинает распространяться.

```bash
python3 main.py mode=critical
```

#### Поиск точек перегиба

То же, но ищет значение по точке перегиба кривой распространения — там где рост числа stifler'ов максимален.

```bash
python3 main.py mode=inflection
```

#### Рассчет коэффицента диффузии

Считает коэффицент диффузии агентов

```bash
python3 main.py mode=diffusion
```


### Переопределение параметров

Любой параметр можно переопределить прямо в командной строке:

```bash
# изменить размер сети и количество прогонов
python3 main.py mode=monte_carlo n=50 n_runs=20

# изменить вероятности
python3 main.py mode=monte_carlo spread_prob=0.5 stifle_prob=0.2 cooperate_prob=0.15

# фазовый анализ с мелким шагом и несколькими размерами
python3 main.py mode=critical param_start=0.05 param_step=0.02 sizes=[10,50,100]

# воспроизводимый запуск с фиксированным seed
python3 main.py mode=monte_carlo seed=42
```

### Управление дашбордом

После запуска с `dashboard=true` открой браузер по адресу `http://localhost:8050`.

| Элемент | Описание |
|---|---|
| Кнопка **Play** | Запустить анимацию по тикам |
| Слайдер | Перемотать на конкретный тик вручную |
| 🔵 Синий | ignorant |
| 🔴 Красный | spreader |
| ⚫ Серый | stifler |
| 🟢 Зелёный | cooperator |

## Тестирование

Тесты находятся в папке `tests/` и покрывают все модули проекта.

### Запуск тестов

```bash
# запустить все тесты
pytest tests/

# с отчётом о покрытии в консоль
pytest --cov=. --cov-report=term-missing tests/

# с HTML-отчётом (открыть htmlcov/index.html)
pytest --cov=. --cov-report=html tests/
```

### Покрытие

| Модуль | Покрытие |
|---|---|
| `agent.py` | 100% |
| `interaction_model.py` | 100% |
| `phase_analyzer.py` | 100% |
| `dashboard.py` | 100% |
| `graph.py` | 97% |
| `simulator.py` | 88% |
| `main.py` | 85% |
| **Итого** | **98%** |

Непокрытые строки в `main.py` — декоратор `@hydra.main` и блок `if __name__ == "__main__"`: тестирование entry point не является стандартной практикой. Непокрытые строки в `simulator.py` — внутренности `ProcessPoolExecutor`, которые выполняются в subprocess и не трассируются coverage.

### Структура тестов

```
tests/
├── test_agent.py            # State, Agent.__init__, сеттеры state/position
├── test_graph.py            # Graph, add_node/edge, get_neighbors, ErdosRenyiGraph
├── test_interaction_model.py # все 5 правил взаимодействия, сеттеры вероятностей, interact()
├── test_simulator.py        # collect, step, run, reset, set_spread_prob, _single_run
├── test_phase_analyzer.py   # compute_final_reach, find_param_crit, find_inflection_point, run
├── test_dashboard.py        # _build_layout, _get_agent_positions, _build_frame, COLOR_MAP
└── test_main.py             # build_simulator, monte_carlo, init_phase_analyzer, critical_lambdas
```

## Профилирование

Профилирование реализовано в ноутбуке `profiling.ipynb`. Для запуска:

```bash
pip install jupyter
jupyter notebook profiling.ipynb
```

### Что профилируется

- `Simulator.step()` - 50 вызовов подряд на сети из 20 агентов
- `Simulator.run()` - полный прогон на сети из 100 агентов
- `Simulator.run_monte_carlo(10)` - 10 параллельных прогонов на сети из 50 агентов
- Масштабируемость `run()` при n = 10, 20, 50, 100

### Основные выводы

Узкое место — `itertools.combinations()` внутри `step()`: перебор всех пар агентов на одном узле даёт сложность O(k²) по числу агентов на узле. При росте n время выполнения `run()` растёт быстрее линейного. Второй по весу вызов — `copy.deepcopy()` в `collect()` для сохранения снапшотов.

## Участники

Алексей Кондрашкин — https://github.com/phystechnutiy/rumor-walk-simulation