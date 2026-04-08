# WildHack Slot-based Warehouse Load Orchestrator

Операционный сервис для склада, который не просто прогнозирует нагрузку, а помогает принять действие по маршрутам в ближайшие временные слоты.

Коротко:

- принимает входные данные в формате `json / csv / parquet`
- прогнозирует `target_2h` на горизонте `10` шагов по `30` минут
- оценивает `slot pressure`
- рекомендует действие: `call_now / monitor / hold`
- рассчитывает число машин и срочность
- объясняет, почему система предлагает именно это решение

Продуктовый поток:

`Input data -> Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package`

---

## Публичная ссылка на сервис

После деплоя добавьте сюда публичную ссылку:

- `Сервис: <ДОБАВЬТЕ_ПУБЛИЧНУЮ_ССЫЛКУ_НА_UI>`
- `Swagger: <ДОБАВЬТЕ_ПУБЛИЧНУЮ_ССЫЛКУ_НА_/docs>`

---

## Ссылка на веса и inference bundle

Готовый набор артефактов для инференса:

- `https://drive.google.com/drive/folders/1eCMotnRsqOYSVd-BhQEX-ZgewIeNJwwn?usp=sharing`

В облаке уже лежат нужные папки:

- `artifacts/configs`
- `artifacts/models`
- `artifacts/stats`

Этого достаточно для запуска текущего runtime-пайплайна сервиса.

---

## Что делает сервис

Контекст задачи:

- есть маршруты `route_id`
- по ним во времени меняется поток отгрузок
- если не увидеть пик нагрузки заранее, склад может перегрузиться или вызвать лишний транспорт

Что предсказываем:

- `target_2h` на горизонте `10` будущих слотов
- это ожидаемая нагрузка в ближайшие 2 часа для каждого будущего временного окна

Что делает система поверх прогноза:

- считает `slot pressure`
- оценивает риск по маршруту
- рекомендует действие
- рассчитывает число машин
- возвращает структурированный decision package для склада

---

## Быстрый запуск локально

### 1. Установить зависимости

```bash
pip install -r requirements.txt
```

### 2. Запустить сервис

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

После запуска:

- UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### 3. Быстрая проверка

Откройте UI и:

1. нажмите `Загрузить демо`
2. либо выберите `parquet/csv/json`
3. нажмите `Запустить расчёт`

---

## Запуск через Docker

### Локально

```bash
docker compose up --build
```

или

```bash
docker build -t wildhack-slot-orchestrator .
docker run --rm -p 8000:8000 wildhack-slot-orchestrator
```

### На сервере

Самый простой сценарий:

```bash
git clone <repo_url>
cd wb_hack
docker compose up -d --build
```

Если папки `artifacts/` рядом с проектом нет, контейнер может автоматически подтянуть артефакты по ссылке Google Drive через `ARTIFACTS_GDRIVE_URL`.

Полезные команды:

```bash
docker compose ps
docker compose logs -f
curl http://127.0.0.1:8000/health
```

---

## Как пользоваться интерфейсом

Интерфейс построен как clean operations console и разбит на 3 основные зоны:

- слева — `Панель управления`
- в центре — `Приоритетное решение`
- справа — `Очередь действий`

Ниже находятся:

- таблица всех маршрутов
- блок `Маршрут в фокусе`
- KPI и бизнес-инсайты

### Панель управления

Здесь находятся:

- выбор `режима сервиса`
- кнопка `Загрузить демо`
- кнопка `Запустить расчёт`
- загрузка файла `json / csv / parquet`
- краткая сводка по входным данным
- `Расширенный ввод` с JSON

### Что важно на экране

#### KPI-полоса сверху

Показывает:

- сколько критических маршрутов
- сколько машин нужно вызвать сейчас
- среднюю нагрузку
- ближайшее пиковое окно
- текущий режим сервиса
- риск SLA

#### Приоритетное решение

Главная карточка экрана. Показывает:

- ключевой маршрут
- действие, которое нужно выполнить первым
- pressure level
- пик через X минут
- число машин
- мини-график прогноза
- основные факторы, повлиявшие на решение

#### Очередь действий

Краткий операционный список по маршрутам:

- маршрут
- action
- trucks
- urgency
- pressure
- следующее действие / review / dispatch timing

#### Все маршруты

Таблица даёт обзор всего батча:

- action
- уровень давления
- объём
- trucks
- urgency
- следующее действие
- краткую причину

Клик по строке переводит маршрут в блок `Маршрут в фокусе`.

#### Маршрут в фокусе

Это drill-down карточка маршрута:

- action
- pressure
- peak
- trucks
- следующее действие
- объяснение
- риск-факторы
- прогноз на 10 шагов

---

## Какие кнопки есть в UI

### `Загрузить демо`

Подставляет готовый demo-сценарий и сразу запускает расчёт. Это самый быстрый способ показать сервис на защите.

### `Запустить расчёт`

Запускает расчёт:

- по загруженному файлу, если файл выбран
- по JSON, если файл не выбран

### `Загрузить файл`

Поддерживает:

- `.json`
- `.csv`
- `.parquet`

Сервис понимает и хакатонный тестовый формат:

- `id`
- `route_id`
- `timestamp`

Если служебные параметры отсутствуют, сервис добавляет их сам:

- `model_profile = latest_lb`
- `horizon_steps = 10`
- `service_mode = balanced`

### `Режим сервиса`

Поддерживаются режимы:

- `Экономия`
- `Баланс`
- `Приоритет SLA`

Они реально влияют на:

- чувствительность к нагрузке
- safety buffer
- пороги `monitor / call_now`
- срочность решения

---

## Модели и runtime-профили

Основной профиль: `latest_lb`.

Он использует:

- `Chronos2` real, если лежат реальные артефакты
- `GRU` real
- `TSMixerx` real, если лежит обученный bundle
- fallback на proxy-ветки, если heavy artifacts недоступны

Fallback-профиль: `local_fallback`.

Он использует:

- `GRU`
- `TFT proxy`
- optional `Optuna`

---

## Технические преимущества

### 1. Production-like inference

Сервис не обучает модели в runtime. Он:

- загружает готовые артефакты
- делает inference
- собирает blend
- передаёт прогноз в pressure/action слой

### 2. Настоящий decision engine, а не просто forecast API

После прогноза сервис принимает продуктово полезное решение:

- `call_now`
- `monitor`
- `hold`

То есть система отвечает не только на вопрос “сколько будет нагрузки”, но и на вопрос “что делать прямо сейчас”.

### 3. Slot Pressure Engine

Pressure считается на основе реальных факторов:

- near-term peak
- peak proximity
- route volatility
- model disagreement
- friday regime
- service mode

### 4. Action Engine

Решение не строится по одному тупому порогу. Учитываются:

- pressure score
- urgency
- буфер
- близость пика
- ожидаемая нагрузка
- режим сервиса

### 5. Понятный для бизнеса интерфейс

UI сделан не как технодемо, а как operational console:

- KPI summary
- priority route decision
- action queue
- routes overview
- drill-down по маршруту

### 6. Готовность к серверному деплою

В репозитории уже есть:

- `Dockerfile`
- `docker-compose.yml`
- bootstrap загрузки артефактов
- healthcheck
- конфиги runtime

---

## Архитектура проекта

Ключевые модули:

- `app/core/forecasting.py` — inference по моделям
- `app/core/slot_pressure.py` — расчёт slot pressure
- `app/core/action_engine.py` — action recommendation
- `app/core/decision_logic.py` — сборка decision package
- `app/core/kpi.py` — KPI layer
- `app/core/loaders.py` — загрузка артефактов и моделей
- `app/core/file_payloads.py` — загрузка и валидация `json/csv/parquet`

Конфиги и артефакты:

- `artifacts/configs/`
- `artifacts/models/`
- `artifacts/stats/`

Документация:

- `docs/architecture.md`
- `docs/business_logic.md`
- `docs/metrics.md`
- `docs/assumptions.md`
- `docs/demo.md`

---

## Полезные команды

Проверка моделей:

```bash
python scripts/validate_models.py
```

Проверка submission-like pipeline:

```bash
python scripts/make_submission.py --profile latest_lb
```

Тесты:

```bash
python -m pytest -q
```

Bootstrap артефактов:

```bash
python scripts/bootstrap_runtime.py
```

---

## Проверенный demo-файл

Для быстрой демонстрации удобно использовать:

- `examples/test_first_100_routes.parquet`

Это укороченная версия тестового файла:

- 100 маршрутов
- 1000 строк
- тот же входной формат, что и у исходного `test.parquet`

---

## Ограничения

- `blend_best_chronos_groupwise_main.csv` сохранён как frozen reference, а не как точно восстановленная формула
- KPI layer является честной proxy-реализацией, потому что production dispatch logs в исходных данных отсутствуют
- proxy-модели сохранены как fallback-runtime path на случай отсутствия heavy artifacts

---

## Куда смотреть в коде

- `app/main.py`
- `app/api/routes.py`
- `app/core/forecasting.py`
- `app/core/slot_pressure.py`
- `app/core/action_engine.py`
- `app/core/decision_logic.py`
- `app/core/kpi.py`

