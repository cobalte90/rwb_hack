# WildHack Slot-based Warehouse Load Orchestrator

Production-like сервис для хакатонного кейса WildHack. Система не просто прогнозирует отгрузки, а помогает складу принимать операционные решения по временным слотам:

`Входные данные -> Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package`

На выходе сервис возвращает не только прогноз, но и:

- уровень slot pressure
- рекомендуемое действие: `call_now | monitor | hold`
- рекомендуемое число машин
- срочность
- объяснение, почему принято именно такое решение

## Что решает сервис

Контекст задачи:

- есть маршруты `route_id`
- по ним во времени меняется поток отгрузок
- если заранее не увидеть пик нагрузки, склад может перегрузиться или, наоборот, вызвать лишний транспорт

Что предсказываем:

- `target_2h` на горизонте `10` шагов по `30` минут
- это ожидаемая нагрузка в ближайшие 2 часа для каждого будущего слота

Что делает сервис поверх прогноза:

- считает `slot pressure`
- оценивает риск
- рекомендует действие
- рассчитывает число машин
- формирует понятный decision package для маршрута/склада

## Что реально использует runtime

Основной runtime-профиль: `latest_lb`.

В нём сервис использует:

- `Chronos2` real, если лежат экспортированные веса
- `GRU` real
- `TSMixerx` real, если лежит обученный bundle
- fallback на proxy-ветки, если real-артефакты недоступны

Дополнительный профиль: `local_fallback`.

В нём сервис использует:

- `GRU`
- `TFT proxy`
- optional `Optuna` models

## Структура сервиса

Ключевые модули:

- `app/core/slot_pressure.py` — расчёт напряжения склада по слотам
- `app/core/action_engine.py` — выбор `call_now / monitor / hold`
- `app/core/decision_logic.py` — сборка decision package
- `app/core/kpi.py` — продуктовые KPI
- `app/core/loaders.py` — загрузка артефактов и моделей
- `app/core/forecasting.py` — runtime inference по моделям

## Форматы входных данных

Сервис принимает:

- `JSON`
- `CSV`
- `Parquet`

Поддерживается хакатонный тестовый формат:

- `id`
- `route_id`
- `timestamp`

Если в файле нет служебных параметров сервиса, он подставит их сам:

- `model_profile = latest_lb`
- `horizon_steps = 10`
- `service_mode = balanced`

## Быстрый запуск

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

### 3. Docker

```bash
docker build -t wildhack-slot-orchestrator .
docker run --rm -p 8000:8000 wildhack-slot-orchestrator
```

или

```bash
docker compose up --build
```

## Как пользоваться UI

Главный экран разбит на 3 зоны:

- слева — `Панель управления`
- в центре — `Приоритетное решение`
- справа — `Очередь действий`

Ниже:

- таблица всех маршрутов
- маршрут в фокусе
- KPI и бизнес-инсайты

### Кнопки и элементы управления

#### `Загрузить демо`

Подставляет готовый demo batch и сразу запускает расчёт. Удобно для защиты и быстрой проверки UI.

#### `Запустить расчёт`

Запускает расчёт:

- по выбранному файлу, если файл загружен
- по JSON из блока `Расширенный ввод`, если файл не выбран

#### `Загрузить файл`

Поддерживает:

- `.json`
- `.csv`
- `.parquet`

Рекомендуемый сценарий для показа:

1. открыть UI
2. выбрать `balanced`
3. загрузить `Parquet`
4. нажать `Запустить расчёт`

#### `Режим сервиса`

Поддерживаются режимы:

- `Экономия`
- `Баланс`
- `Приоритет SLA`

Они реально влияют на:

- sensitivity к pressure
- safety buffer
- пороги `monitor` / `call_now`
- срочность решения

### Что смотреть на экране в первую очередь

#### Верхняя KPI-полоса

Показывает:

- сколько критических маршрутов
- сколько машин нужно вызвать сейчас
- среднюю нагрузку
- ближайшее пиковое окно
- активный режим сервиса
- риск SLA

#### `Приоритетное решение`

Это главный маршрут, которому нужно внимание сейчас.

Показывает:

- маршрут и склад
- action
- pressure level
- пик через X минут
- число машин
- why / factors
- мини-график прогноза на 10 шагов

#### `Очередь действий`

Краткий operational view по ключевым маршрутам:

- action
- trucks
- urgency
- pressure
- следующее действие / момент review

#### `Все маршруты`

Обзор всех маршрутов в запросе:

- action
- pressure
- объём
- trucks
- urgency
- короткая причина

Клик по строке переводит маршрут в нижний блок `Маршрут в фокусе`.

#### `Маршрут в фокусе`

Детальная карточка маршрута:

- action
- pressure
- peak
- trucks
- следующее действие
- explanation
- факторы риска
- прогноз на 10 шагов

## Важные API ручки

- `GET /health` — готовность сервиса и профилей
- `GET /config` — конфиг runtime и бизнес-правил
- `POST /predict` — только прогноз
- `POST /plan` — полный decision package
- `POST /explain` — объяснение по маршрутам
- `POST /kpi` — KPI snapshot
- `POST /plan/file` — расчёт напрямую по файлу

## Команды для проверки

```bash
python scripts/validate_models.py
python scripts/make_submission.py --profile latest_lb
python -m pytest -q
```

## Откуда брать веса моделей

Ниже перечислены файлы, которые нужны для inference.

### Вариант 1. Рекомендуемый пакет для `latest_lb`

Это лучший вариант для демонстрации и защиты. В него входят real runtime-модели:

- `artifacts/models/chronos2/model.safetensors`
- `artifacts/models/chronos2/config.json`
- `artifacts/models/chronos2/chronos2_config.json`
- `artifacts/models/chronos2/export_metadata.json`
- `artifacts/models/gru/gru.pt`
- `artifacts/models/gru/gru_config.json`
- `artifacts/models/tsmixerx/tsmixerx_config.json`
- `artifacts/models/tsmixerx/static_features.parquet`
- `artifacts/models/tsmixerx/bundle/TSMixerx_0.ckpt`
- `artifacts/models/tsmixerx/bundle/configuration.pkl`
- `artifacts/models/tsmixerx/bundle/alias_to_model.pkl`

### Вариант 2. Fallback-файлы, которые лучше тоже положить

Если хотите, чтобы сервис поднимался даже без real heavy artifacts:

- `artifacts/models/chronos_proxy/chronos_proxy.npz`
- `artifacts/models/chronos_proxy/chronos_proxy_meta.json`
- `artifacts/models/timexer_proxy/timexer_proxy.npz`
- `artifacts/models/timexer_proxy/timexer_proxy_meta.json`
- `artifacts/models/tft_lite/tft_lite.npz`
- `artifacts/models/tft_lite/tft_lite_meta.json`

### Что ещё нужно кроме весов

Для inference нужны не только веса, но и сопутствующие артефакты:

#### Конфиги

- `artifacts/configs/model_registry.json`
- `artifacts/configs/blend_config.json`
- `artifacts/configs/business_rules.yaml`
- `artifacts/configs/preprocessing.json`

#### Статистики и history tail

- `artifacts/stats/route_stats.parquet`
- `artifacts/stats/office_stats.parquet`
- `artifacts/stats/route_office_map.parquet`
- `artifacts/stats/history_tail.parquet`
- `artifacts/stats/route_time_profiles.parquet`
- `artifacts/stats/office_time_profiles.parquet`
- `artifacts/stats/global_time_profiles.parquet`
- `artifacts/stats/status_route_friday_profiles.parquet`

### Практическая рекомендация

Чтобы не собирать inference-пакет вручную, лучше залить в облако целиком такие папки:

- `artifacts/models/chronos2/`
- `artifacts/models/gru/`
- `artifacts/models/tsmixerx/`
- `artifacts/configs/`
- `artifacts/stats/`

И опционально:

- `artifacts/models/chronos_proxy/`
- `artifacts/models/timexer_proxy/`
- `artifacts/models/tft_lite/`

## Ссылка на скачивание весов

Готовая ссылка на inference bundle:

- `https://drive.google.com/drive/folders/1eCMotnRsqOYSVd-BhQEX-ZgewIeNJwwn?usp=sharing`

В папке лежат:

- `artifacts/configs`
- `artifacts/models`
- `artifacts/stats`

Этого достаточно для запуска текущего runtime.

## Какие файлы лучше положить в облако

Если нужен один понятный архив для запуска сервиса, лучше сделать архив:

- `wildhack_inference_bundle.zip`

Внутри:

- `artifacts/models/chronos2/`
- `artifacts/models/gru/`
- `artifacts/models/tsmixerx/`
- `artifacts/configs/`
- `artifacts/stats/`

И отдельный optional архив:

- `wildhack_fallback_models.zip`

Внутри:

- `artifacts/models/chronos_proxy/`
- `artifacts/models/timexer_proxy/`
- `artifacts/models/tft_lite/`

## Запуск на сервере через Docker

### Вариант 1. Самый простой

Если рядом с репозиторием уже есть папка `artifacts`, просто выполните:

```bash
docker compose up -d --build
```

### Вариант 2. Если артефактов на сервере ещё нет

Контейнер умеет сам подтянуть их из Google Drive по переменной:

- `ARTIFACTS_GDRIVE_URL`

В `docker-compose.yml` ссылка уже прописана. То есть на сервере достаточно:

```bash
git clone <repo_url>
cd wb_hack
docker compose up -d --build
```

При первом запуске сервис скачает `artifacts/configs`, `artifacts/models` и `artifacts/stats`, сохранит их в `./artifacts` и затем поднимет API.

### Полезные команды на сервере

Статус контейнера:

```bash
docker compose ps
```

Логи:

```bash
docker compose logs -f
```

Проверка health:

```bash
curl http://127.0.0.1:8000/health
```

## Проверенный demo-файл

Для быстрой демонстрации удобно использовать:

- `examples/test_first_100_routes.parquet`

Это укороченная версия хакатонного `test.parquet`:

- 100 маршрутов
- 1000 строк
- тот же формат входа

## Ограничения

- `blend_best_chronos_groupwise_main.csv` сохранён как frozen reference, а не как точно восстановленная формула
- KPI layer — честная proxy-реализация, потому что production dispatch logs в исходных данных отсутствуют
- proxy-модели остаются в репозитории как fallback runtime path

## Документация

- `docs/architecture.md`
- `docs/business_logic.md`
- `docs/metrics.md`
- `docs/assumptions.md`
- `docs/demo.md`
