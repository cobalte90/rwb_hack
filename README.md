# SAVE A SLOT   

Операционный сервис для склада, который не просто строит прогноз, а помогает принять решение по маршрутам на ближайшие временные слоты. 🚚

Сервис:

- принимает входные данные в форматах `json`, `csv`, `parquet`
- прогнозирует нагрузку `target_2h` на горизонте `10` шагов по `30` минут
- считает `slot pressure`
- рекомендует действие: `call_now`, `monitor`, `hold`
- оценивает число машин и срочность
- отдаёт удобный web UI и API

Продуктовый поток:

`Input data -> Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package`   

---
# ССЫЛКА НА СЕРВИС: http://85.239.60.13:8000
---

## Что внутри

Основной профиль `latest_lb` использует:

- `Chronos2` real
- `GRU` real
- `TSMixerx` real
- groupwise blend

Fallback-профиль `local_fallback` использует:

- `GRU`
- `TFT proxy`
- optional `Optuna`

Важно: при корректной сборке контейнера и наличии артефактов `/health` должен возвращать:

- `status: ok`
- `Chronos2 real, TSMixerx real`

---

## Артефакты

Для запуска нужны inference-артефакты:

- `artifacts/configs`
- `artifacts/models`
- `artifacts/stats`

Ссылка на готовый bundle:

- `https://drive.google.com/drive/folders/1eCMotnRsqOYSVd-BhQEX-ZgewIeNJwwn?usp=sharing`

После скачивания структура должна выглядеть так:

```text
wb_hack/
  app/
  artifacts/
    configs/
    models/
    stats/
  examples/
  Dockerfile
  docker-compose.yml
```

Без папки `artifacts/` сервис не стартует.

---

## Быстрый старт локально

### 1. Установить зависимости

```bash
pip install -r requirements.txt
```

### 2. Проверить, что артефакты лежат в корне проекта

Нужны директории:

- `./artifacts/configs`
- `./artifacts/models`
- `./artifacts/stats`

### 3. Запустить сервис

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Открыть

- UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

---

## Запуск через Docker

### Локально

```bash
docker compose up -d --build
```

Проверка:

```bash
docker compose ps
docker compose logs -f --tail=100
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ui/demo-payload
```

---

## Деплой на сервер

Ниже сценарий, который мы проверили на реальном сервере и который поднимает сервис без proxy fallback. ✅

### 1. Подготовить сервер

Нужно:

- Ubuntu/Linux сервер
- установленный Docker
- установленный Docker Compose
- открытый порт `8000/tcp`

### 2. Клонировать репозиторий

```bash
git clone <repo_url> /opt/rwb_hack
cd /opt/rwb_hack
```

### 3. Положить артефакты в корень проекта

После распаковки должны существовать:

- `/opt/rwb_hack/artifacts/configs`
- `/opt/rwb_hack/artifacts/models`
- `/opt/rwb_hack/artifacts/stats`

### 4. Собрать и поднять контейнер

```bash
cd /opt/rwb_hack
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 5. Проверить состояние

```bash
cd /opt/rwb_hack
docker compose ps
docker compose logs -f --tail=100
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ui/demo-payload
```

Ожидаемый health:

```json
{
  "status": "ok",
  "profiles": {
    "latest_lb": {
      "note": "Chronos2 real, TSMixerx real"
    }
  }
}
```

### 6. Проверить зависимости внутри контейнера

```bash
cd /opt/rwb_hack
docker compose exec transport-planner python -c "from chronos import Chronos2Pipeline; from neuralforecast import NeuralForecast; print('imports_ok')"
```

Если всё в порядке, сервис доступен снаружи:

- `http://<SERVER_IP>:8000/`
- `http://<SERVER_IP>:8000/docs`
- `http://<SERVER_IP>:8000/health`

---

## Как показать демо

Самый быстрый путь:

1. Открыть UI
2. Нажать `Загрузить демо`
3. Нажать `Запустить расчёт`

Ещё один удобный вариант:

- загрузить `examples/test_first_100_routes.parquet`

Это укороченный demo-набор, на котором удобно быстро проверять UI и расчёт.

---

## Форматы входных данных

Сервис понимает:

- `.json`
- `.csv`
- `.parquet`

Минимально ожидаемые поля:

- `id`
- `route_id`
- `timestamp`

Если часть служебных параметров не передана, сервис подставляет значения по умолчанию:

- `model_profile = latest_lb`
- `horizon_steps = 10`
- `service_mode = balanced`

---

## Основные API

- `GET /` — web UI
- `GET /docs` — Swagger UI
- `GET /health` — состояние runtime
- `GET /ui/meta` — метаданные интерфейса
- `GET /ui/demo-payload` — demo JSON для UI
- `POST /ui/plan-dashboard-file` — загрузка файла через UI
- `POST /predict` — прогноз без decision layer
- `POST /plan` — полный orchestration pipeline
- `POST /explain` — explainability
- `POST /kpi` — KPI snapshot

---

## Полезные команды

Проверка тестов:

```bash
python -m pytest -q
```

Проверка моделей:

```bash
python scripts/validate_models.py
```

Submission-like pipeline:

```bash
python scripts/make_submission.py --profile latest_lb
```

---

## Если что-то пошло не так

### `/ui/demo-payload` отдаёт `500`

Проверьте, что в Docker-образ попала папка `examples/`.

В контейнере должен существовать файл:

- `/app/examples/demo_plan_request.json`

Проверка:

```bash
cd /opt/rwb_hack
docker compose exec transport-planner ls -la /app/examples
```

### `/health` показывает proxy fallback

Проверьте:

```bash
cd /opt/rwb_hack
docker compose exec transport-planner python -c "import importlib.util; print('chronos', bool(importlib.util.find_spec('chronos'))); print('neuralforecast', bool(importlib.util.find_spec('neuralforecast')))"
```

И убедитесь, что контейнер собран с актуальными зависимостями:

- `chronos-forecasting`
- `neuralforecast`

### Сайт не открывается снаружи

Если внутри сервера `curl http://127.0.0.1:8000/health` работает, а снаружи сайт не открывается, почти всегда причина одна из этих:

- не открыт порт `8000/tcp`
- серверный firewall блокирует вход
- открывается не тот IP или порт

---

## Архитектура проекта

Ключевые модули:

- `app/core/forecasting.py` — inference по моделям
- `app/core/loaders.py` — загрузка артефактов и runtime
- `app/core/slot_pressure.py` — расчёт slot pressure
- `app/core/action_engine.py` — выбор действия
- `app/core/decision_logic.py` — сборка decision package
- `app/core/kpi.py` — KPI layer
- `app/core/file_payloads.py` — чтение `json/csv/parquet`
- `app/api/routes.py` — HTTP endpoints

Документация:

- `docs/architecture.md`
- `docs/business_logic.md`
- `docs/metrics.md`
- `docs/assumptions.md`
- `docs/demo.md`

---

## Итог

Этот проект уже готов к демонстрации и серверному запуску:

- есть web UI
- есть API
- есть Docker-упаковка
- есть health-check
- есть demo payload
- есть fallback-runtime
- есть путь к запуску на реальных `Chronos2` и `TSMixerx` 🚀
