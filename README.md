# SAVE A SLOT

Операционный сервис для склада, который не просто строит прогноз, а переводит его в понятное решение по маршрутам на ближайшие временные слоты. 🚚

Сервис:

- принимает входные данные в форматах `json`, `csv`, `parquet`
- прогнозирует нагрузку `target_2h` на горизонте `10` шагов по `30` минут
- считает `slot pressure`
- рекомендует действие: `call_now`, `monitor`, `hold`
- оценивает число машин, срочность и риск
- отдаёт web UI, API и explainability-слой

Продуктовый поток:

`Input data -> Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package`

---

## Что Решает Продукт

Сервис помогает команде склада и транспорта раньше видеть риск перегрузки слотов и не реагировать постфактум.

На практике это значит:

- меньше срывов SLA на пике
- меньше недовызова и перевызова машин
- раньше виден рост давления по маршруту
- решение приходит в виде действия, а не только числа прогноза
- у операционной команды появляется единый сценарий: увидеть риск, понять причину и принять решение в одном интерфейсе

---

## Почему Решение Сильное

1. Это не просто forecasting-сервис, а полноценный `decision engine`.
2. Прогноз сразу переводится в действие: `call_now`, `monitor`, `hold`.
3. Используется ансамбль моделей, а не одна модель с единой точкой отказа.
4. Есть управляемые бизнес-режимы: `cost_saving`, `balanced`, `sla_first`.
5. Система умеет работать через controlled fallback, если часть real-runtime недоступна.
6. `/health` честно показывает, используются реальные модели или proxy.
7. Есть удобный UI для демо и операционной работы.
8. Поддержаны `json`, `csv`, `parquet`, включая upload через интерфейс.
9. Сервис упакован в `FastAPI + Docker`, готов к локальному запуску и серверному деплою.
10. Архитектура отделяет offline-подготовку артефактов от online-инференса, поэтому runtime стартует быстро и стабильно.

---

## Технологии

### Backend

- `FastAPI`  
  Быстрый и удобный API-слой с автодокументацией через Swagger.

- `Pydantic` / typed schemas  
  Даёт контролируемые контракты данных и меньше сюрпризов при обработке payload.

- `Uvicorn`  
  Лёгкий ASGI runtime для локального и серверного запуска.

### Forecasting Stack

- `Chronos2`  
  Сильная foundation-модель для time series, хорошо работает как real-runtime путь для общего прогноза.

- `GRU`  
  Быстрая и практичная нейросетевая ветка, даёт устойчивый локальный сигнал.

- `TSMixerx` через `NeuralForecast`  
  Используется как lightweight residual branch и помогает добирать качество на остатках.

- `groupwise blend`  
  Финальный прогноз строится не грубым усреднением, а контролируемым late blend.

### Fallback / Supporting Models

- `TFT proxy`
- `Optuna`
- `CatBoost`
- `LightGBM`
- `Ridge stack`
- `TimeXer / Timexer`
- `NHITS`

Это важно не только как R&D-история, но и как инженерная устойчивость решения: проект умеет жить не только в “идеальном” режиме.

### Product / Infra

- `Docker` и `docker compose`
- health-check и runtime status
- `pytest`
- self-contained `artifacts/` bundle

---

## Почему Эти Технологии Хорошо Сочетаются

- `Chronos2` даёт сильную глобальную time-series основу.
- `GRU` добавляет быструю и предсказуемую локальную ветку.
- `TSMixerx` усиливает ensemble как residual branch без чрезмерной тяжести.
- decision-слой поверх прогноза делает решение полезным для бизнеса, а не только для ML-метрики.
- `FastAPI` и Docker позволяют быстро показать решение в демо, пилоте и серверном окружении.

---

## Архитектура Решения

### Offline Layer

Источник истины остаётся в `info_for_codex/`.

Основные задачи offline-слоя:

- копирование и нормализация артефактов в `artifacts/`
- сбор route / office / time-profile статистик
- экспорт runtime bundle для `Chronos2`
- подготовка residual-ветки `TSMixerx`
- обучение reproducible proxy-моделей
- сохранение blend-конфига, бизнес-правил и provenance report

Ключевой скрипт:

- `scripts/build_artifacts.py`

### Runtime Layer

Ключевые модули:

- `app/core/loaders.py` — загрузка runtime и артефактов
- `app/core/preprocessing.py` — подготовка входных данных
- `app/core/forecasting.py` — inference по моделям
- `app/core/blending.py` — комбинирование прогнозов
- `app/core/slot_pressure.py` — расчёт `pressure_score` и `pressure_level`
- `app/core/action_engine.py` — выбор действия и буфера
- `app/core/decision_logic.py` — сборка финального decision package
- `app/core/kpi.py` — KPI слой
- `app/api/routes.py` — HTTP endpoints

### Profiles

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

## Бизнес-Логика

Система не суммирует весь горизонт в одно число, а интерпретирует forecast через ключевые признаки:

- `peak_2h_load = max(y_pred[1..10])`
- `near_term_peak = max(y_pred[1..4])`
- `peak_step = argmax(y_pred[1..10])`

`Slot Pressure Engine` учитывает:

- near-term peak intensity
- peak proximity
- route volatility
- ensemble disagreement
- Friday regime
- service mode signal

`Action Engine` использует это для выбора:

- `call_now`
- `monitor`
- `hold`

и дополнительно оценивает:

- recommended trucks
- urgency
- call time
- explanation

Для бизнеса это означает, что сервис отвечает не на абстрактный вопрос “какой будет forecast?”, а на более полезный вопрос:

- где именно вырастет давление по слотам
- насколько скоро случится пик
- нужно ли звать машины прямо сейчас
- сколько машин разумно запрашивать с учётом буфера и режима сервиса

---

## Бизнес-Допущения И Границы Применимости

Чтобы решение было быстрым, воспроизводимым и пригодным для демо и пилота, в нём зафиксирован ряд допущений.

### Операционные допущения

- все машины считаются взаимозаменяемыми по вместимости и описываются единым параметром `truck_capacity`
- модель не различает типы кузова, ограничения по габаритам, подрядчиков и индивидуальную экономику рейса
- `route_id -> office_from_id` трактуется как детерминированное соответствие, потому что таким оно выглядит в предоставленных данных
- `target_2h` интерпретируется как rolling two-hour load и используется как основной сигнал давления по слоту

### Продуктовые допущения

- главная история продукта — это оркестрация слотов и решений по вызову транспорта, а не “чистый ML-бенчмарк”
- runtime ничего не обучает на старте и работает только по заранее собранным артефактам
- KPI-слой является forecast-driven proxy, потому что production-логов фактического исполнения и dispatch feedback в исходных материалах нет

### R&D и восстановление артефактов

- `blend_best_timexer_main.csv` используется как frozen leaderboard reference
- точный код для `blend_best_chronos_groupwise_main.csv` не был полностью восстановлен из `coding.ipynb`, поэтому в репозитории сохранён воспроизводимый anchor proxy
- `coding.ipynb` явно ссылается на `Chronos-2`, поэтому runtime экспортирует именно это pretrained family
- явного training-блока для `TSMixerx` в ноутбуке не было; из-за ограничений железа TimeXer заменён на совместимую lightweight residual branch на базе `TSMixerx`
- proxy-ветки для `Chronos` и `TimeXer` остаются в проекте как controlled fallback для окружений, где real-runtime ещё не собран до конца

### Что Это Значит Для Бизнеса

Эти допущения не мешают показать ценность решения, но задают рамки пилота:

- решение хорошо подходит для приоритизации, мониторинга и оперативного вызова транспорта
- решение пока не заменяет полноценный fleet optimization с разными типами машин и реальной стоимостью рейса
- KPI подходят для демо, защиты и валидации логики, но не заменяют production replay analytics

---

## Что Есть В UI

Интерфейс сделан не как технодемка, а как операционная панель.

На сайте есть:

- KPI strip
- service mode selector
- кнопки `Загрузить демо` и `Запустить расчёт`
- upload файла
- `Приоритетное решение`
- `Очередь действий`
- таблица маршрутов
- блок `Маршрут в фокусе`
- insights и технические детали

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

## Быстрый Старт Локально

### 1. Клонировать репозиторий

```bash
git clone https://github.com/cobalte90/rwb_hack.git
cd rwb_hack
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Подготовить артефакты

Нужны директории:

- `./artifacts/configs`
- `./artifacts/models`
- `./artifacts/stats`

Если артефакты лежат не в этом репозитории, можно указать путь через переменную окружения `ARTIFACTS_DIR`.

Пример для PowerShell:

```powershell
$env:ARTIFACTS_DIR="C:\path\to\artifacts"
```

Пример для bash:

```bash
export ARTIFACTS_DIR=/absolute/path/to/artifacts
```

### 4. Запустить сервис

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Открыть

- UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

---

## Запуск Через Docker

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

## Деплой На Сервер

Ниже сценарий, который был проверен на реальном сервере и поднимает сервис без proxy fallback. ✅

### 1. Подготовить сервер

Нужно:

- Ubuntu/Linux сервер
- установленный Docker
- установленный Docker Compose
- открытый порт `8000/tcp`

### 2. Клонировать репозиторий

```bash
git clone https://github.com/cobalte90/rwb_hack.git /opt/rwb_hack
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

## Как Показать Демо

Самый быстрый путь:

1. Открыть UI
2. Нажать `Загрузить демо`
3. Нажать `Запустить расчёт`

Ещё один удобный вариант:

- загрузить `examples/test_first_100_routes.parquet`

Это укороченный demo-набор, на котором удобно быстро проверять UI и расчёт.

---

## Форматы Входных Данных

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

## Полезные Команды

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

## Если Что-То Пошло Не Так

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
- в отдельных сетях может понадобиться VPN из-за ограничений временного хостинга

---

## Документация В Репозитории

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

Публичная ссылка на сервис:

- `http://85.239.60.13:8000`

Если у вас нестабильно открывается сайт без VPN, это, скорее всего, связано с ограничениями временного хостинга или маршрутизацией до сервера, а не с логикой самого приложения.
