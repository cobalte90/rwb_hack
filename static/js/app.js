function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toLocaleString("ru-RU", { maximumFractionDigits: digits });
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setHidden(id, hidden) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("hidden", hidden);
}

function serviceModeLabel(mode) {
  const labels = {
    cost_saving: "Экономия",
    balanced: "Баланс",
    sla_first: "Приоритет SLA",
  };
  return labels[mode] || mode || "—";
}

function actionLabel(action) {
  const labels = {
    call_now: "Вызвать сейчас",
    monitor: "Наблюдать",
    hold: "Удерживать",
  };
  return labels[action] || action || "—";
}

function urgencyLabel(urgency) {
  const labels = {
    low: "Низкая",
    medium: "Средняя",
    high: "Высокая",
    critical: "Критическая",
  };
  return labels[urgency] || urgency || "—";
}

function pressureLevelLabel(level) {
  const labels = {
    low: "Низкий",
    medium: "Средний",
    high: "Высокий",
    critical: "Критический",
  };
  return labels[level] || level || "—";
}

function actionBadge(action) {
  return `<span class="badge badge-${action}">${actionLabel(action)}</span>`;
}

function actionRank(action) {
  if (action === "call_now") return 3;
  if (action === "monitor") return 2;
  return 1;
}

function pressurePercent(score) {
  return Math.max(0, Math.min(100, Math.round(Number(score || 0) * 100)));
}

function loadPercent(value, maxValue) {
  if (!maxValue || Number(maxValue) <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((Number(value || 0) / Number(maxValue)) * 100)));
}

function pressureBar(score) {
  return `
    <div class="metric-bar">
      <div class="metric-bar-fill pressure-fill" style="width:${pressurePercent(score)}%"></div>
    </div>
  `;
}

function loadBar(value, maxValue) {
  return `
    <div class="metric-bar">
      <div class="metric-bar-fill load-fill" style="width:${loadPercent(value, maxValue)}%"></div>
    </div>
  `;
}

function minutesToLabel(minutes) {
  if (!minutes || minutes <= 0) return "сейчас";
  if (minutes < 60) return `через ${minutes} мин`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `через ${hours} ч ${rest} мин` : `через ${hours} ч`;
}

function shortReason(decision) {
  if (decision && Array.isArray(decision.reasons) && decision.reasons.length) {
    return decision.reasons[0];
  }
  return decision && decision.explanation ? decision.explanation : "Причина уточняется.";
}

function toDateLabel(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function nextActionLabel(decision) {
  if (!decision) return "—";
  if (decision.recommended_action === "call_now") {
    return decision.call_time ? `Вызов к ${toDateLabel(decision.call_time)}` : "Вызов сейчас";
  }
  if (decision.recommended_action === "monitor") {
    return decision.call_time ? `Проверка к ${toDateLabel(decision.call_time)}` : "Проверка через 30 мин";
  }
  return "Без действия";
}

function confidenceLabel(disagreement) {
  const value = Number(disagreement || 0);
  if (value <= 0.18) return "Высокая";
  if (value <= 0.35) return "Средняя";
  return "Осторожная";
}

function factorRows(decision) {
  const risk = (decision && decision.risk_fields) || {};
  const normalized = risk.normalized_pressure_factors || {};
  const rows = [
    {
      label: "Ближняя нагрузка",
      value: `${formatNumber(decision?.horizon_summary?.near_term_peak)} / ${formatNumber(decision?.horizon_summary?.peak_2h_load)}`,
      chip: `${Math.round(Number(normalized.near_term_peak_ratio || 0) * 100)}% от peak`,
    },
    {
      label: "Близость пика",
      value: minutesToLabel(Number(decision?.horizon_summary?.peak_step || 0) * 30),
      chip: `шаг ${decision?.horizon_summary?.peak_step || "—"}`,
    },
    {
      label: "Волатильность",
      value: formatNumber(risk.route_cv),
      chip: Number(risk.route_cv || 0) >= 1 ? "выше нормы" : "стабильно",
    },
    {
      label: "Согласие моделей",
      value: formatNumber(risk.model_disagreement),
      chip: confidenceLabel(risk.model_disagreement),
    },
  ];

  if (risk.is_friday) {
    rows.push({
      label: "Режим дня",
      value: "Пятничный пик",
      chip: "усилен буфер",
    });
  }

  return rows;
}

function buildSparkline(points, highlightCount = 4) {
  if (!points || !points.length) {
    return '<div class="sparkline-empty">Нет прогноза</div>';
  }
  const width = 340;
  const height = 96;
  const pad = 10;
  const values = points.map((item) => Number(item.y_pred || 0));
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const range = Math.max(maxValue - minValue, 1e-6);
  const stepX = (width - pad * 2) / Math.max(points.length - 1, 1);
  const coords = values.map((value, index) => {
    const x = pad + stepX * index;
    const y = height - pad - ((value - minValue) / range) * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  const peakIndex = values.indexOf(Math.max(...values));
  const peakX = pad + stepX * peakIndex;
  const peakY = height - pad - ((values[peakIndex] - minValue) / range) * (height - pad * 2);
  const highlightWidth = highlightCount > 1 ? stepX * (highlightCount - 1) + 24 : 24;

  return `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="${pad - 6}" y="${pad - 4}" width="${highlightWidth}" height="${height - pad * 2 + 8}" rx="10" class="sparkline-window"></rect>
      <polyline points="${coords}" class="sparkline-line"></polyline>
      <circle cx="${peakX}" cy="${peakY}" r="4.5" class="sparkline-peak"></circle>
    </svg>
  `;
}

document.addEventListener("DOMContentLoaded", async () => {
  const appShell = document.querySelector(".app-shell");
  const truckImage = appShell ? appShell.dataset.truckImage : "";
  const payloadJson = document.getElementById("payload-json");
  const serviceMode = document.getElementById("service-mode");
  const payloadFile = document.getElementById("payload-file");
  const loadDemoBtn = document.getElementById("load-demo-btn");
  const runManualBtn = document.getElementById("run-manual-btn");
  const routesFilter = document.getElementById("routes-filter");
  const routesSort = document.getElementById("routes-sort");

  const heroPanel = document.querySelector(".hero-panel");
  const focusBox = document.getElementById("focus-box");
  const queueCards = document.getElementById("queue-cards");
  const tableBody = document.getElementById("decision-table-body");
  const routeDetails = document.getElementById("route-details");
  const routeFocusPanel = document.getElementById("route-focus-panel");
  const kpiBox = document.getElementById("kpi-box");
  const insightsList = document.getElementById("insights-list");
  const rawResponse = document.getElementById("raw-response");
  const progressBox = document.getElementById("progress-box");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const progressStatus = document.getElementById("progress-status");

  let selectedFile = null;
  let demoPayload = null;
  let dashboard = null;
  let currentRouteId = null;
  let progressTimer = null;
  let progressValue = 0;
  let progressStageIndex = -1;

  applyRoutesHeaderV2();

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (error) {
        detail = `${detail}`;
      }
      throw new Error(detail);
    }
    return response.json();
  }

  function applyRoutesHeaderV2() {
    const headerRow = document.querySelector(".data-table thead tr");
    if (!headerRow) return;
    headerRow.innerHTML = `
      <th>Маршрут</th>
      <th>Склад</th>
      <th>Действие</th>
      <th>Уровень</th>
      <th>Нагрузка</th>
      <th>Объём</th>
      <th>Машины</th>
      <th>Срочность</th>
      <th>Следующее действие</th>
      <th>Причина</th>
    `;
  }

  function progressStages() {
    return [
      { key: "upload", label: "Загружаем входные данные", target: 18 },
      { key: "validate", label: "Проверяем формат и схему", target: 34 },
      { key: "forecast", label: "Строим прогноз по маршрутам", target: 66 },
      { key: "decision", label: "Считаем pressure и действия", target: 88 },
      { key: "ready", label: "Решение готово", target: 100 },
    ];
  }

  function updateProgressUi() {
    const stages = progressStages();
    if (progressFill) {
      progressFill.style.width = `${progressValue}%`;
    }
    if (progressPercent) {
      progressPercent.textContent = `${Math.round(progressValue)}%`;
    }
    stages.forEach((stage, index) => {
      const el = document.getElementById(`progress-stage-${stage.key}`);
      if (!el) return;
      el.classList.toggle("is-active", index === progressStageIndex);
      el.classList.toggle("is-done", index < progressStageIndex || progressValue >= 100);
    });
  }

  function setProgressStage(index) {
    const stages = progressStages();
    progressStageIndex = Math.max(0, Math.min(index, stages.length - 1));
    if (progressStatus) {
      progressStatus.textContent = stages[progressStageIndex].label;
    }
    progressValue = Math.max(progressValue, stages[progressStageIndex].target);
    updateProgressUi();
  }

  function startProgressFlow(kind) {
    progressValue = 6;
    progressStageIndex = 0;
    setHidden("progress-box", false);
    if (progressStatus) {
      progressStatus.textContent = kind === "file" ? "Загружаем и читаем файл" : "Принимаем сценарий";
    }
    updateProgressUi();
    if (progressTimer) {
      clearInterval(progressTimer);
    }
    progressTimer = setInterval(() => {
      if (progressValue < 90) {
        progressValue += progressValue < 40 ? 4 : progressValue < 70 ? 2 : 1;
      }
      if (progressValue >= 20 && progressStageIndex < 1) setProgressStage(1);
      if (progressValue >= 44 && progressStageIndex < 2) setProgressStage(2);
      if (progressValue >= 72 && progressStageIndex < 3) setProgressStage(3);
      updateProgressUi();
    }, 500);
  }

  function finishProgressFlow() {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    setProgressStage(4);
    progressValue = 100;
    updateProgressUi();
    window.setTimeout(() => {
      setHidden("progress-box", true);
    }, 900);
  }

  function resetProgressFlow() {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    progressValue = 0;
    progressStageIndex = -1;
    updateProgressUi();
    setHidden("progress-box", true);
  }

  function renderLoadingSkeleton() {
    if (heroPanel) {
      heroPanel.classList.remove("hero-critical", "hero-monitor");
    }
    focusBox.innerHTML = `
      <div class="skeleton hero-skeleton hero-skeleton-lg"></div>
      <div class="hero-metrics">
        <div class="skeleton hero-skeleton-card"></div>
        <div class="skeleton hero-skeleton-card"></div>
        <div class="skeleton hero-skeleton-card"></div>
      </div>
      <div class="skeleton hero-skeleton hero-skeleton-md"></div>
      <div class="skeleton hero-skeleton hero-skeleton-sm"></div>
    `;
    queueCards.innerHTML = `
      <div class="queue-card queue-card-empty skeleton-card">
        <div class="skeleton queue-skeleton-lg"></div>
        <div class="skeleton queue-skeleton-sm"></div>
        <div class="skeleton queue-skeleton-sm"></div>
      </div>
      <div class="queue-card queue-card-empty skeleton-card">
        <div class="skeleton queue-skeleton-lg"></div>
        <div class="skeleton queue-skeleton-sm"></div>
        <div class="skeleton queue-skeleton-sm"></div>
      </div>
    `;
    tableBody.innerHTML = `
      <tr><td colspan="9"><div class="skeleton table-skeleton"></div></td></tr>
      <tr><td colspan="9"><div class="skeleton table-skeleton"></div></td></tr>
      <tr><td colspan="9"><div class="skeleton table-skeleton"></div></td></tr>
    `;
    routeDetails.innerHTML = `
      <div class="detail-block">
        <div class="skeleton hero-skeleton hero-skeleton-md"></div>
        <div class="skeleton table-skeleton"></div>
        <div class="skeleton hero-skeleton hero-skeleton-sm"></div>
      </div>
    `;
  }

  function setLoading(isLoading) {
    loadDemoBtn.disabled = isLoading;
    runManualBtn.disabled = isLoading;
    setHidden("loading-box", !isLoading);
    if (isLoading) {
      renderLoadingSkeleton();
    }
  }

  function setError(message) {
    const errorBox = document.getElementById("error-box");
    if (!message) {
      errorBox.textContent = "";
      setHidden("error-box", true);
      return;
    }
    errorBox.textContent = message;
    setHidden("error-box", false);
  }

  function updateInputSummary(summary = {}) {
    setText("input-source", summary.source || "—");
    setText("input-file-name", summary.fileName || "—");
    setText("input-routes", summary.routes != null ? String(summary.routes) : "—");
    setText("input-records", summary.records != null ? String(summary.records) : "—");
    setText("input-horizon", summary.horizon != null ? `${summary.horizon} шагов` : "—");
  }

  function summarizeJsonPayload(rawText) {
    try {
      const payload = JSON.parse(rawText || "{}");
      const records = Array.isArray(payload.records) ? payload.records : [];
      const routeCount = new Set(records.map((item) => item.route_id)).size;
      updateInputSummary({
        source: "JSON",
        fileName: "—",
        routes: routeCount,
        records: records.length,
        horizon: payload.horizon_steps || 10,
      });
    } catch (error) {
      updateInputSummary({
        source: selectedFile ? "Файл" : "JSON",
        fileName: selectedFile ? selectedFile.name : "—",
        routes: null,
        records: null,
        horizon: 10,
      });
    }
  }

  function renderMeta(meta) {
    setText("artifact-report", meta.artifact_report_path || "—");
    if (meta.default_service_mode) {
      serviceMode.value = meta.default_service_mode;
    }
    updateInputSummary({
      source: "Демо",
      fileName: "—",
      routes: null,
      records: null,
      horizon: 10,
    });
  }

  function renderTopKpis(data) {
    const decisions = data.decision_packages || [];
    const criticalRoutes = decisions.filter((item) => item.slot_pressure_level === "critical" || item.recommended_action === "call_now").length;
    const trucksNow = decisions
      .filter((item) => item.recommended_action === "call_now")
      .reduce((acc, item) => acc + Number(item.recommended_trucks || 0), 0);
    const avgPressure = data.overview ? data.overview.avg_pressure : null;
    const mode = data.overview ? data.overview.service_mode : null;
    const slaRisk = data.kpi_snapshot ? data.kpi_snapshot.slot_overload_rate : null;
    const earliestPeakStep = decisions.length
      ? Math.min(...decisions.map((item) => Number(item.horizon_summary.peak_step || 0)).filter((value) => value > 0))
      : null;

    setText("kpi-critical-routes", String(criticalRoutes));
    setText("kpi-trucks-now", String(trucksNow));
    setText("kpi-avg-pressure", formatNumber(avgPressure));
    setText("kpi-peak-window", earliestPeakStep ? minutesToLabel(earliestPeakStep * 30) : "—");
    setText("kpi-service-mode", serviceModeLabel(mode));
    setText("kpi-sla-risk", formatPercent(slaRisk));
  }

  function renderAnalytics(data) {
    const snapshot = data.kpi_snapshot || {};
    setText("analytics-utilization", formatPercent(snapshot.expected_utilization));
    setText("analytics-under-call", formatPercent(snapshot.under_call_rate));
    setText("analytics-over-call", formatPercent(snapshot.over_call_rate));
    setText("analytics-stability", formatPercent(snapshot.decision_stability));

    if (kpiBox) {
      kpiBox.innerHTML = [
        ["under_call_rate", snapshot.under_call_rate],
        ["over_call_rate", snapshot.over_call_rate],
        ["slot_overload_rate", snapshot.slot_overload_rate],
        ["expected_utilization", snapshot.expected_utilization],
        ["decision_stability", snapshot.decision_stability],
      ].map(([label, value]) => (
        `<div class="meta-row"><span>${label}</span><strong>${formatPercent(value)}</strong></div>`
      )).join("");
    }
  }

  function buildInsights(decisions, overview, kpiSnapshot) {
    if (!decisions.length) {
      return ["Нет данных для вывода бизнес-инсайтов."];
    }
    const callNow = decisions.filter((item) => item.recommended_action === "call_now").length;
    const monitor = decisions.filter((item) => item.recommended_action === "monitor").length;
    const highUrgency = decisions.filter((item) => ["high", "critical"].includes(item.urgency)).length;
    const peakWindowMinutes = Math.min(...decisions.map((item) => Number(item.horizon_summary.peak_step || 0)).filter((value) => value > 0)) * 30;
    const insights = [
      callNow ? `Немедленное действие требуется по ${callNow} маршрут(ам).` : "Критических вызовов транспорта сейчас нет.",
      monitor ? `${monitor} маршрут(ов) нужно держать под мониторингом и пересчитать в следующем окне.` : "Маршрутов под наблюдением сейчас нет.",
      highUrgency ? `${highUrgency} маршрут(ов) имеют высокий или критический уровень срочности.` : "Высокая срочность по маршрутам не выявлена.",
      peakWindowMinutes ? `Ближайшее пиковое окно ожидается ${minutesToLabel(peakWindowMinutes)}.` : "Пиковое окно не определено.",
      `Риск SLA оценивается как ${formatPercent(kpiSnapshot.slot_overload_rate)}.`,
      `Режим сервиса: ${serviceModeLabel(overview.service_mode)}.`,
    ];
    return insights.slice(0, 5);
  }

  function rankedDecisions() {
    if (!dashboard || !dashboard.decision_packages) return [];
    return [...dashboard.decision_packages].sort((a, b) => (
      actionRank(b.recommended_action) - actionRank(a.recommended_action)
      || Number(b.slot_pressure_score || 0) - Number(a.slot_pressure_score || 0)
      || Number(b.recommended_trucks || 0) - Number(a.recommended_trucks || 0)
      || Number(a.route_id) - Number(b.route_id)
    ));
  }

  function forecastForRoute(routeId) {
    return ((dashboard && dashboard.forecast_by_route) || {})[String(routeId)] || [];
  }

  function openRouteFocus(routeId, scroll = false) {
    currentRouteId = Number(routeId);
    renderRouteDetailsV2(currentRouteId);
    if (scroll && routeFocusPanel) {
      routeFocusPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      routeFocusPanel.classList.remove("focus-flash");
      window.setTimeout(() => routeFocusPanel.classList.add("focus-flash"), 20);
      window.setTimeout(() => routeFocusPanel.classList.remove("focus-flash"), 1400);
    }
  }

  function renderFocus(data) {
    const decisions = rankedDecisions();
    if (!decisions.length) {
      if (heroPanel) {
        heroPanel.classList.remove("hero-critical", "hero-monitor");
      }
      focusBox.textContent = "После запуска расчёта здесь появится приоритетный маршрут.";
      return;
    }
    const top = decisions[0];
    const rank = 1;
    const peakInMinutes = Number(top.horizon_summary.peak_step || 0) * 30;
    const forecast = forecastForRoute(top.route_id);
    const mainReason = shortReason(top);

    if (heroPanel) {
      heroPanel.classList.toggle("hero-critical", top.recommended_action === "call_now");
      heroPanel.classList.toggle("hero-monitor", top.recommended_action === "monitor");
    }

    focusBox.innerHTML = `
      <div class="hero-route-row">
        <div>
          <div class="hero-rank">Маршрут №${rank}</div>
          <div class="hero-route-name">Маршрут ${top.route_id}</div>
          <div class="hero-subline">Склад ${top.warehouse_id} · ${pressureLevelLabel(top.slot_pressure_level)} уровень</div>
        </div>
        ${actionBadge(top.recommended_action)}
      </div>

      <div class="hero-metrics">
        <div class="hero-metric">
          <span>Нагрузка слота</span>
          <strong>${formatNumber(top.slot_pressure_score)}</strong>
          ${pressureBar(top.slot_pressure_score)}
        </div>
        <div class="hero-metric">
          <span>Пик</span>
          <strong>${minutesToLabel(peakInMinutes)}</strong>
          <div class="hero-secondary">Шаг ${top.horizon_summary.peak_step}</div>
        </div>
        <div class="hero-metric">
          <span>Машины</span>
          <strong>${top.recommended_trucks}</strong>
          <div class="hero-secondary">Срочность: ${urgencyLabel(top.urgency)}</div>
        </div>
      </div>

      <div class="hero-content">
        <div class="hero-visual">
          <img class="truck-hero" src="${truckImage}" alt="Грузовик Wildberries">
        </div>
        <div class="hero-side">
          <div class="hero-reason-box">
            <div class="hero-reason-title">Главная причина приоритета</div>
            <p>${mainReason}</p>
          </div>
          <div class="hero-chart-box">
            <div class="hero-chart-title">Ближайший прогноз</div>
            ${buildSparkline(forecast, 4)}
          </div>
        </div>
      </div>

      <div class="why-box">
        <div class="why-title">Почему система выбрала это действие</div>
        <ul class="why-list">
          ${(top.reasons || []).slice(0, 4).map((reason) => `<li>${reason}</li>`).join("")}
        </ul>
      </div>
    `;
  }

  function renderQueue(data) {
    const decisions = rankedDecisions().slice(0, 5);
    if (!decisions.length) {
      queueCards.innerHTML = '<div class="queue-card queue-card-empty">После расчёта здесь появятся действия по маршрутам.</div>';
      return;
    }

    queueCards.innerHTML = decisions.map((decision) => `
      <button type="button" class="queue-card queue-${decision.recommended_action} open-queue-card" data-route-id="${decision.route_id}">
        <div class="queue-head">
          <div>
            <div class="queue-route">Маршрут ${decision.route_id}</div>
            <div class="queue-sub">Склад ${decision.warehouse_id}</div>
          </div>
          ${actionBadge(decision.recommended_action)}
        </div>
        <div class="queue-metrics">
          <span>Машины: <strong>${decision.recommended_trucks}</strong></span>
          <span>Срочность: <strong>${urgencyLabel(decision.urgency)}</strong></span>
          <span>Нагрузка: <strong>${formatNumber(decision.slot_pressure_score)}</strong></span>
          <span>${nextActionLabel(decision)}</span>
        </div>
        <p class="queue-text">${shortReason(decision)}</p>
      </button>
    `).join("");

    queueCards.querySelectorAll(".open-queue-card").forEach((button) => {
      button.addEventListener("click", () => openRouteFocus(button.dataset.routeId, true));
    });
  }

  function filteredSortedDecisions() {
    const decisions = rankedDecisions();
    const filter = routesFilter ? routesFilter.value : "all";
    const sort = routesSort ? routesSort.value : "priority";
    let items = decisions;

    if (filter && filter !== "all") {
      items = items.filter((item) => item.recommended_action === filter);
    }

    if (sort === "pressure") {
      items = [...items].sort((a, b) => Number(b.slot_pressure_score || 0) - Number(a.slot_pressure_score || 0));
    } else if (sort === "trucks") {
      items = [...items].sort((a, b) => Number(b.recommended_trucks || 0) - Number(a.recommended_trucks || 0));
    } else if (sort === "route") {
      items = [...items].sort((a, b) => Number(a.route_id) - Number(b.route_id));
    }

    return items;
  }

  function renderRoutesTable() {
    const decisions = filteredSortedDecisions();
    if (!decisions.length) {
      tableBody.innerHTML = '<tr><td colspan="9" class="empty-cell">Нет маршрутов под выбранный фильтр.</td></tr>';
      return;
    }

    const maxPeak = Math.max(...decisions.map((item) => Number(item.horizon_summary.peak_2h_load || 0)), 0);
    tableBody.innerHTML = decisions.map((item) => `
      <tr>
        <td>${item.route_id}</td>
        <td>${item.warehouse_id}</td>
        <td>${actionBadge(item.recommended_action)}</td>
        <td>
          <div class="table-metric">
            <strong>${formatNumber(item.slot_pressure_score)}</strong>
            ${pressureBar(item.slot_pressure_score)}
          </div>
        </td>
        <td>
          <div class="table-metric">
            <strong>${formatNumber(item.horizon_summary.peak_2h_load)}</strong>
            ${loadBar(item.horizon_summary.peak_2h_load, maxPeak)}
          </div>
        </td>
        <td>${item.recommended_trucks}</td>
        <td>${urgencyLabel(item.urgency)}</td>
        <td class="table-reason">${shortReason(item)}</td>
        <td><button class="btn btn-small open-route-btn" data-route-id="${item.route_id}" type="button">Открыть</button></td>
      </tr>
    `).join("");

    tableBody.querySelectorAll(".open-route-btn").forEach((button) => {
      button.addEventListener("click", () => openRouteFocus(button.dataset.routeId, true));
    });
  }

  function renderRouteDetails(routeId) {
    if (!dashboard) return;
    const decisions = dashboard.decision_packages || [];
    const decision = decisions.find((item) => Number(item.route_id) === Number(routeId));
    if (!decision) {
      routeDetails.innerHTML = '<div class="muted">Маршрут не найден.</div>';
      return;
    }

    const maxPeak = dashboard.overview ? Number(dashboard.overview.peak_warehouse_load || 0) : 0;
    const forecastRows = forecastForRoute(routeId)
      .map((item) => `<tr><td>h${item.step}</td><td>${item.timestamp}</td><td>${formatNumber(item.y_pred)}</td></tr>`)
      .join("");

    routeDetails.innerHTML = `
      <div class="detail-block">
        <div class="detail-head">
          <img class="truck-detail" src="${truckImage}" alt="Грузовик">
          <div>
            <div class="detail-title">Маршрут ${decision.route_id}</div>
            <div class="detail-subtitle">Склад ${decision.warehouse_id} · режим ${serviceModeLabel(decision.service_mode)}</div>
          </div>
          ${actionBadge(decision.recommended_action)}
        </div>

        <div class="detail-grid">
          <div><span>Нагрузка слота</span><strong>${formatNumber(decision.slot_pressure_score)}</strong>${pressureBar(decision.slot_pressure_score)}</div>
          <div><span>Пиковый объём</span><strong>${formatNumber(decision.horizon_summary.peak_2h_load)}</strong>${loadBar(decision.horizon_summary.peak_2h_load, maxPeak)}</div>
          <div><span>Машины</span><strong>${decision.recommended_trucks}</strong></div>
          <div><span>Срочность</span><strong>${urgencyLabel(decision.urgency)}</strong></div>
          <div><span>Ближний пик</span><strong>${formatNumber(decision.horizon_summary.near_term_peak)}</strong></div>
          <div><span>Шаг пика</span><strong>${decision.horizon_summary.peak_step}</strong></div>
        </div>

        <div class="detail-chart">
          <div class="detail-chart-title">Прогноз на 10 шагов</div>
          ${buildSparkline(forecastForRoute(routeId), 4)}
        </div>

        <div class="why-box compact">
          <div class="why-title">Почему система предлагает именно это</div>
          <ul class="why-list">
            ${(decision.reasons || []).slice(0, 4).map((reason) => `<li>${reason}</li>`).join("")}
          </ul>
        </div>

        <div class="table-wrap">
          <table class="data-table data-table-compact">
            <thead><tr><th>Шаг</th><th>Время</th><th>Прогноз</th></tr></thead>
            <tbody>${forecastRows || '<tr><td colspan="3">Нет строк прогноза</td></tr>'}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  function renderFocusV2() {
    const decisions = rankedDecisions();
    if (!decisions.length) {
      if (heroPanel) {
        heroPanel.classList.remove("hero-critical", "hero-monitor");
      }
      focusBox.textContent = "После запуска расчёта здесь появится приоритетный маршрут.";
      return;
    }

    const top = decisions[0];
    const factors = factorRows(top);
    const peakInMinutes = Number(top.horizon_summary.peak_step || 0) * 30;

    if (heroPanel) {
      heroPanel.classList.toggle("hero-critical", top.recommended_action === "call_now");
      heroPanel.classList.toggle("hero-monitor", top.recommended_action === "monitor");
    }

    focusBox.innerHTML = `
      <div class="hero-route-row">
        <div>
          <div class="hero-rank">Маршрут №1</div>
          <div class="hero-route-name">Маршрут ${top.route_id}</div>
          <div class="hero-subline">Склад ${top.warehouse_id} · ${pressureLevelLabel(top.slot_pressure_level)} уровень · ${nextActionLabel(top)}</div>
        </div>
        <div class="hero-badge-wrap">
          ${actionBadge(top.recommended_action)}
          <span class="hero-pressure-pill">${pressureLevelLabel(top.slot_pressure_level)}</span>
        </div>
      </div>

      <div class="hero-metrics">
        <div class="hero-metric">
          <span>Нагрузка слота</span>
          <strong>${formatNumber(top.slot_pressure_score)}</strong>
          ${pressureBar(top.slot_pressure_score)}
        </div>
        <div class="hero-metric">
          <span>Пик</span>
          <strong>${minutesToLabel(peakInMinutes)}</strong>
          <div class="hero-secondary">Шаг ${top.horizon_summary.peak_step}</div>
        </div>
        <div class="hero-metric">
          <span>Машины</span>
          <strong>${top.recommended_trucks}</strong>
          <div class="hero-secondary">Срочность: ${urgencyLabel(top.urgency)}</div>
        </div>
      </div>

      <div class="hero-content">
        <div class="hero-visual">
          <img class="truck-hero" src="${truckImage}" alt="Грузовик Wildberries">
        </div>
        <div class="hero-side">
          <div class="hero-reason-box">
            <div class="hero-reason-title">Главная причина приоритета</div>
            <p>${shortReason(top)}</p>
          </div>
          <div class="hero-chart-box">
            <div class="hero-chart-title">Ближайший прогноз</div>
            ${buildSparkline(forecastForRoute(top.route_id), 4)}
          </div>
        </div>
      </div>

      <div class="factor-grid">
        ${factors.map((factor) => `
          <div class="factor-card">
            <span class="factor-label">${factor.label}</span>
            <strong>${factor.value}</strong>
            <span class="factor-chip">${factor.chip}</span>
          </div>
        `).join("")}
      </div>

      <div class="why-box">
        <div class="why-title">Почему система выбрала это действие</div>
        <ul class="why-list">
          ${(top.reasons || []).slice(0, 4).map((reason) => `<li>${reason}</li>`).join("")}
        </ul>
      </div>
    `;
  }

  function renderQueueV2() {
    const decisions = rankedDecisions().slice(0, 5);
    if (!decisions.length) {
      queueCards.innerHTML = '<div class="queue-card queue-card-empty">После расчёта здесь появятся действия по маршрутам.</div>';
      return;
    }

    queueCards.innerHTML = decisions.map((decision) => `
      <button type="button" class="queue-card queue-${decision.recommended_action} open-queue-card" data-route-id="${decision.route_id}">
        <div class="queue-head">
          <div>
            <div class="queue-route">Маршрут ${decision.route_id}</div>
            <div class="queue-sub">Склад ${decision.warehouse_id}</div>
          </div>
          ${actionBadge(decision.recommended_action)}
        </div>
        <div class="queue-metrics">
          <span>Машины: <strong>${decision.recommended_trucks}</strong></span>
          <span>Срочность: <strong>${urgencyLabel(decision.urgency)}</strong></span>
          <span>Pressure: <strong>${formatNumber(decision.slot_pressure_score)}</strong></span>
          <span>${nextActionLabel(decision)}</span>
        </div>
        <div class="queue-pressure-row">
          <span class="queue-level">${pressureLevelLabel(decision.slot_pressure_level)}</span>
          ${pressureBar(decision.slot_pressure_score)}
        </div>
        <p class="queue-text">${shortReason(decision)}</p>
      </button>
    `).join("");

    queueCards.querySelectorAll(".open-queue-card").forEach((button) => {
      button.addEventListener("click", () => openRouteFocus(button.dataset.routeId, true));
    });
  }

  function renderRoutesTableV2() {
    const decisions = filteredSortedDecisions();
    if (!decisions.length) {
      tableBody.innerHTML = '<tr><td colspan="10" class="empty-cell">Нет маршрутов под выбранный фильтр.</td></tr>';
      return;
    }

    const maxPeak = Math.max(...decisions.map((item) => Number(item.horizon_summary.peak_2h_load || 0)), 0);
    tableBody.innerHTML = decisions.map((item) => `
      <tr class="route-row-clickable" data-route-id="${item.route_id}">
        <td>${item.route_id}</td>
        <td>${item.warehouse_id}</td>
        <td>${actionBadge(item.recommended_action)}</td>
        <td>${pressureLevelLabel(item.slot_pressure_level)}</td>
        <td>
          <div class="table-metric">
            <strong>${formatNumber(item.slot_pressure_score)}</strong>
            ${pressureBar(item.slot_pressure_score)}
          </div>
        </td>
        <td>
          <div class="table-metric">
            <strong>${formatNumber(item.horizon_summary.peak_2h_load)}</strong>
            ${loadBar(item.horizon_summary.peak_2h_load, maxPeak)}
          </div>
        </td>
        <td>${item.recommended_trucks}</td>
        <td>${urgencyLabel(item.urgency)}</td>
        <td>${nextActionLabel(item)}</td>
        <td class="table-reason">${shortReason(item)}</td>
      </tr>
    `).join("");

    tableBody.querySelectorAll(".route-row-clickable").forEach((row) => {
      row.addEventListener("click", () => openRouteFocus(row.dataset.routeId, true));
    });
  }

  function renderRouteDetailsV2(routeId) {
    if (!dashboard) return;
    const decisions = dashboard.decision_packages || [];
    const decision = decisions.find((item) => Number(item.route_id) === Number(routeId));
    if (!decision) {
      routeDetails.innerHTML = '<div class="muted">Маршрут не найден.</div>';
      return;
    }

    const factors = factorRows(decision);
    const risk = decision.risk_fields || {};
    const maxPeak = dashboard.overview ? Number(dashboard.overview.peak_warehouse_load || 0) : 0;
    const forecastRows = forecastForRoute(routeId)
      .map((item) => `<tr><td>h${item.step}</td><td>${item.timestamp}</td><td>${formatNumber(item.y_pred)}</td></tr>`)
      .join("");

    routeDetails.innerHTML = `
      <div class="detail-block">
        <div class="detail-head">
          <img class="truck-detail" src="${truckImage}" alt="Грузовик">
          <div>
            <div class="detail-title">Маршрут ${decision.route_id}</div>
            <div class="detail-subtitle">Склад ${decision.warehouse_id} · режим ${serviceModeLabel(decision.service_mode)}</div>
          </div>
          ${actionBadge(decision.recommended_action)}
        </div>

        <div class="detail-grid">
          <div><span>Нагрузка слота</span><strong>${formatNumber(decision.slot_pressure_score)}</strong>${pressureBar(decision.slot_pressure_score)}</div>
          <div><span>Пиковый объём</span><strong>${formatNumber(decision.horizon_summary.peak_2h_load)}</strong>${loadBar(decision.horizon_summary.peak_2h_load, maxPeak)}</div>
          <div><span>Машины</span><strong>${decision.recommended_trucks}</strong></div>
          <div><span>Срочность</span><strong>${urgencyLabel(decision.urgency)}</strong></div>
          <div><span>Следующее действие</span><strong>${nextActionLabel(decision)}</strong></div>
          <div><span>Доверие</span><strong>${confidenceLabel(risk.model_disagreement)}</strong></div>
        </div>

        <div class="factor-grid factor-grid-compact">
          ${factors.map((factor) => `
            <div class="factor-card">
              <span class="factor-label">${factor.label}</span>
              <strong>${factor.value}</strong>
              <span class="factor-chip">${factor.chip}</span>
            </div>
          `).join("")}
        </div>

        <div class="detail-chart">
          <div class="detail-chart-title">Прогноз на 10 шагов</div>
          ${buildSparkline(forecastForRoute(routeId), 4)}
        </div>

        <div class="why-box compact">
          <div class="why-title">Почему система предлагает именно это</div>
          <ul class="why-list">
            ${(decision.reasons || []).slice(0, 4).map((reason) => `<li>${reason}</li>`).join("")}
          </ul>
        </div>

        <div class="table-wrap">
          <table class="data-table data-table-compact">
            <thead><tr><th>Шаг</th><th>Время</th><th>Прогноз</th></tr></thead>
            <tbody>${forecastRows || '<tr><td colspan="3">Нет строк прогноза</td></tr>'}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  function renderInsights(data) {
    const insights = buildInsights(data.decision_packages || [], data.overview || {}, data.kpi_snapshot || {});
    insightsList.innerHTML = insights.map((item) => `<li>${item}</li>`).join("");
  }

  function renderDashboard(data) {
    dashboard = data;
    finishProgressFlow();
    renderTopKpis(data);
    renderAnalytics(data);
    renderFocusV2();
    renderQueueV2();
    renderRoutesTableV2();
    renderInsights(data);
    rawResponse.textContent = JSON.stringify(data, null, 2);

    const routeCount = (data.overview && data.overview.routes_in_batch) || 0;
    const recordCount = (data.overview && data.overview.records_in_batch) || 0;
    updateInputSummary({
      source: selectedFile ? "Файл" : "Запрос",
      fileName: selectedFile ? selectedFile.name : "JSON / демо",
      routes: routeCount,
      records: recordCount,
      horizon: 10,
    });

    const top = rankedDecisions()[0];
    if (top) {
      openRouteFocus(top.route_id, false);
    }
  }

  async function submitPayload(payload) {
    startProgressFlow("json");
    setLoading(true);
    setError("");
    try {
      const data = await fetchJson("/ui/plan-dashboard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderDashboard(data);
    } catch (error) {
      resetProgressFlow();
      setError(`Сценарий не выполнился: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function submitFile(file) {
    startProgressFlow("file");
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_profile", "latest_lb");
      formData.append("horizon_steps", "10");
      formData.append("service_mode", serviceMode.value);
      const response = await fetch("/ui/plan-dashboard-file", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const payload = await response.json();
          detail = payload.detail || detail;
        } catch (error) {
          detail = `${detail}`;
        }
        throw new Error(detail);
      }
      const data = await response.json();
      renderDashboard(data);
    } catch (error) {
      resetProgressFlow();
      setError(`Файл не обработан: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  loadDemoBtn.addEventListener("click", async () => {
    selectedFile = null;
    payloadFile.value = "";
    if (!demoPayload) {
      setError("Демо-набор ещё не загружен.");
      return;
    }
    payloadJson.value = JSON.stringify(demoPayload, null, 2);
    summarizeJsonPayload(payloadJson.value);
    await submitPayload(JSON.parse(JSON.stringify(demoPayload)));
  });

  runManualBtn.addEventListener("click", async () => {
    if (selectedFile) {
      await submitFile(selectedFile);
      return;
    }
    try {
      const payload = JSON.parse(payloadJson.value || "{}");
      payload.planning_config_override = payload.planning_config_override || {};
      payload.planning_config_override.service_mode = serviceMode.value;
      await submitPayload(payload);
    } catch (error) {
      setError("Не удалось распарсить JSON-запрос.");
    }
  });

  payloadFile.addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    selectedFile = file;
    updateInputSummary({
      source: "Файл",
      fileName: file.name,
      routes: null,
      records: null,
      horizon: 10,
    });
    const lowerName = (file.name || "").toLowerCase();
    if (lowerName.endsWith(".json")) {
      payloadJson.value = await file.text();
      summarizeJsonPayload(payloadJson.value);
    } else {
      payloadJson.value = `Выбран файл: ${file.name}\nФормат будет обработан сервером автоматически.`;
    }
  });

  if (routesFilter) {
    routesFilter.addEventListener("change", () => {
      if (dashboard) renderRoutesTableV2();
    });
  }

  if (routesSort) {
    routesSort.addEventListener("change", () => {
      if (dashboard) renderRoutesTableV2();
    });
  }

  if (payloadJson) {
    payloadJson.addEventListener("input", () => {
      if (!selectedFile) summarizeJsonPayload(payloadJson.value);
    });
  }

  try {
    const meta = await fetchJson("/ui/meta");
    renderMeta(meta);
  } catch (error) {
    setError(`Не удалось загрузить meta: ${error.message}`);
  }

  try {
    demoPayload = await fetchJson("/ui/demo-payload");
    payloadJson.value = JSON.stringify(demoPayload, null, 2);
    summarizeJsonPayload(payloadJson.value);
  } catch (error) {
    setError(`Не удалось загрузить демо-набор: ${error.message}`);
  }
});
