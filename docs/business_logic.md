# Business Logic

## Forecast Interpretation

`target_2h` already represents rolling two-hour load. The system therefore does not sum all horizon points into a single demand number.

The key route summary is:

- `peak_2h_load = max(y_pred[1..10])`
- `near_term_peak = max(y_pred[1..4])`
- `peak_step = argmax(y_pred[1..10])`

The forecast ensemble now prefers this runtime stack:

- `Chronos2` real inference from the model referenced in `coding.ipynb`
- real `GRU`
- real `TSMixerx` as the lightweight residual branch
- exact late blend weights `0.96 / 0.95 / 0.93`

If the heavyweight artifacts are not present, the service falls back to proxy branches and marks the profile as degraded in `/health`.

## Slot Pressure Score

The Slot Pressure Engine combines normalized signals:

- near-term peak intensity
- peak proximity
- route volatility
- ensemble disagreement
- Friday regime
- service mode signal

Conceptually:

`pressure_score = weighted_sum(signals) + mode_pressure_bias`

The score is clipped to `[0, 1]`.

Pressure levels:

- `low`
- `medium`
- `high`
- `critical`

## Service Modes

Configured in `artifacts/configs/business_rules.yaml`.

### `cost_saving`

- lower pressure sensitivity
- smaller safety buffers
- higher dispatch threshold
- lower urgency bias

### `balanced`

- neutral operating mode

### `sla_first`

- higher pressure sensitivity
- larger buffers
- lower `call_now` threshold
- higher urgency bias

## Safety Buffer

The action engine converts forecast pressure into a truck buffer:

`buffer_applied = base + volatility_add + friday_add + near_term_add + mode_adjustments`

`safety_multiplier = 1 + buffer_applied`

Where:

- `volatility_add` reacts to `route_cv`
- `friday_add` covers Friday regime
- `near_term_add` reacts to immediate peak risk
- mode adjustments come from the selected service mode

## Action Rules

The engine is explicitly three-level, not binary.

### `call_now`

Triggered when:

- pressure is high or critical
- peak slot is close
- forecasted load is already actionable

### `monitor`

Triggered when:

- demand is building
- risk exists but is not yet urgent
- service should keep the route under observation rather than dispatch immediately

### `hold`

Triggered when:

- pressure stays low
- near-term load remains below the action threshold

## Decision Package

`/plan` returns, for every route:

- warehouse / office id
- route id
- horizon summary
- slot pressure score and level
- recommended action
- recommended trucks
- urgency
- selected service mode
- explanation
- risk fields

`TruckRequest` objects are emitted only for `call_now` decisions.
