# Assumptions

- The main product story is warehouse slot orchestration, not raw transport forecasting.
- Runtime never trains models; it only loads artifacts from `artifacts/`.
- `route_id -> office_from_id` is treated as deterministic because it is deterministic in the provided data.
- `target_2h` is interpreted as rolling two-hour load and is therefore used as the core slot pressure signal.
- `blend_best_timexer_main.csv` is the top frozen leaderboard reference.
- Exact source code for `blend_best_chronos_groupwise_main.csv` was not recoverable from `coding.ipynb`, so the repo documents it as a frozen reference and serves a reproducible anchor proxy instead.
- `coding.ipynb` explicitly references `Chronos-2`, so the repo exports that exact pretrained family for runtime use.
- `coding.ipynb` does not expose an explicit `TSMixerx` training block. Because TimeXer does not fit the available hardware, the repo trains `TSMixerx` with compatible parameters mapped from the real saved lightweight-branch config in `info_for_codex/artifacts_timexer/meta/timexer_config.json`.
- Proxy Chronos / TimeXer artifacts remain in the repo as fallback runtime branches for environments where real `Chronos2` export or real `TSMixerx` training has not been completed yet.
- KPI values are forecast-driven proxies because production replay or dispatch outcome logs are not part of the provided materials.
