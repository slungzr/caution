# Codex Handoff

## Model
- Current model in this session: `gpt-5.4`

## Final Strategy
- Base signal: original `竞价爬升`
- Filters:
- `竞价匹配金额_openapi >= 5000万`
- `实体涨跌幅昨日 < 实体涨跌幅前日`
- `市场20日高低差 >= 0`
- Ranking:
- sort by `个股热度排名昨日` ascending
- Execution:
- buy up to top `3` names at `T` open, equal weight
- from `T+1`, exit on the first day whose close is not limit-up

## Final Result
- File: `竞价爬升-20240504-新因子策略版摘要.csv`
- Final NAV: `4.29084501`
- Total return: `329.08%`
- Max drawdown: `-20.75%`
- Trades: `99`
- Win rate: `56.57%`
- Return/drawdown ratio: `15.8562`

## Key Files
- Main backtest: `竞价爬升策略回测.py`
- Final factor dataset: `竞价爬升-20240504-扩展因子-验证期.csv`
- Final strategy outputs:
- `竞价爬升-20240504-新因子策略版摘要.csv`
- `竞价爬升-20240504-新因子策略版净值.csv`
- `竞价爬升-20240504-新因子策略版交易明细.csv`
- `竞价爬升-20240504-新因子策略版候选池.csv`
- Direction exploration: `竞价爬升-20240504-局部验证-结果.csv`
- Risk control exploration: `竞价爬升-20240504-风控验证-结果.csv`

## Important Bug Fix
- There was a bug in `竞价爬升策略回测.py` where the final `新因子策略版` run passed an empty `breadth_df` into `run_config_backtest`.
- That meant `市场20日高低差 >= 0` was shown in the summary but was not actually applied.
- This has been fixed. The final result `4.29084501` is the post-fix result.

## Exploration Conclusions

### Signal / Filter Exploration
- Strongest complete-validation filter stack:
- `竞价匹配金额_openapi >= 5000万`
- `实体涨跌幅昨日 < 实体涨跌幅前日`
- `市场20日高低差 >= 0`

### Risk Control Exploration
- Best risk control was not "stop after 2 losing days".
- Best overlay was a market-state gate:
- `市场20日高低差 >= 0 才开仓`
- Files: `竞价爬升-20240504-风控验证-结果.csv`

## Data Enrichment Pipeline

### Main enrichment result
- Completed factor coverage in `竞价爬升-20240504-扩展因子-验证期.csv`
- Covered trading dates: `127`
- Rows with factor values: `1239`

### Enrichment scripts
- `wencai_openapi.py`
- Early-stage OpenAPI client; later often hit `403`
- `wencai_unifiedwap.py`
- Browser-session `unified-wap` client using `cookie + hexin-v`
- `竞价因子补充.py`
- Early OpenAPI enrichment script
- `竞价因子补充_unifiedwap.py`
- Main low-frequency supplementation script for later dates
- `竞价因子补充_pywencai.py`
- Fallback supplementation script; used to finish the last dates
- `重建失败日期清单.py`
- Rebuild remaining dates from factor coverage and trading-calendar intersection
- `重建扩展因子文件.py`
- Recover the factor file from caches if a write step corrupts or drops prior factor columns

## Stateful / Time-Sensitive Notes
- `wencai_unifiedwap.py` contains time-sensitive `hexin-v` and `Cookie`
- `竞价因子补充_pywencai.py` also depends on time-sensitive cookie state
- If Codex needs to enrich new dates later, these tokens may need refreshing from browser session again

## Legacy / Non-Primary Scripts
- `竞价爬升区间.py`
- Original scraper; not the preferred enrichment path now
- `竞价爬升常规统计.py`
- Old average-return curve logic; not the real-position backtest source of truth
- `wencai_direct.py`
- Experimental direct web client used during debugging
- `问财因子探测.py`
- Field probing / debugging only

## Suggested Next Steps For Codex
1. Extract the final strategy into a cleaner standalone production script
2. Build a daily execution report for manual use
3. If future enrichment is needed, refresh browser session tokens first, then prefer:
- `竞价因子补充_unifiedwap.py`
- fallback to `竞价因子补充_pywencai.py`

## Minimal Handoff Prompt
```text
Please continue from C:\Users\zr\Documents\caution.

Final strategy is:
1. Original 竞价爬升 base signal
2. Filter: 竞价匹配金额_openapi >= 5000万
3. Filter: 实体涨跌幅昨日 < 实体涨跌幅前日
4. Filter: 市场20日高低差 >= 0
5. Rank by 个股热度排名昨日 ascending
6. Buy top 3 at T open, equal weight
7. Exit on the first day after entry whose close is not limit-up

Final official result is in:
- 竞价爬升-20240504-新因子策略版摘要.csv

Important:
- The final strategy market-breadth filter bug in 竞价爬升策略回测.py has already been fixed.
- Enrichment helper scripts using unified-wap / pywencai have time-sensitive cookie and hexin-v state.
```
