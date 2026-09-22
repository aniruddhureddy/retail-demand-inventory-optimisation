# Detailed methodology

## Dataset and subset

Original competition: https://github.com/Mcompetitions/M5-methods

Download mirror: https://github.com/Nixtla/m5-forecasts

The mirror identifies itself as an archive of the organisers' dataset. Only source data was used; no competition solution or pretrained forecast was copied. The raw input's provenance, checksums and subset-selection rules are in `data/provenance.json`. Data rights remain with the original providers; review their terms before publicly redistributing raw files.

Selection is fixed before evaluation: store CA_1, food departments, top 20 products per department by units sold over days 1241–1605, requiring at least one sale by day 365. Ties use item ID. All model evaluation origins occur after the selection cutoff. This favours established higher-volume products and is not representative of cold starts or the full M5 catalogue.

The Nixtla calendar file omits the original `d` column. `prepare_data.py` verifies daily continuity from 2011-01-29 and reconstructs sequential day IDs. There are 1,913 historical sales days and 1,969 calendar days. Each item-store-day is unique. Weekly prices join by item, store and week, and are forward-filled only. Sales are nonnegative counts. Future prices are not used in prediction.

## Forecast evaluation

- Forecast origin means the close of an observed day. All features are available by that close.
- Validation origins: 1745, 1773, 1801. Final test origins: 1829, 1857, 1885. Each predicts the next 28 days.
- The gradient-boosting training set samples weekly historical origins over roughly two years; each training label must end no later than the current forecast origin.
- A direct model includes horizon, item and department, target calendar attributes, trailing demand statistics, an origin-known price, and historic weekday averages. Target calendar/SNAP/event features are treated as published in advance.
- Validation mean WAPE selects the forecasting method. This choice is recorded before test evaluation. The method is refitted at each test origin using only data observed by then; earlier test windows can therefore become later training history in a walk-forward evaluation.
- Test comparisons with all three methods are reported without changing the selected method. The daily inventory forecasts refresh weekly using newly observed history, but reuse the model fitted at the beginning of that 28-day block.
- WAPE aggregates absolute errors divided by actual units. The report averages WAPE across the three equal-length origins. Bias is signed aggregate error divided by units. Per-series RMSSE uses first-difference scale from the available training history after the first positive sale; the reported mean is **not** the official hierarchy-weighted M5 WRMSSE.
- Nominal 80% intervals add item-level 10th/90th percentile validation residuals to forecasts, truncate at zero and report empirical test coverage. They are empirical residual intervals, not guaranteed calibrated prediction intervals. The small number of test blocks does not support strong claims of statistical superiority.

## Inventory design and assumptions

The test period covers 84 days. Reviews occur every seven days. Base-case lead time is two days, with sensitivity at zero and four days. Each comparison starts with the same stock: ceiling of the prior 28-day daily mean times review interval plus lead time. This initial stock is an explicit assumption, not observed inventory.

Price and unit cost are fixed at the first test origin for simulation comparability. Procurement cost is assumed to be 60% of selling price. Daily holding cost is 0.1% of procurement cost per ending unit. Lost-sales penalty is 50% of selling price per unfulfilled unit. Demand is proxied by historical sales, so unobserved historical stockouts may censor true demand. Orders arrive before demand on their scheduled arrival day; unfulfilled demand is lost rather than backordered.

At each review, purchasing budget equals seven days of trailing-mean demand at assumed procurement cost, times a factor of 0.8, 1.0 or 1.2. The same budget is offered to each policy. Actual spending can differ because these are limits, not spending quotas.

1. **Trailing mean cover:** target protection-period mean demand plus one standard-deviation safety buffer, adjusted for on-hand and pipeline inventory. Scale orders proportionally if the budget binds.
2. **Forecast cover:** substitute the selected forecast for the mean-demand component; use the same safety buffer and budget allocation rule.
3. **Scenario optimisation:** add jointly sampled validation-residual blocks to the point forecast to form 48 demand scenarios. For each product, create integer order choices from scenario quantiles. Solve a multiple-choice integer programme to minimise expected excess-stock and shortage penalties subject to the budget. This is optimisation over a finite candidate set and a simplified protection-period objective, not an exact optimal solution to a full dynamic inventory-control problem.

All policies are assessed on the same daily sales. `inventory_operating_cost` means holding cost plus shortage penalty. Procurement spend is reported separately: it buys inventory and is not included in this operating-cost total. Increased stock can reduce shortages while raising working-capital needs. No live savings or business deployment is claimed.

Initial stock scales with lead time, so cross-lead-time results do not isolate the causal effect of slower delivery. Compare policies within each scenario. Empirical forecast-interval coverage is also reported even when it misses the nominal target; intervals are not recalibrated on the test set.

