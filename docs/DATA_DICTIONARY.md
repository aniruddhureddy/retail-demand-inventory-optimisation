# Data dictionary

| Field | Grain | Meaning |
|---|---|---|
| `item_id` | Product | M5 product identifier, retained as a categorical predictor |
| `dept_id`, `cat_id` | Product | Department and category |
| `store_id`, `state_id` | Store | This subset uses store CA_1 in California |
| `d_1` … `d_1913` | Product-store-day | Historical unit sales, including zero sales |
| `day` | Day | Sequential index derived from consecutive calendar dates |
| `date` | Day | Calendar date |
| `wm_yr_wk` | Week | Calendar key for joining weekly prices |
| `wday`, `month`, `year` | Day | Calendar attributes |
| `event_name_1`, `event_name_2` | Day | Scheduled events; converted to an event indicator |
| `snap_CA` | Day | California SNAP calendar indicator, assumed known ahead |
| `sell_price` | Product-store-week | Observed weekly selling price |
| `price_asof` | Product-store-day | Last observed weekly price, forward-filled without looking ahead |
| `mean_7`, `mean_28`, `mean_56` | Product-origin | Sales averages through the observed forecast origin |
| `std_28`, `zero_28` | Product-origin | Recent demand variation and share of zero-sale days |

Inventory, procurement cost, lead time, budget, holding cost and lost-sales penalty are not observed M5 fields. They are explicitly assumed simulation inputs. Original M5 source: https://github.com/Mcompetitions/M5-methods .
