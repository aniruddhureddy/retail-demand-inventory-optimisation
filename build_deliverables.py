"""Build an executed explanatory notebook and self-contained report from the actual run."""
from pathlib import Path
import os,json,io,contextlib,base64,html
import pandas as pd
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT);R=ROOT/'results'

def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text):return {'cell_type':'code','metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]}
cells=[
md('# Retail Demand Forecasting and Inventory Optimisation\n\n**Decision:** improve replenishment under limited purchasing funds. This notebook reads actual executed analysis outputs. Set `REBUILD=True` to regenerate forecasting and simulations. The separate Spark feature pipeline is documented in the README.'),
code("from pathlib import Path\nimport json, runpy\nimport pandas as pd\nR = Path('results')\nREBUILD = False\nif REBUILD or not (R / 'diagnostics.json').exists():\n    _ = runpy.run_path('run_analysis.py', run_name='__main__')\nprint((R / 'diagnostics.json').read_text())"),
md('## 1 Data engineering and scope\n\nThe 60-product cohort is selected with information ending at day 1605, before all validation and test origins. PySpark joins and rolling windows are cross-checked against pandas and SQLite. The modelling export uses the verified Spark feature values.'),
code("print((R / 'feature_audit.json').read_text())\nprint(pd.read_csv('data/selection_audit.csv').groupby('dept_id').agg(products=('item_id','size'), selection_period_units=('training_units','sum')).to_string())"),
md('## 2 SQL aggregation\n\nThe queryable database includes raw sales, prices and calendar tables, a joined daily view, trailing-window features and department totals.'),
code("import sqlite3\nwith sqlite3.connect(R / 'retail.sqlite') as con:\n    overview = pd.read_sql_query('SELECT * FROM department_monthly_sales WHERE year = 2016 ORDER BY month, dept_id', con)\nprint(overview.to_string(index=False))"),
md('## 3 Validation and the untouched test sequence\n\nModels are selected on three earlier 28-day windows, then evaluated on three later windows. Refitting at later origins may use sales observed in completed earlier windows. Every training label ends no later than its forecast origin.'),
code("print((R / 'locked_selection.json').read_text())\ncutoffs = pd.read_csv(R / 'training_cutoff_audit.csv')\nassert (cutoffs.latest_training_label <= cutoffs.origin).all()\nprint(cutoffs.to_string(index=False))"),
md('## 4 Forecast accuracy\n\nThe selected model is not changed after seeing test results. All baselines remain visible. The reported simple mean RMSSE is not the official M5 WRMSSE.\n\n![Forecast comparison](results/figures/forecast_comparison.png)'),
code("print(pd.read_csv(R / 'forecast_comparison.csv').to_string(index=False))\nprint('\\nPer-origin results:')\nprint(pd.read_csv(R / 'forecast_metrics.csv').to_string(index=False))"),
md('## 5 Prediction intervals\n\nItem-level validation residual quantiles create nominal 80% intervals. Held-out empirical coverage is reported without test-based recalibration.\n\n![Forecast timeline](results/figures/forecast_timeline.png)'),
code("print(pd.read_csv(R / 'interval_coverage.csv').to_string(index=False))"),
md('## 6 Inventory decisions\n\nAll policies share the same budget rule and starting conditions within each scenario. Operating cost is holding cost plus assumed shortage penalty. Procurement spending and inventory value are separate.\n\n![Inventory comparison](results/figures/inventory_comparison.png)'),
code("inventory = pd.read_csv(R / 'inventory_base_case.csv')\nprint(inventory.to_string(index=False))\norders = pd.read_csv(R / 'inventory_orders.csv')\nassert (orders.purchase_spend <= orders.budget + 1e-5).all()\nprint('\\nEvery review satisfies the purchasing budget.')\nprint('Optimisation solver status counts:')\nprint(orders.solver_status.dropna().value_counts().to_string())"),
md('## 7 Sensitivity and trade-offs\n\nNine combinations vary purchasing limits and replenishment lead time. Initial stock also changes with lead time; compare policies within each scenario, rather than interpreting cross-scenario differences as isolated causal effects.\n\n![Inventory cost timeline](results/figures/inventory_timeline.png)'),
code("sensitivity = pd.read_csv(R / 'inventory_sensitivity.csv')\nprint(sensitivity.pivot(index=['lead_time','budget_factor'],columns='policy',values='inventory_operating_cost').to_string())"),
md('## 8 Verification\n\nFocused tests check future-sales isolation, forecast baselines, metric arithmetic, purchasing limits and a known small optimisation problem.'),
code("import subprocess, sys\nresult = subprocess.run([sys.executable, 'test_analysis.py'], capture_output=True, text=True, check=True)\nprint(result.stdout + result.stderr)"),
md('## Recommendation\n\nPilot the scenario-based ordering policy with actual retailer costs and inventory data. Report the additional working capital alongside simulated service gains. The model is a reproducible offline prototype; no live replenishment or realised company savings are claimed.')]
ns={};count=0
for c in cells:
    if c['cell_type']!='code':continue
    count+=1;buf=io.StringIO()
    with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):exec(compile(''.join(c['source']),f'<notebook cell {count}>','exec'),ns)
    c['execution_count']=count;c['outputs']=[{'output_type':'stream','name':'stdout','text':buf.getvalue().splitlines(True)}]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':4}
(ROOT/'Forecast_Analysis.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')

d=json.loads((R/'diagnostics.json').read_text());comp=pd.read_csv(R/'forecast_comparison.csv');base=pd.read_csv(R/'inventory_base_case.csv').set_index('policy')
old=base.loc['Trailing mean cover'];new=base.loc['Scenario optimisation'];cover=base.loc['Forecast cover']
cost_reduction=100*(1-new.inventory_operating_cost/old.inventory_operating_cost)
capital_rise=100*(new.average_inventory_value/old.average_inventory_value-1)
test=comp[comp.split=='test'].set_index('model');selected=d['selected_model']
wape=test.loc[selected,'wape'];naive=test.loc['Weekly seasonal naive','wape'];weekday=test.loc['Weekday mean 56d','wape']
improvement=100*(1-wape/naive)
coverage=pd.read_csv(R/'interval_coverage.csv').empirical_coverage.mean()

def fig(name,alt):
    b=base64.b64encode((R/'figures'/name).read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b}">'
forecastrows=''
for _,x in comp.iterrows():forecastrows+=f'<tr><td>{x["split"].title()}</td><td>{x.model}</td><td>{x.wape:.2%}</td><td>{x.bias:+.2%}</td><td>{x.mean_rmsse:.3f}</td></tr>'
inventoryrows=''
for name,x in base.iterrows():inventoryrows+=f'<tr><td>{name}</td><td>{x.fill_rate:.2%}</td><td>${x.inventory_operating_cost:,.0f}</td><td>${x.average_inventory_value:,.0f}</td><td>${x.purchase_spend:,.0f}</td><td>{int(x.stockout_item_days)}</td></tr>'
report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Retail Demand Forecasting and Inventory Optimisation</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f8;color:#172b4d;font:16px/1.65 system-ui,-apple-system,Segoe UI,Arial,sans-serif}}main{{max-width:1100px;margin:28px auto;background:#fff;padding:48px 58px}}.eyebrow{{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:#0f766e;font-weight:700}}h1{{font-size:42px;line-height:1.13;letter-spacing:-.03em;margin:14px 0 20px}}h2{{font-size:25px;line-height:1.3;margin:36px 0 12px}}h3{{font-size:18px;margin-top:24px}}p{{margin:12px 0}}.lead{{font-size:20px;color:#40546b}}.note{{font-size:13px;color:#5a6b7d}}nav{{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}}a{{color:#0f766e}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;border-top:1px solid #dbe3e9;border-bottom:1px solid #dbe3e9;padding:24px 0;margin:25px 0}}.metrics b{{display:block;font-size:30px;line-height:1.2}}.metrics span{{font-size:13px;color:#526275}}img{{width:100%;height:auto;margin:16px 0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}}th,td{{padding:10px 9px;border-bottom:1px solid #dbe3e9;text-align:left}}th{{background:#edf3f6;font-size:12px;text-transform:uppercase}}td{{font-variant-numeric:tabular-nums}}li{{margin:8px 0}}footer{{border-top:1px solid #dbe3e9;margin-top:30px;padding-top:20px}}code{{background:#edf3f6;padding:2px 5px}}@media(max-width:700px){{main{{margin:0;padding:25px 20px}}h1{{font-size:31px}}.metrics b{{font-size:24px}}table{{font-size:12px}}}}@media print{{body{{background:#fff}}main{{margin:0;padding:0}}nav{{display:none}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}</style></head><body><main>
<div class="eyebrow">Analytics portfolio · Retail operations</div><h1>Retail Demand Forecasting and Inventory Optimisation</h1>
<p class="lead">Better stock availability under a fixed budget—at the cost of more inventory.</p><p class="note">Prepared for Aniruddh Reddy · M5 historical data · Python, SQL, local PySpark, gradient boosting and integer optimisation</p>
<nav><a href="#decision">Decision</a><a href="#forecast">Forecasting</a><a href="#inventory">Inventory</a><a href="#methods">Method</a><a href="#limits">Limitations</a></nav>
<div class="metrics"><div><b>114,780</b><span>item-day observations</span></div><div><b>6 × 28 days</b><span>validation and test windows</span></div><div><b>{cost_reduction:.1f}%</b><span>lower simulated operating cost*</span></div></div>
<p class="note">*Versus trailing-mean stock cover in the base case. Operating cost means holding cost plus assumed stockout penalties. These are simulation results.</p>
<section id="decision"><h2>Recommended decision</h2><p>Pilot the scenario-based replenishment policy using actual retailer costs and inventory. In the base case, it reduced simulated operating cost from <strong>${old.inventory_operating_cost:,.0f} to ${new.inventory_operating_cost:,.0f}</strong> and improved unit fill rate from <strong>{old.fill_rate:.1%} to {new.fill_rate:.1%}</strong>.</p>
<p>The trade-off is material: average inventory value increased from <strong>${old.average_inventory_value:,.0f} to ${new.average_inventory_value:,.0f}</strong> ({capital_rise:.1f}%). The purchasing-budget rule is identical across policies, but actual procurement spend differs. The recommendation depends on whether improved availability justifies the additional working capital.</p></section>
<section id="forecast"><h2>Forecast performance</h2><p>The project forecasts 60 established food products at store CA_1. Products were selected before evaluation. Three earlier validation windows selected <strong>{selected.lower()}</strong>; three later windows provide the walk-forward test.</p>
{fig('forecast_comparison.png','Validation and test WAPE for seasonal naive, weekday-average and gradient-boosting forecasts')}
<div class="scroll"><table><thead><tr><th>Period</th><th>Method</th><th>WAPE</th><th>Bias</th><th>Mean RMSSE</th></tr></thead><tbody>{forecastrows}</tbody></table></div>
<p>The selected model’s mean test WAPE was <strong>{wape:.2%}</strong>, versus {naive:.2%} for weekly seasonal naive—a {improvement:.1f}% relative error reduction. The weekday-average method recorded <strong>{weekday:.2%}</strong> test WAPE, slightly better than the selected model. The validation-selected method was retained; test results were not used to switch models.</p>
{fig('forecast_timeline.png','Observed and predicted aggregate units over the final 84-day test period')}
<p class="note">WAPE is averaged across three origins. RMSSE is an unweighted mean across the selected products, not official M5 WRMSSE. Nominal 80% validation-residual intervals covered {coverage:.1%} of test observations: undercoverage remains a limitation.</p></section>
<section id="inventory"><h2>Inventory decisions and economics</h2><p>The base case reviews orders weekly, assumes a two-day replenishment lead time, and offers each policy a purchasing limit based on the same trailing demand. The optimiser chooses integer order quantities across products using 48 validation-error demand scenarios.</p>
{fig('inventory_comparison.png','Simulated inventory operating cost and fill rate for the three replenishment policies')}
<div class="scroll"><table><thead><tr><th>Policy</th><th>Fill rate</th><th>Operating cost</th><th>Average stock value</th><th>Purchasing spend</th><th>Stockout item-days</th></tr></thead><tbody>{inventoryrows}</tbody></table></div>
<p class="note">All dollar values use assumed procurement cost equal to 60% of price, holding cost of 0.1% of unit cost per day, and a lost-sales penalty equal to 50% of price per unit. Purchase outlay is shown separately from operating cost. Stockout item-days count each product-day with unmet demand.</p>
{fig('inventory_timeline.png','Cumulative simulated holding and shortage cost across the test period')}
<p>Across nine combinations of budget and lead time, the scenario policy had lower simulated operating cost than both cover rules. All <strong>{d['optimisation_runs']}</strong> optimisation runs reached the solver’s optimal status within its specified tolerance, and no purchasing limit was breached. This sensitivity grid uses the same historical period; it is not nine independent business trials.</p></section>
<section id="methods"><h2>How the project works</h2><ol><li><strong>Prepare and verify:</strong> join daily sales, calendar and weekly prices; reconcile 1,337,156 unit sales. Execute local PySpark windows and Parquet output, with Python and SQL parity checks.</li><li><strong>Prevent future-data leakage:</strong> training targets end by each origin. Future sales and future prices do not enter predictors; known calendar and event features do.</li><li><strong>Compare forecasts:</strong> predict all 28 horizons directly. Refit only as time advances, and lock the method using validation.</li><li><strong>Connect forecasts to decisions:</strong> refresh predictions at weekly reviews, account for stock on hand and in transit, and enforce a shared cash limit.</li><li><strong>Evaluate the trade-off:</strong> track fulfilled demand, stockouts, inventory value, procurement spending, holding cost and shortage penalties.</li></ol></section>
<section id="limits"><h2>Limits before a real deployment</h2><ul><li>The cohort favours established, relatively high-volume products in one store. It excludes cold-start and broader store-network behaviour.</li><li>Historical sales are used as demand. Unobserved historical stockouts may mean true demand was higher.</li><li>Inventory, costs, lead times and penalties are assumed. Higher holding-cost or lower stockout-penalty assumptions can change the decision.</li><li>The optimisation chooses from a finite set of order quantities under a simplified protection-period objective. It is not a globally optimal dynamic inventory policy.</li><li>Initial stock changes with lead time, so cross-lead-time results do not isolate delivery-speed effects. Compare policies within scenarios.</li><li>Prediction intervals undercovered on test data, and only three test windows are available. Further calibration and prospective testing are needed.</li></ul></section>
<footer><h3>Reproduce the work</h3><p>The repository includes scripts, SQL, an executed notebook, reference metrics, solver logs and tests. Source data, feature panels, Parquet output and the SQL database are retrieved or generated locally. Run <code>python run_analysis.py</code> to regenerate the core analysis; the README explains how to rerun Spark.</p><p><a href="https://github.com/Mcompetitions/M5-methods">M5 organisers</a> · <a href="https://github.com/Nixtla/m5-forecasts">Source archive mirror</a> · <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html">Integer optimisation documentation</a></p><p class="note">Offline analytical prototype. No live replenishment, realised cost savings or production deployment is claimed.</p></footer></main></body></html>'''
from report_theme import render_report
report = render_report(report)
(ROOT/'Forecast_Results.html').write_text(report,encoding='utf-8')
bullets=f'''# Project resume wording

Retail Demand Forecasting and Inventory Optimisation | Python, SQL, PySpark, scikit-learn, SciPy

- Engineered and validated 114,780 daily sales observations across 60 products using SQL and local PySpark, with rolling demand features and partitioned Parquet outputs.
- Benchmarked direct 28-day gradient-boosting forecasts across six rolling validation/test windows; achieved {wape:.1%} mean test WAPE versus {naive:.1%} for a weekly seasonal-naive baseline.
- Built a budget-constrained integer replenishment optimiser; reduced simulated holding-plus-stockout cost by {cost_reduction:.1f}% and raised fill rate from {old.fill_rate:.1%} to {new.fill_rate:.1%} in the base case, with {capital_rise:.1f}% higher average inventory value.

For a compact resume use two bullets, keeping the word simulated with inventory results. The simpler weekday-average forecast achieved {weekday:.2%} test WAPE, slightly better than boosting; do not describe boosting as the best test model. The pipeline ran locally, not on a distributed production cluster. Detailed assumptions and tests are in the README.
'''
(ROOT/'RESUME_BULLETS.md').write_text(bullets,encoding='utf-8')
print('Created executed notebook, self-contained results report and project bullets.')
