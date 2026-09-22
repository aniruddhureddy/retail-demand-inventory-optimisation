-- Each day is observed before forecasts are issued at its close.
CREATE VIEW daily_panel AS
SELECT s.item_id,s.dept_id,s.store_id,s.day,s.units,
       c.date,c.wm_yr_wk,c.wday,c.month,c.year,c.snap_CA,p.sell_price
FROM sales s
JOIN calendar c ON s.day=c.day
LEFT JOIN prices p ON s.item_id=p.item_id AND s.store_id=p.store_id AND c.wm_yr_wk=p.wm_yr_wk;

CREATE VIEW rolling_features AS
SELECT item_id,day,
 AVG(units) OVER(PARTITION BY item_id ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS mean_7,
 AVG(units) OVER(PARTITION BY item_id ORDER BY day ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS mean_28,
 AVG(units) OVER(PARTITION BY item_id ORDER BY day ROWS BETWEEN 55 PRECEDING AND CURRENT ROW) AS mean_56
FROM sales;

CREATE VIEW department_monthly_sales AS
SELECT dept_id,year,month,SUM(units) AS units,COUNT(*) AS item_days
FROM daily_panel GROUP BY dept_id,year,month;
