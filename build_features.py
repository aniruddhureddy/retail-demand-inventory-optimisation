"""Join source tables and validate past-only rolling features, optionally using PySpark."""
from pathlib import Path
import json,sqlite3,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)

def build(use_spark=False):
    sales=pd.read_csv(DATA/'sales_subset.csv');cal=pd.read_csv(DATA/'calendar.csv');prices=pd.read_csv(DATA/'prices_subset.csv')
    meta=['item_id','dept_id','cat_id','store_id','state_id']
    panel=sales.melt(id_vars=meta,var_name='d',value_name='units');panel['day']=panel.d.str[2:].astype(int)
    assert not panel.duplicated(['item_id','day']).any()
    db=OUT/'retail.sqlite'
    if db.exists():db.unlink()
    with sqlite3.connect(db) as con:
        panel[['item_id','dept_id','store_id','day','units']].to_sql('sales',con,index=False)
        cal[['day','date','wm_yr_wk','wday','month','year','snap_CA']].to_sql('calendar',con,index=False)
        prices.to_sql('prices',con,index=False)
        sql=(ROOT/'analysis.sql').read_text(encoding='utf-8');con.executescript(sql)
        joined=pd.read_sql_query('SELECT * FROM daily_panel ORDER BY item_id, day',con)
        sql_lags=pd.read_sql_query('SELECT * FROM rolling_features ORDER BY item_id, day',con)
    assert len(joined)==60*1913
    assert int(joined.units.sum())==int(panel.units.sum())
    joined['price_asof']=joined.groupby('item_id').sell_price.ffill()
    g=joined.groupby('item_id').units
    for n in [7,28,56]:joined[f'mean_{n}']=g.transform(lambda x:x.rolling(n,min_periods=n).mean())
    joined['std_28']=g.transform(lambda x:x.rolling(28,min_periods=28).std(ddof=1))
    joined['zero_28']=g.transform(lambda x:x.eq(0).rolling(28,min_periods=28).mean())
    for n in [7,28,56]:
        mask=joined.day>=n
        np.testing.assert_allclose(joined.loc[mask,f'mean_{n}'],sql_lags.loc[mask,f'mean_{n}'])
    engine='pandas and SQLite';spark_version=None
    if use_spark:
        from pyspark.sql import SparkSession,Window,functions as F
        spark=SparkSession.builder.master('local[2]').appName('RetailForecastFeatures').config('spark.ui.enabled','false').config('spark.sql.shuffle.partitions','4').config('spark.driver.memory','1g').getOrCreate()
        spark.sparkContext.setLogLevel('ERROR');spark_version=spark.version
        a=spark.createDataFrame(panel[['item_id','dept_id','store_id','day','units']])
        c=spark.createDataFrame(cal[['day','date','wm_yr_wk','wday','month','year','snap_CA']])
        p=spark.createDataFrame(prices)
        s=a.join(F.broadcast(c),'day').join(p,['store_id','item_id','wm_yr_wk'],'left')
        w=Window.partitionBy('item_id').orderBy('day')
        s=s.withColumn('price_asof',F.last('sell_price',ignorenulls=True).over(w.rowsBetween(Window.unboundedPreceding,0)))
        for n in [7,28,56]:s=s.withColumn(f'mean_{n}',F.avg('units').over(w.rowsBetween(1-n,0)))
        s=s.withColumn('std_28',F.stddev_samp('units').over(w.rowsBetween(-27,0))).withColumn('zero_28',F.avg((F.col('units')==0).cast('double')).over(w.rowsBetween(-27,0)))
        # Compare every usable model row with independent pandas calculations.
        cols=['item_id','day','price_asof','mean_7','mean_28','mean_56','std_28','zero_28']
        check=s.select(cols).where('day >= 56').orderBy('item_id','day').toPandas()
        ref=joined[joined.day>=56].sort_values(['item_id','day'])
        for col in cols[2:]:np.testing.assert_allclose(check[col],ref[col],equal_nan=True,rtol=1e-8,atol=1e-8)
        # Use the independently verified Spark values in the modelling export.
        joined.loc[joined.day>=56,cols[2:]]=check[cols[2:]].to_numpy()
        s.where('day >= 56').write.mode('overwrite').partitionBy('dept_id').parquet(str(OUT/'spark_features.parquet'))
        spark.stop();engine='PySpark; verified against pandas and SQLite'
    joined.to_csv(OUT/'feature_panel.csv.gz',index=False,compression='gzip')
    audit={'rows':len(joined),'units_reconciled':int(joined.units.sum()),'engine':engine,'spark_version':spark_version,'rolling_features_include_origin_day':'Allowed: origin-day sales are observed when the forecast is made at close of day','price_rule':'Current observed weekly price, forward-filled only; future prices excluded from model inputs','missing_price_after_day_365':int(joined.loc[joined.day>=365,'price_asof'].isna().sum()),'sql_rolling_parity':'passed','spark_rolling_parity':'passed' if use_spark else 'not run'}
    (OUT/'feature_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2));return joined
if __name__=='__main__':build('--spark' in sys.argv)
