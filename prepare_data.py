"""Recreate a training-selected M5 subset from the documented public archive."""
from pathlib import Path
import zipfile,json,hashlib,urllib.request,shutil
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
URL='https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip'
def prepare():
    archive=DATA/'m5.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=90) as r,open(archive,'wb') as f:shutil.copyfileobj(r,f)
    with zipfile.ZipFile(archive) as z:
        cal=pd.read_csv(z.open('calendar.csv'))
        cal['day']=np.arange(1,len(cal)+1)
        dates=pd.to_datetime(cal.date)
        assert (dates.diff().dropna().dt.days==1).all()
        assert str(dates.iloc[0].date())=='2011-01-29'
        sales=pd.read_csv(z.open('sales_train_validation.csv'))
        assert sales.shape[0]==30490
        s=sales[(sales.store_id=='CA_1')&(sales.cat_id=='FOODS')].copy()
        s['training_units']=s[[f'd_{i}' for i in range(1241,1606)]].sum(axis=1)
        s=s[s[[f'd_{i}' for i in range(1,366)]].sum(axis=1)>0]
        selected=s.sort_values(['training_units','item_id'],ascending=[False,True]).groupby('dept_id').head(20).sort_values('item_id')
        selection=selected[['item_id','dept_id','training_units']].copy()
        selected=selected.drop(columns='training_units')
        assert len(selected)==60
        parts=[]
        for chunk in pd.read_csv(z.open('sell_prices.csv'),chunksize=250000):
            parts.append(chunk[(chunk.store_id=='CA_1')&chunk.item_id.isin(selected.item_id)])
        prices=pd.concat(parts)
    assert not prices.duplicated(['store_id','item_id','wm_yr_wk']).any()
    assert (prices.sell_price>0).all()
    selected.to_csv(DATA/'sales_subset.csv',index=False)
    selection.to_csv(DATA/'selection_audit.csv',index=False)
    cal.to_csv(DATA/'calendar.csv',index=False)
    prices.to_csv(DATA/'prices_subset.csv',index=False)
    provenance={'archive_url':URL,'original_source':'https://github.com/Mcompetitions/M5-methods','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'scope':'60 CA_1 foods SKUs; 20 per department','selection_information_cutoff':1605,'selection_sales_window':[1241,1605],'early_availability_rule':'At least one sale by day 365','calendar_repair':'Nixtla calendar omits d index; derive sequential day from verified consecutive dates beginning 2011-01-29','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'sales_subset.csv',DATA/'calendar.csv',DATA/'prices_subset.csv']}}
    (DATA/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Prepared',len(selected),'SKUs,',len(cal),'calendar days and',len(prices),'weekly prices.')
if __name__=='__main__':prepare()
