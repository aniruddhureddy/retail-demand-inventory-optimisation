"""Direct multi-horizon forecasts and constrained replenishment, without future sales leakage."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,time,platform
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix,csc_matrix
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';OUT=ROOT/'results';FIG=OUT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
SEED=20260923;VALID_ORIGINS=[1745,1773,1801];TEST_ORIGINS=[1829,1857,1885];H=28
MODEL_NAMES=['Weekly seasonal naive','Weekday mean 56d','Gradient boosting']
CAT=['item_code','dept_code','target_weekday']
def js(path,obj):path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

class RetailData:
    def __init__(self):
        self.sales=pd.read_csv(DATA/'sales_subset.csv').sort_values('item_id').reset_index(drop=True)
        self.cal=pd.read_csv(DATA/'calendar.csv').set_index('day');self.n=len(self.sales)
        self.y=self.sales[[f'd_{i}' for i in range(1,1914)]].to_numpy(float)
        self.items=self.sales.item_id.to_numpy();self.dept=pd.Categorical(self.sales.dept_id).codes
        f=pd.read_csv(OUT/'feature_panel.csv.gz').sort_values(['item_id','day'])
        self.features={c:f[c].to_numpy().reshape(self.n,1913) for c in ['mean_7','mean_28','mean_56','std_28','zero_28','price_asof']}
    def frame(self,origin,horizon=H):
        assert origin>=56
        hs=np.arange(1,horizon+1);days=origin+hs
        cal=self.cal.loc[days];dow=cal.wday.to_numpy()
        olddays=np.arange(origin-55,origin+1)
        olddow=self.cal.loc[olddays,'wday'].to_numpy()
        seasonal=np.column_stack([self.y[:,olddays[olddow==d]-1].mean(axis=1) for d in dow])
        vals={'item_code':np.repeat(np.arange(self.n),horizon),'dept_code':np.repeat(self.dept,horizon),'horizon':np.tile(hs,self.n),'target_weekday':np.tile(dow,self.n),'target_month':np.tile(cal.month,self.n),'snap_CA':np.tile(cal.snap_CA,self.n),'event':np.tile((cal.event_name_1.notna()|cal.event_name_2.notna()).astype(int),self.n),'seasonal_mean':seasonal.ravel()}
        for c,a in self.features.items():vals[c]=np.repeat(a[:,origin-1],horizon)
        vals['trend']=vals['mean_7']/np.maximum(vals['mean_28'],.1)
        result=pd.DataFrame(vals)
        assert not result.isna().any().any();return result
    def fit(self,origin):
        origins=np.arange(max(365,origin-756),origin-H+1,7)
        # Every label in each training example has matured by this forecast origin.
        assert origins.max()+H<=origin
        x=pd.concat([self.frame(o) for o in origins],ignore_index=True)
        target=np.concatenate([self.y[:,o:o+H].ravel() for o in origins])
        model=HistGradientBoostingRegressor(loss='poisson',max_iter=120,max_leaf_nodes=15,min_samples_leaf=60,learning_rate=.07,l2_regularization=10,categorical_features=CAT,early_stopping=False,random_state=SEED)
        model.fit(x,target)
        return model,{'origin':origin,'first_training_origin':int(origins.min()),'last_training_origin':int(origins.max()),'latest_training_label':int(origins.max()+H),'training_examples':len(target)}
    def forecast(self,origin,name,model=None):
        if name=='Weekly seasonal naive':return self.y[:,origin-7:origin][:,np.arange(H)%7]
        x=self.frame(origin)
        if name=='Weekday mean 56d':return x.seasonal_mean.to_numpy().reshape(self.n,H)
        return np.maximum(0,model.predict(x)).reshape(self.n,H)

def forecast_metrics(actual,pred,history):
    scales=[]
    for row in history:
        nz=np.flatnonzero(row>0);segment=row[nz[0]:] if len(nz) else row
        scales.append(np.mean(np.diff(segment)**2))
    scales=np.array(scales);assert (scales>0).all()
    rmsse=np.sqrt(np.mean((pred-actual)**2,axis=1)/scales)
    return {'wape':np.abs(pred-actual).sum()/actual.sum(),'bias':(pred-actual).sum()/actual.sum(),'mean_rmsse':rmsse.mean()},rmsse

def allocate_proportional(request,cost,budget):
    request=np.maximum(0,np.ceil(request)).astype(int)
    factor=min(1,budget/max(float(request@cost),1e-12));q=np.floor(request*factor).astype(int)
    left=budget-q@cost
    for i in np.argsort(-(request*factor-q),kind='stable'):
        if q[i]<request[i] and cost[i]<=left+1e-9:q[i]+=1;left-=cost[i]
    assert q@cost<=budget+1e-6
    return q

def optimise_orders(position,scenarios,unit_cost,sell_price,budget,period):
    # Multiple-choice stochastic inventory problem: choose one integer order option per SKU.
    candidates=[];objectives=[];items=[]
    for i in range(len(position)):
        levels=np.quantile(scenarios[:,i],[.25,.5,.65,.8,.9,.97])
        options=np.unique(np.r_[0,np.maximum(0,np.ceil(levels-position[i])).astype(int)])
        for q in options:
            excess=np.maximum(position[i]+q-scenarios[:,i],0)
            shortage=np.maximum(scenarios[:,i]-position[i]-q,0)
            objectives.append(float((.001*unit_cost[i]*period/2*excess+.5*sell_price[i]*shortage).mean()))
            candidates.append(int(q));items.append(i)
    k=len(items);A=lil_matrix((len(position)+1,k));cash=np.array([candidates[j]*unit_cost[i] for j,i in enumerate(items)])
    for j,i in enumerate(items):A[i,j]=1
    A[-1,:]=cash
    constraints=LinearConstraint(csc_matrix(A),np.r_[np.ones(len(position)),-np.inf],np.r_[np.ones(len(position)),budget])
    result=milp(np.asarray(objectives),integrality=np.ones(k),bounds=Bounds(np.zeros(k),np.ones(k)),constraints=constraints,options={'time_limit':15,'mip_rel_gap':.001})
    if result.x is None:raise RuntimeError('Inventory optimisation found no feasible solution')
    q=np.zeros(len(position),dtype=int)
    for j,i in enumerate(items):
        if result.x[j]>.5:q[i]=candidates[j]
    assert q@unit_cost<=budget+1e-5
    return q,{'status':int(result.status),'mip_gap':float(result.mip_gap),'objective':float(result.fun),'variables':k}

def simulate(data,review_forecasts,residuals,policy,budget_factor=1.,lead=2):
    start=TEST_ORIGINS[0];end=1913;length=end-start;R=7
    # Same starting stock and fixed price/cost basis for all compared policies.
    sell=data.features['price_asof'][:,start-1];cost=.6*sell
    onhand=np.ceil(data.y[:,start-28:start].mean(axis=1)*(R+lead)).astype(float)
    receipts=np.zeros((length+lead+R+1,data.n));records=[];orders=[]
    rng=np.random.default_rng(SEED+lead)
    for t in range(length):
        onhand+=receipts[t]
        if t%R==0:
            origin=start+t;period=R+lead
            trailing=data.y[:,origin-28:origin]
            mean=trailing.mean(axis=1);sd=trailing.std(axis=1,ddof=1)
            budget=float(budget_factor*R*(mean@cost))
            position=onhand+receipts[t+1:].sum(axis=0)
            forecasts=review_forecasts[origin]
            if policy=='Trailing mean cover':
                target=mean*period+sd*np.sqrt(period)
                q=allocate_proportional(target-position,cost,budget);solver=None
            elif policy=='Forecast cover':
                target=forecasts[:,:period].sum(axis=1)+sd*np.sqrt(period)
                q=allocate_proportional(target-position,cost,budget);solver=None
            else:
                # Shared blocks across SKUs preserve observed error correlation.
                draws=[]
                for _ in range(48):
                    fold=int(rng.integers(3));offset=int(rng.integers(0,H-period+1))
                    draws.append(residuals[fold,:,offset:offset+period].sum(axis=1))
                scenarios=np.maximum(0,forecasts[:,:period].sum(axis=1)[None,:]+np.array(draws))
                q,solver=optimise_orders(position,scenarios,cost,sell,budget,period)
            if lead==0:onhand+=q
            else:receipts[t+lead]+=q
            orders.append({'policy':policy,'budget_factor':budget_factor,'lead_time':lead,'origin':origin,'order_units':int(q.sum()),'purchase_spend':float(q@cost),'budget':budget,'solver_status':None if solver is None else solver['status'],'solver_gap':None if solver is None else solver['mip_gap']})
        demand=data.y[:,start+t]
        fulfilled=np.minimum(onhand,demand);lost=demand-fulfilled;onhand-=fulfilled
        assert np.all(onhand>=-1e-8)
        records.append({'policy':policy,'budget_factor':budget_factor,'lead_time':lead,'day':start+t+1,'demand':float(demand.sum()),'fulfilled':float(fulfilled.sum()),'lost':float(lost.sum()),'ending_units':float(onhand.sum()),'inventory_value':float(onhand@cost),'holding_cost':float(onhand@(.001*cost)),'shortage_penalty':float(lost@(.5*sell)),'stockout_item_days':int((lost>0).sum())})
    daily=pd.DataFrame(records);order=pd.DataFrame(orders)
    summary={'policy':policy,'budget_factor':budget_factor,'lead_time':lead,'fill_rate':daily.fulfilled.sum()/daily.demand.sum(),'average_inventory_units':daily.ending_units.mean(),'average_inventory_value':daily.inventory_value.mean(),'holding_cost':daily.holding_cost.sum(),'shortage_penalty':daily.shortage_penalty.sum(),'inventory_operating_cost':daily.holding_cost.sum()+daily.shortage_penalty.sum(),'purchase_spend':order.purchase_spend.sum(),'stockout_item_days':int(daily.stockout_item_days.sum()),'budget_breaches':int((order.purchase_spend>order.budget+1e-5).sum())}
    return summary,daily,order

def savefig(name):plt.tight_layout();plt.savefig(FIG/name,dpi=170,bbox_inches='tight');plt.close()

def main():
    started=time.time();data=RetailData();metrics=[];frames=[];audits=[];models={}
    for origin in VALID_ORIGINS:
        model,audit=data.fit(origin);models[origin]=model;audits.append(audit)
        for name in MODEL_NAMES:
            pred=data.forecast(origin,name,model);actual=data.y[:,origin:origin+H]
            m,sku=forecast_metrics(actual,pred,data.y[:,:origin]);metrics.append(dict(origin=origin,split='validation',model=name,**m))
            frames.append(pd.DataFrame({'origin':origin,'split':'validation','model':name,'item_id':np.repeat(data.items,H),'day':np.tile(np.arange(origin+1,origin+H+1),data.n),'actual':actual.ravel(),'forecast':pred.ravel()}))
        print('Validation origin',origin,'complete',flush=True)
    val=pd.DataFrame(metrics)
    selected=val.groupby('model').wape.mean().idxmin()
    lock={'model':selected,'criterion':'Mean WAPE across three validation origins only','validation_origins':VALID_ORIGINS,'test_origins':TEST_ORIGINS,'seed':SEED,'training_label_rule':'All training target days <= forecast origin','inventory_policy_assumptions':{'base_lead_days':2,'review_days':7,'base_budget_factor':1.,'purchase_cost_fraction_of_price':.6,'daily_holding_cost_fraction_of_unit_cost':.001,'lost_sales_penalty_fraction_of_price':.5}}
    js(OUT/'locked_selection.json',lock)
    print('Locked selected forecast:',selected,flush=True)
    validation_predictions=pd.concat(frames)
    chosen=validation_predictions[validation_predictions.model==selected]
    residuals=np.array([(x.actual-x.forecast).to_numpy().reshape(data.n,H) for _,x in chosen.groupby('origin',sort=True)])
    # Residual calibration uses validation only; fixed for all test horizons.
    err=residuals.transpose(1,0,2).reshape(data.n,-1)
    lower,upper=np.quantile(err,[.1,.9],axis=1)
    coverage=[];review_forecasts={}
    for origin in TEST_ORIGINS:
        model,audit=data.fit(origin);models[origin]=model;audits.append(audit)
        for name in MODEL_NAMES:
            pred=data.forecast(origin,name,model);actual=data.y[:,origin:origin+H]
            m,sku=forecast_metrics(actual,pred,data.y[:,:origin]);metrics.append(dict(origin=origin,split='test',model=name,**m))
            frames.append(pd.DataFrame({'origin':origin,'split':'test','model':name,'item_id':np.repeat(data.items,H),'day':np.tile(np.arange(origin+1,origin+H+1),data.n),'actual':actual.ravel(),'forecast':pred.ravel()}))
            if name==selected:
                lo=np.maximum(0,pred+lower[:,None]);hi=np.maximum(lo,pred+upper[:,None])
                coverage.append({'origin':origin,'nominal_coverage':.8,'empirical_coverage':float(((actual>=lo)&(actual<=hi)).mean()),'mean_width':float((hi-lo).mean())})
        for review in range(origin,origin+H,7):review_forecasts[review]=data.forecast(review,selected,model)
        print('Test origin',origin,'complete',flush=True)
    pd.DataFrame(audits).to_csv(OUT/'training_cutoff_audit.csv',index=False)
    metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'forecast_metrics.csv',index=False)
    forecasts=pd.concat(frames,ignore_index=True);forecasts.to_csv(OUT/'forecasts.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'interval_coverage.csv',index=False)
    reviews=pd.concat([pd.DataFrame({'origin':o,'item_id':np.repeat(data.items,H),'day':np.tile(np.arange(o+1,o+H+1),data.n),'forecast':pred.ravel()}) for o,pred in review_forecasts.items()])
    reviews.to_csv(OUT/'weekly_review_forecasts.csv',index=False)
    policies=['Trailing mean cover','Forecast cover','Scenario optimisation']
    summaries=[];daily=[];orders=[]
    for lead in [0,2,4]:
        for factor in [.8,1.,1.2]:
            for policy in policies:
                s,d,o=simulate(data,review_forecasts,residuals,policy,factor,lead)
                summaries.append(s);daily.append(d);orders.append(o)
            print('Inventory sensitivity complete:',lead,factor,flush=True)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'inventory_sensitivity.csv',index=False)
    all_daily=pd.concat(daily);all_orders=pd.concat(orders)
    all_daily.to_csv(OUT/'inventory_daily.csv',index=False);all_orders.to_csv(OUT/'inventory_orders.csv',index=False)
    base=summary[(summary.lead_time==2)&(summary.budget_factor==1)].copy();base.to_csv(OUT/'inventory_base_case.csv',index=False)
    assert summary.budget_breaches.sum()==0
    # Diagnostics and reproducibility evidence.
    averages=metrics.groupby(['split','model'])[['wape','bias','mean_rmsse']].mean().reset_index();averages.to_csv(OUT/'forecast_comparison.csv',index=False)
    diagnostics={'skus':data.n,'item_days':int(data.y.size),'store':'CA_1','departments':data.sales.dept_id.unique().tolist(),'validation_days':84,'test_days':84,'forecast_horizon':H,'test_start':str(data.cal.loc[1830,'date']),'test_end':str(data.cal.loc[1913,'date']),'selected_model':selected,'optimisation_runs':int(all_orders.solver_status.notna().sum()),'optimal_solver_runs':int((all_orders.solver_status==0).sum()),'budget_breaches':int(summary.budget_breaches.sum()),'engine_audit':json.loads((OUT/'feature_audit.json').read_text())}
    js(OUT/'diagnostics.json',diagnostics)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(3);width=.35
    for j,split in enumerate(['validation','test']):
        vals=averages[averages.split==split].set_index('model').loc[MODEL_NAMES].wape.to_numpy()*100
        ax.bar(x+(j-.5)*width,vals,width,label=split.title(),color=['#64748b','#0f766e'][j])
    ax.set_xticks(x,MODEL_NAMES);ax.set_ylabel('Mean WAPE across origins (%)');ax.set_title('28-day forecast comparison');ax.legend();savefig('forecast_comparison.png')
    actual=data.y[:,1829:1913].sum(axis=0);points=forecasts[(forecasts.split=='test')&(forecasts.model==selected)].groupby('day').forecast.sum()
    fig,ax=plt.subplots(figsize=(10,4));dates=pd.to_datetime(data.cal.loc[np.arange(1830,1914),'date']);ax.plot(dates,actual,label='Observed sales',color='#172b4d');ax.plot(dates,points.to_numpy(),label=selected,color='#0f766e');ax.set_ylabel('Units per day, 60 products');ax.set_title('Test-period aggregate sales and fixed 28-day forecasts');ax.legend();fig.autofmt_xdate();savefig('forecast_timeline.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4));colors=['#64748b','#0f766e','#b45309']
    for ax,col,title in [(axes[0],'inventory_operating_cost','Simulated inventory operating cost ($)'),(axes[1],'fill_rate','Unit fill rate (%)')]:
        vals=base[col].to_numpy()*(100 if col=='fill_rate' else 1);ax.bar(np.arange(3),vals,color=colors);ax.set_xticks(np.arange(3),['Trailing\nmean','Forecast\ncover','Scenario\noptimisation']);ax.set_title(title)
        for i,v in enumerate(vals):ax.text(i,v*1.01,f'{v:,.1f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(vals)*1.15)
    savefig('inventory_comparison.png')
    fig,ax=plt.subplots(figsize=(9,4))
    base_daily=all_daily[(all_daily.lead_time==2)&(all_daily.budget_factor==1)]
    for name,color in zip(policies,colors):
        tmp=base_daily[base_daily.policy==name];ax.plot(tmp.day,np.cumsum(tmp.holding_cost+tmp.shortage_penalty),label=name,color=color)
    ax.set_xlabel('M5 day index');ax.set_ylabel('Cumulative simulated operating cost ($)');ax.set_title('Inventory decisions under the same purchasing budget');ax.legend();savefig('inventory_timeline.png')
    runtime={'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'seconds':round(time.time()-started,2)};js(OUT/'runtime.json',runtime)
    print(averages.to_string(index=False));print(base.to_string(index=False));print('Complete:',runtime,flush=True)
if __name__=='__main__':main()
