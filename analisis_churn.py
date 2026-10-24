"""Experimento reproducible. No usar el holdout para elegir modelos ni umbrales."""
from pathlib import Path
import sys
if (Path(__file__).parent / 'ml_direct').exists():
    sys.path.insert(0, str(Path(__file__).parent / 'ml_direct'))
import hashlib
import json
import platform
import time
import urllib.request
import numpy as np
import pandas as pd
import sklearn
import imblearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix, brier_score_loss)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

SEED = 42
URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
SHA256 = '16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91'
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados'
OUT.mkdir(exist_ok=True)

def top_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    k = int(np.ceil(.2 * len(y)))
    # Orden estable. El orden inicial del archivo resuelve empates sin consultar y.
    ix = np.argsort(-p, kind='stable')[:k]
    hits = int(y[ix].sum())
    return {'p20': hits/k, 'r20': hits/int(y.sum()), 'lift20': (hits/k)/y.mean(),
            'k': k, 'hits': hits}

def metrics(y, p, threshold=.5):
    pred = np.asarray(p) >= threshold
    return {'roc_auc':roc_auc_score(y,p), 'ap':average_precision_score(y,p),
            'precision':precision_score(y,pred,zero_division=0),
            'recall':recall_score(y,pred), 'f1':f1_score(y,pred),
            'accuracy':accuracy_score(y,pred), 'brier':brier_score_loss(y,p),
            **top_metrics(y,p)}

def prepare(frame):
    x = frame.drop(columns=['customerID','Churn','gender','SeniorCitizen']).copy()
    x['TotalCharges'] = pd.to_numeric(x['TotalCharges'],errors='coerce')
    x['total_sin_facturar'] = x['TotalCharges'].isna().map({True:'Yes',False:'No'})
    x.loc[x['TotalCharges'].isna() & x['tenure'].eq(0),'TotalCharges'] = 0.
    services=['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    x['cantidad_servicios'] = frame[services].eq('Yes').sum(axis=1)
    x['pago_automatico'] = frame['PaymentMethod'].str.contains('automatic').map({True:'Yes',False:'No'})
    return x

def preprocessor(x):
    num = x.select_dtypes(include='number').columns.tolist()
    cat = [c for c in x if c not in num]
    return ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),
                         ('scale',StandardScaler())]),num),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                         ('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)
    ])

def smotenc_pipeline(x):
    num=x.select_dtypes(include='number').columns.tolist()
    cat=[c for c in x if c not in num]
    before=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                         ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),cat)])
    cat_ix=list(range(len(num),len(num)+len(cat)))
    after=ColumnTransformer([('num','passthrough',list(range(len(num)))),
                            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat_ix)])
    return ImbPipeline([('before',before),('smotenc',SMOTENC(categorical_features=cat_ix,random_state=SEED)),
                        ('after',after),('model',LogisticRegression(C=1,max_iter=2000,random_state=SEED))])

def run():
    data=ROOT/'Telco-Customer-Churn.csv'
    if not data.exists():
        data.write_bytes(urllib.request.urlopen(URL).read())
    assert hashlib.sha256(data.read_bytes()).hexdigest()==SHA256, 'Cambió el archivo de datos'
    raw=pd.read_csv(data)
    assert len(raw)==7043 and raw.shape[1]==21
    assert raw['customerID'].is_unique
    y=raw['Churn'].eq('Yes').astype(int)
    idx_train,idx_test=train_test_split(np.arange(len(raw)),test_size=.2,stratify=y,random_state=SEED)
    # Se persisten índices antes del análisis y de las búsquedas.
    pd.DataFrame({'row':np.r_[idx_train,idx_test],
                  'split':['train']*len(idx_train)+['test']*len(idx_test)}).to_csv(OUT/'particion.csv',index=False)
    X=prepare(raw)
    Xtr,Xte=X.iloc[idx_train].reset_index(drop=True),X.iloc[idx_test].reset_index(drop=True)
    ytr,yte=y.iloc[idx_train].reset_index(drop=True),y.iloc[idx_test].reset_index(drop=True)
    trainraw=raw.iloc[idx_train].copy()
    trainraw['target']=y.iloc[idx_train]
    eda={}
    for field in ['Contract','InternetService','PaymentMethod']:
        eda[field]=trainraw.groupby(field)['target'].agg(['count','sum','mean']).reset_index().to_dict('records')
    trainraw['antiguedad']=pd.cut(trainraw['tenure'],[-1,12,24,48,72],labels=['0 a 12','13 a 24','25 a 48','49 a 72'])
    eda['tenure']=trainraw.groupby('antiguedad',observed=True)['target'].agg(['count','sum','mean']).reset_index().to_dict('records')
    nt=pd.to_numeric(trainraw['TotalCharges'],errors='coerce')
    eda['corr_total_tenure']=float(nt.corr(trainraw['tenure']))
    eda['corr_total_monthly']=float(nt.corr(trainraw['MonthlyCharges']))
    eda['ranges']={c:[float(pd.to_numeric(trainraw[c],errors='coerce').min()),float(pd.to_numeric(trainraw[c],errors='coerce').max())] for c in ['tenure','MonthlyCharges','TotalCharges']}
    eda['median_monthly_by_churn']=trainraw.groupby('Churn')['MonthlyCharges'].median().to_dict()
    blank=pd.to_numeric(raw['TotalCharges'],errors='coerce').isna()
    quality={'n':len(raw),'columns':raw.shape[1],'positive':int(y.sum()),'prevalence':float(y.mean()),
             'missing_total':int(blank.sum()),'missing_total_tenure0':int((blank & raw['tenure'].eq(0)).sum()),
             'duplicate_ids':int(raw['customerID'].duplicated().sum()),'duplicate_rows':int(raw.duplicated().sum()),
             'other_missing':int(raw.drop(columns='TotalCharges').isna().sum().sum()),
             'n_train':len(ytr),'n_test':len(yte),'train_positive':int(ytr.sum()),'test_positive':int(yte.sum()),
             'features':X.columns.tolist()}
    specs=[]
    for c in [.1,1,10]:
        specs.append((f'LR C={c:g}', 'LR', Pipeline([('pre',preprocessor(Xtr)),('model',LogisticRegression(C=c,max_iter=2000,random_state=SEED))]),False))
    specs.append(('LR balanced','LR',Pipeline([('pre',preprocessor(Xtr)),('model',LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED))]),False))
    for leaf in [3,10]:
        specs.append((f'RF hoja={leaf}','RF',Pipeline([('pre',preprocessor(Xtr)),('model',RandomForestClassifier(n_estimators=250,min_samples_leaf=leaf,max_features='sqrt',random_state=SEED,n_jobs=2))]),False))
    for depth in [2,3]:
        specs.append((f'XGB profundidad={depth}','XGB',Pipeline([('pre',preprocessor(Xtr)),('model',XGBClassifier(n_estimators=200,max_depth=depth,learning_rate=.04,subsample=.8,colsample_bytree=.8,reg_lambda=1,random_state=SEED,n_jobs=2,eval_metric='logloss'))]),False))
    specs.append(('LR SMOTENC','SMOTENC',smotenc_pipeline(Xtr),False))
    ab=Xtr.drop(columns='TotalCharges')
    specs.append(('LR sin TotalCharges','Ablacion',Pipeline([('pre',preprocessor(ab)),('model',LogisticRegression(C=1,max_iter=2000,random_state=SEED))]),True))
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    allrows=[]; oofs={}; foldrows=[]
    for name,family,est,drop_total in specs:
        t=time.time(); xx=Xtr.drop(columns='TotalCharges') if drop_total else Xtr
        oof=np.zeros(len(ytr)); ms=[]
        for fold,(a,b) in enumerate(cv.split(xx,ytr),1):
            model=clone(est).fit(xx.iloc[a],ytr.iloc[a])
            p=model.predict_proba(xx.iloc[b])[:,1]; oof[b]=p
            m=metrics(ytr.iloc[b],p); ms.append(m)
            foldrows.append({'name':name,'fold':fold,**m})
        row={'name':name,'family':family}
        for key in ['p20','r20','roc_auc','ap','precision','recall','f1','accuracy']:
            row[key]=float(np.mean([m[key] for m in ms])); row[key+'_sd']=float(np.std([m[key] for m in ms],ddof=1))
        allrows.append(row); oofs[name]=oof
        print(name, 'P20',round(row['p20'],4),'AP',round(row['ap'],4),'seg',round(time.time()-t,1),flush=True)
    rows=pd.DataFrame(allrows)
    eligible=rows[rows.family.ne('Ablacion')]
    bestp=eligible.p20.max()
    # Regla previa: dentro de 1 pp del mejor P20, preferir LR sin remuestreo.
    close=eligible[eligible.p20>=bestp-.01]
    simple=close[close['name'].isin(['LR C=0.1','LR C=1','LR C=10'])]
    selected=(simple if len(simple) else close).sort_values(['p20','ap'],ascending=False).iloc[0]
    name=selected['name']; p_oof=oofs[name]
    choices=[]
    for threshold in np.linspace(.05,.8,151):
        pred=p_oof>=threshold
        pr=precision_score(ytr,pred,zero_division=0); re=recall_score(ytr,pred)
        if pr>=.5 and re>=.75:
            choices.append((f1_score(ytr,pred),float(threshold),pr,re))
    chosen_threshold=max(choices)[1] if choices else .5
    # Decisión registrada ANTES de predecir/evaluar el holdout.
    decision={'selected':name,'selection_rule':'P20 medio de CV; tolerancia 1 pp para LR simple; desempate AP',
              'threshold_oof':chosen_threshold,'threshold_target_feasible_oof':bool(choices)}
    (OUT/'decision_previa.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    spec=next(s for s in specs if s[0]==name)
    final=clone(spec[2]).fit(Xtr,ytr)
    p=final.predict_proba(Xte)[:,1]
    test_default=metrics(yte,p)
    test_threshold=metrics(yte,p,chosen_threshold)
    k=int(np.ceil(.2*len(yte))); ix=np.argsort(-p,kind='stable')[:k]
    campaign=np.zeros(len(yte),dtype=int); campaign[ix]=1
    test_default['confusion_05']=confusion_matrix(yte,p>=.5).tolist()
    test_threshold['confusion']=confusion_matrix(yte,p>=chosen_threshold).tolist()
    campaign_cm=confusion_matrix(yte,campaign).tolist()
    rng=np.random.default_rng(SEED)
    boots=[]
    for _ in range(2000):
        b=rng.integers(0,len(yte),len(yte))
        boots.append(top_metrics(yte.iloc[b],p[b])['p20'])
    ci=np.quantile(boots,[.025,.975]).tolist()
    audits={}
    for col in ['gender','SeniorCitizen']:
        fr=raw.iloc[idx_test][[col]].reset_index(drop=True)
        fr['y']=yte; fr['score']=p; fr['contact']=campaign
        audits[col]=[]
        for group,g in fr.groupby(col):
            hits=int((g.y*g.contact).sum()); contacts=int(g.contact.sum())
            audits[col].append({'grupo':str(group),'n':len(g),'positivos':int(g.y.sum()),
                               'contactos':contacts,'seleccion':float(g.contact.mean()),
                               'p20_grupo':hits/contacts if contacts else None,
                               'recall_grupo':hits/int(g.y.sum()) if g.y.sum() else None})
    # Coeficientes descriptivos del modelo elegido si es regresión logística.
    coefficients=[]
    if name.startswith('LR C='):
        features=final.named_steps['pre'].get_feature_names_out()
        co=final.named_steps['model'].coef_[0]
        coefficients=pd.DataFrame({'feature':features,'coefficient':co}).sort_values('coefficient').to_dict('records')
    pd.DataFrame({'row':idx_test,'y_true':yte,'score':p,'top20':campaign}).to_csv(OUT/'predicciones_test.csv',index=False)
    rows.to_csv(OUT/'comparacion_cv.csv',index=False)
    pd.DataFrame(foldrows).to_csv(OUT/'metricas_folds.csv',index=False)
    pd.DataFrame({'row':idx_train,'y_true':ytr,'oof_selected':p_oof}).to_csv(OUT/'predicciones_oof.csv',index=False)
    result={'quality':quality,'eda_train':eda,'cv':allrows,'decision':decision,
            'test_05':test_default,'test_threshold':test_threshold,'campaign_confusion':campaign_cm,
            'p20_bootstrap95':ci,'audit':audits,'coefficients':coefficients,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                        'scikit-learn':sklearn.__version__,'imbalanced-learn':imblearn.__version__,'xgboost':xgboost.__version__},
            'url':URL,'sha256':SHA256}
    (OUT/'resultados.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'decision':decision,'test':test_default,'threshold':test_threshold,'ci':ci},indent=2),flush=True)
    return result

if __name__=='__main__':
    run()
