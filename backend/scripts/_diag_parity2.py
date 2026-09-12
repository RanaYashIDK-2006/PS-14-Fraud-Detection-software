import pandas as pd, numpy as np, time
t0=time.time()
S='data/_sorted_cache/altman_sorted.parquet'
F='data/_sorted_cache/features_all.parquet'
s = pd.read_parquet(S, columns=['User','Card','Merchant Name','Merchant City','amt','is_fraud'])
f = pd.read_parquet(F, columns=['user_tx_count','card_tx_count','merch_tx_count','user_avg_amt','amt_zscore','user_fraud_rate','merch_fraud_rate','city_fraud_rate'])
print('loaded', round(time.time()-t0,1))
N=len(s)
print('N', N)
users=s['User'].values; cards=s['Card'].values; merch=s['Merchant Name'].values; cities=s['Merchant City'].values
amts=s['amt'].astype(np.float32).values; labels=s['is_fraud'].astype(int).values

# single continuous pass computing per-row values for ALL rows (fast ints) but store only checkpoints
us_st={}; cs_st={}; ms_st={}; cts_st={}
utc=np.zeros(N,dtype=np.float32); ctc=np.zeros(N,dtype=np.float32); mtc=np.zeros(N,dtype=np.float32)
ua=np.zeros(N,dtype=np.float32); az=np.zeros(N,dtype=np.float32)
ufr=np.full(N,0.001,dtype=np.float32); mfr=np.full(N,0.001,dtype=np.float32); cfr=np.full(N,0.001,dtype=np.float32)
t1=time.time()
for i in range(N):
    uid=int(users[i]); cid=int(cards[i]); mid=int(merch[i]); ctid=int(cities[i])
    a=float(amts[i]); lab=labels[i]
    us=us_st.get(uid)
    if us is None:
        us={'c':0,'s':0.0,'ss':0.0,'fc':0}; us_st[uid]=us
    cs=cs_st.get(cid)
    if cs is None: cs={'c':0}; cs_st[cid]=cs
    ms=ms_st.get(mid)
    if ms is None: ms={'c':0,'fc':0,'t':0}; ms_st[mid]=ms
    cts=cts_st.get(ctid)
    if cts is None: cts={'fc':0,'t':0}; cts_st[ctid]=cts
    utc[i]=us['c']; ctc[i]=cs['c']; mtc[i]=ms['c']
    if us['c']>0:
        ua[i]=us['s']/us['c']
        if us['c']>=2:
            mean=us['s']/us['c']; var=max(us['ss']/us['c']-mean**2,0.0)
            az[i]=(a-mean)/(var**0.5+1e-6)
        else:
            az[i]=(a-us['s'])/(us['s']+1e-6)
    if us['c']>0: ufr[i]=us['fc']/us['c']
    if ms['t']>0: mfr[i]=ms['fc']/ms['t']
    if cts['t']>0: cfr[i]=cts['fc']/cts['t']
    us['c']+=1; us['s']+=a; us['ss']+=a*a; us['fc']+=lab
    cs['c']+=1
    ms['c']+=1; ms['fc']+=lab; ms['t']+=1
    cts['fc']+=lab; cts['t']+=1
    if i % 3_000_000 == 0 and i>0:
        print(f'  pass row {i:,} ({time.time()-t1:.0f}s)')
print(f'pass done {time.time()-t1:.0f}s')

def cmp_col(col, sim):
    c=f[col].values
    for (i0,i1,label) in [(0,400_000,'head'), (N//4,N//4+400_000,'quarter'), (N//2,N//2+400_000,'mid')]:
        d=np.abs(sim[i0:i1].astype(np.float64)-c[i0:i1].astype(np.float64))
        print(f'  {col} [{label} {i0:,}]: max={d.max():.6f} mean={d.mean():.8f} n>1e-3={int((d>1e-3).sum())}')
cmp_col('user_tx_count',utc); cmp_col('card_tx_count',ctc); cmp_col('merch_tx_count',mtc)
cmp_col('user_avg_amt',ua); cmp_col('amt_zscore',az)
cmp_col('user_fraud_rate',ufr); cmp_col('merch_fraud_rate',mfr); cmp_col('city_fraud_rate',cfr)
