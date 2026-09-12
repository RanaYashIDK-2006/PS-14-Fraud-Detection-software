import pandas as pd, numpy as np, time
t0=time.time()
S='data/_sorted_cache/altman_sorted.parquet'
F='data/_sorted_cache/features_all.parquet'
s = pd.read_parquet(S, columns=['User','Card','Merchant Name','Merchant City','amt','is_fraud','hr','chip','is_online','mcc_n','has_zip','has_state'])
f = pd.read_parquet(F, columns=['user_tx_count','merch_tx_count','user_fraud_rate','merch_fraud_rate','city_fraud_rate','log_amt','amt_sq','chip','is_online','mcc_n','has_zip','has_state','hour_cos','is_business_hours'])
print('loaded', time.time()-t0)
print('sorted dtypes User:', s['User'].dtype, 'merch:', s['Merchant Name'].dtype)
print('n unique users:', s['User'].nunique(), 'merchants:', s['Merchant Name'].nunique())
# row 0 deterministic cols
for c in ['log_amt','chip','is_online','mcc_n','has_zip','has_state']:
    a = (np.log1p(s['amt']) if c=='log_amt' else s[c]).values[:5]
    b = f[c].values[:5]
    print(f'row0 {c}: sorted={a} cache={b} match={np.allclose(a,b,atol=1e-3)}')
# user_tx_count at row 0..5 should be 0 (first user rows)
print('cache user_tx_count row0-5:', f['user_tx_count'].values[:6])
print('cache merch_tx_count row0-5:', f['merch_tx_count'].values[:6])
# Check first user transitions: find where User changes
u = s['User'].values
first_rows_user0 = np.where(u==u[0])[0]
print('user0 occupies rows 0..', first_rows_user0[-1], 'count', len(first_rows_user0))
# row indices where user_tx_count in cache resets: check cache vs sorted alignment on user changes
utc = f['user_tx_count'].values
# simulate expected user_tx_count if cache aligned to sorted rows and per-user cumulative, starting 0
from collections import Counter
cnt = Counter()
sim = np.zeros(len(s), dtype=np.int32)
us = s['User'].values
for i in range(min(len(s), 3_000_000)):
    sim[i] = cnt[us[i]]
    cnt[us[i]] += 1
m = (sim[:3_000_000] == utc[:3_000_000]).mean()
print('aligned-user_tx sim match (first 3M):', m)
d = np.where(sim[:3_000_000] != utc[:3_000_000])[0]
print('first 5 mismatch rows:', d[:5])
if len(d):
    i = d[0]
    print(f'row {i}: user={us[i]} sim={sim[i]} cache={utc[i]}')
    # what does cache think at prev row?
    print(f'row {i-1}: user={us[i-1]} sim={sim[i-1]} cache={utc[i-1]}')
    print(f'row {i+1}: user={us[i+1]} sim={sim[i+1]} cache={utc[i+1]}')
