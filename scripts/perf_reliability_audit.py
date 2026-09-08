# -*- coding: utf-8 -*-
"""
PS-14 Check #26 -- END-TO-END PERFORMANCE, LATENCY & RELIABILITY AUDIT

Measures the live decision path against running services (launch_perf_stack.py
must have booted identity:8001 / privacy:8002 / risk:8003 on db/perf_audit):

  ingest-transaction (privacy: feature retrieval/calculation + DB-2 write)
      -> features -> /internal/evaluate (risk: mapper -> model -> decision
         -> DB-3 alert row -> background audit append on DB-4)

Output: reports/perf_reliability_audit.json (+ console)
"""
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "perf_reliability_audit.json"
DB_DIR = ROOT / "db" / "perf_audit"
REPORT = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

import httpx  # noqa: E402

# ---- env: internal token (same value the services loaded from .env) --------
TOKEN = ""
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("INTERNAL_TOKEN="):
        TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
assert TOKEN, "INTERNAL_TOKEN missing from .env"
HDRS = {"X-Internal-Token": TOKEN, "Content-Type": "application/json"}
PRIVACY = "http://127.0.0.1:8002"
RISK = "http://127.0.0.1:8003"
T0 = time.time()
client = httpx.Client(timeout=30.0, headers=HDRS)


def log(m):
    print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def percentiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    def p(q):
        i = min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))
        return round(xs[i], 3)
    return {"p50": p(50), "p95": p(95), "p99": p(99), "max": round(max(xs), 3), "n": len(xs)}


def gen_fraud_id(rng):
    return "F" + "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")) for _ in range(15))


def gen_event(rng):
    return rng.choice(["EVT", "TXN", "SPD", "PAY", "CHK"]) + \
        hashlib.sha1(rng.randbytes(8)).hexdigest()[:24].upper()


def ingest_payload(rng, fraud_id):
    return {
        "event_id": gen_event(rng), "fraud_id": fraud_id,
        "amount": round(rng.uniform(2.0, 4000.0), 2),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hour_of_day": rng.randint(0, 23),
        "device_id": f"dev-{rng.randint(1, 200)}",
        "location_id": f"loc-{rng.randint(1, 50)}",
        "recipient_id": f"recip-{rng.randint(1, 200)}",
        "failed_auth_count_24h": rng.choice([0, 0, 0, 1, 2]),
    }


EVAL_FEATURE_KEYS = {
    "amount_ratio", "txn_freq_last_24h", "txn_time_unusual", "new_device_flag",
    "unusual_location_flag", "unusual_recipient_flag", "failed_auth_count_24h",
    "days_since_last_similar_txn", "gradual_escalation_score", "known_device_count",
    "account_tenure_days", "hour_of_day", "is_weekend", "shared_device_accounts",
    "shared_recipient_accounts", "mule_ring_score", "account_daily_spend_ratio",
    "device_daily_count", "hour_deviation", "amount_zscore", "velocity_deviation",
    "recipient_novelty", "txn_regularity", "user_id", "merchant_id", "city_id",
    "user_tx_count", "user_avg_amt", "card_tx_count", "merch_tx_count",
}


def ingest_one(rng, fraud_id):
    return client.post(f"{PRIVACY}/internal/ingest-transaction", json=ingest_payload(rng, fraud_id))


def evaluate_one(rng, fraud_id, event_id=None):
    """Full e2e path: ingest -> forward features -> evaluate. If event_id given,
    skip ingest and re-score the existing event (idempotency probe)."""
    if event_id is None:
        r0 = ingest_one(rng, fraud_id)
        if r0.status_code != 200:
            return r0
        body = r0.json()
    else:
        body = {"event_id": event_id, "fraud_id": fraud_id}
        # rebuild features by fetching the stored vector shape from a live ingest
        r0 = ingest_one(rng, fraud_id)
        body = {**r0.json(), "event_id": event_id}
    feats = {k: v for k, v in body.items() if k in EVAL_FEATURE_KEYS}
    req = {"event_id": body["event_id"], "fraud_id": fraud_id, "features": feats}
    return client.post(f"{RISK}/internal/evaluate", json=req)


def load_run(workers, fn, n_requests):
    """Run fn() n_requests times across `workers` threads. Returns (lats_ms, errs, dur_s)."""
    lats, errs = [], []
    t0 = time.monotonic()

    def work(_):
        for _ in range(max(1, n_requests // workers)):
            t = time.monotonic()
            try:
                r = fn()
                dt = (time.monotonic() - t) * 1000
                if r.status_code == 200:
                    lats.append(dt)
                else:
                    errs.append((r.status_code, r.text[:120]))
            except Exception as e:
                errs.append(("EXC", str(e)[:120]))

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, range(workers)))
    return lats, errs, time.monotonic() - t0


def timed_loop(workers, fn, window_s):
    """Run fn() in a loop for window_s seconds across `workers` threads."""
    lats, errs = [], []
    stop = time.monotonic() + window_s

    def work(_):
        while time.monotonic() < stop:
            t = time.monotonic()
            try:
                r = fn()
                dt = (time.monotonic() - t) * 1000
                if r.status_code == 200:
                    lats.append(dt)
                else:
                    errs.append((r.status_code, r.text[:100]))
            except Exception as e:
                errs.append(("EXC", str(e)[:100]))

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, range(workers)))
    return lats, errs


def rss_kb(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines()[1:]:
            if f'"{pid}"' in line:
                parts = [p.strip('"') for p in line.split('","')]
                if len(parts) > 4:
                    return int(parts[4].replace(",", ""))
    except Exception:
        pass
    return None


def build_feature_cache(rng, fraud_ids):
    """One ingest per fraud_id -> cached evaluate-ready feature dicts.
    Used by evaluate-only tests so the measured path excludes ingest."""
    cache = {}
    for fid in fraud_ids:
        r = ingest_one(rng, fid)
        if r.status_code != 200:
            raise RuntimeError(f"ingest failed {r.status_code} {r.text[:200]}")
        body = r.json()
        cache[fid] = {k: v for k, v in body.items() if k in EVAL_FEATURE_KEYS}
    return cache


def main():
    pids = json.load(open(ROOT / "db" / "perf_stack_pids.json"))
    risk_pid = pids["risk"]["pid"]
    rng = random.Random(1234)
    fraud_pool = [gen_fraud_id(rng) for _ in range(60)]

    # stage 0: warm-up + feature cache ------------------------------------
    log("stage 0: warm-up (20 e2e txns) + feature cache...")
    for i in range(20):
        r = evaluate_one(rng, fraud_pool[i % len(fraud_pool)])
        if r.status_code != 200:
            log(f"  warm-up failure: {r.status_code} {r.text[:300]}")
            sys.exit(3)
    feat_cache = build_feature_cache(rng, fraud_pool)
    log("warm-up OK")

    # stage 1: in-process model microbench ---------------------------------
    log("stage 1: in-process AltmanEnsembleEngine microbench (same artifacts risk uses)...")
    sys.path.insert(0, str(ROOT))
    from src.risk_engine.altman_ensemble import AltmanEnsembleEngine
    eng = AltmanEnsembleEngine(ROOT / "models" / "production")
    log(f"  engine model_version={eng.model_version}")
    base = {"amount_ratio": 1.0, "txn_freq_last_24h": 2, "new_device_flag": 0,
            "hour_of_day": 14, "amount": 120.0, "user_id": "u1", "merchant_id": "m1",
            "city_id": "c1", "user_tx_count": 10, "merch_tx_count": 100}
    t0 = time.monotonic()
    for _ in range(300):
        eng.predict(dict(base))
    single_ms = (time.monotonic() - t0) * 1000 / 300
    rows = []
    for i in range(10000):
        f = dict(base)
        f["amount"] = float(i % 5000)
        f["merchant_id"] = f"m{i % 300}"
        rows.append(f)
    t0 = time.monotonic()
    eng.predict_many(rows)
    batch_ms = (time.monotonic() - t0) * 1000 / len(rows)
    REPORT["model_microbench"] = {
        "engine": eng.model_version,
        "single_row_ms": round(single_ms, 4),
        "single_row_rps": round(1000 / single_ms, 1),
        "batch_10k_per_row_ms": round(batch_ms, 4),
        "batch_10k_rows_per_s": round(1000 / batch_ms, 1),
    }
    log(f"  single-row {single_ms:.3f} ms/row | batch-10k {batch_ms:.4f} ms/row "
        f"({1000/batch_ms:,.0f} rows/s)")

    # stage 2: per-stage latency -------------------------------------------
    log("stage 2: per-stage latency (2 workers)...")
    fid = fraud_pool[0]
    feats_fid = feat_cache[fid]

    def ev_only():
        return client.post(f"{RISK}/internal/evaluate", json={
            "event_id": gen_event(rng), "fraud_id": fid, "features": feats_fid})

    l1, e1, d1 = load_run(2, lambda: ingest_one(rng, fid), 150)
    REPORT["latency_ingest"] = percentiles(l1)
    REPORT["latency_ingest"]["errors"] = len(e1)
    l2, e2, d2 = load_run(2, ev_only, 150)
    REPORT["latency_evaluate"] = percentiles(l2)
    REPORT["latency_evaluate"]["errors"] = len(e2)
    l3, e3, d3 = load_run(2, lambda: evaluate_one(rng, fid), 120)
    REPORT["latency_e2e"] = percentiles(l3)
    REPORT["latency_e2e"]["errors"] = len(e3)
    REPORT["estimated_feature_gen_p50_ms"] = round(
        max(0.0, percentiles(l3)["p50"] - percentiles(l2)["p50"]), 1)
    log(f"  ingest p50={percentiles(l1)['p50']}ms eval p50={percentiles(l2)['p50']}ms "
        f"e2e p50={percentiles(l3)['p50']}ms (feat-gen est {REPORT['estimated_feature_gen_p50_ms']}ms)")

    # stage 3: concurrency ramp (evaluate-only, no ingest) ------------------
    log("stage 3: concurrency ramp (evaluate-only)...")
    ramp_eval = {}
    for workers in [1, 4, 8, 16, 32, 64, 128]:
        n = 600 if workers <= 16 else 900

        def ev_only_ramp():
            f2 = fraud_pool[random.randrange(len(fraud_pool))]
            return client.post(f"{RISK}/internal/evaluate", json={
                "event_id": gen_event(rng), "fraud_id": f2, "features": feat_cache[f2]})

        lats, errs, dur = load_run(workers, ev_only_ramp, n)
        res = percentiles(lats)
        res["throughput_rps"] = round(len(lats) / dur, 1)
        res["errors"] = len(errs)
        res["error_samples"] = errs[:3]
        res["duration_s"] = round(dur, 2)
        ramp_eval[str(workers)] = res
        log(f"  w={workers}: tp={res['throughput_rps']}/s p50={res['p50']}ms "
            f"p95={res['p95']}ms p99={res['p99']}ms errors={len(errs)} {errs[:1]}")
    REPORT["ramp_evaluate"] = ramp_eval

    # stage 4: concurrency ramp (full e2e chain) ----------------------------
    log("stage 4: concurrency ramp (full e2e chain)...")
    ramp_e2e = {}
    for workers in [1, 4, 8, 16, 32]:
        n = 300 if workers <= 16 else 400
        lats, errs, dur = load_run(workers, lambda: evaluate_one(rng, fraud_pool[random.randrange(len(fraud_pool))]), n)
        res = percentiles(lats)
        res["throughput_rps"] = round(len(lats) / dur, 1)
        res["errors"] = len(errs)
        res["error_samples"] = errs[:3]
        res["duration_s"] = round(dur, 2)
        ramp_e2e[str(workers)] = res
        log(f"  w={workers}: tp={res['throughput_rps']}/s p50={res['p50']}ms "
            f"p95={res['p95']}ms p99={res['p99']}ms errors={len(errs)} {errs[:1]}")
    REPORT["ramp_e2e"] = ramp_e2e

    # stage 5: sustained load ----------------------------------------------
    log("stage 5: sustained load (16 workers e2e, 45s)...")
    lats, errs = timed_loop(16, lambda: evaluate_one(rng, fraud_pool[random.randrange(len(fraud_pool))]), 45)
    res = percentiles(lats)
    res["throughput_rps"] = round(len(lats) / 45.0, 1)
    res["errors"] = len(errs)
    res["error_samples"] = errs[:5]
    REPORT["sustained"] = res
    REPORT["rss_kb_during_sustained"] = rss_kb(risk_pid)
    log(f"  sustained: ok={res['n']} errors={res['errors']} tp={res['throughput_rps']}/s "
        f"p50={res['p50']}ms p95={res['p95']}ms p99={res['p99']}ms rss={REPORT['rss_kb_during_sustained']}KB")

    # stage 6: traffic spike -----------------------------------------------
    log("stage 6: sudden spike (128 workers, 8s)...")
    lats, errs = timed_loop(128, lambda: evaluate_one(rng, fraud_pool[random.randrange(len(fraud_pool))]), 8)
    dur = 8.0
    res = percentiles(lats)
    res["throughput_rps"] = round(len(lats) / dur, 1)
    res["errors"] = len(errs)
    res["error_samples"] = errs[:5]
    REPORT["spike"] = res
    log(f"  spike: ok={res['n']} errors={res['errors']} tp={res['throughput_rps']}/s "
        f"p50={res['p50']}ms p95={res['p95']}ms p99={res['p99']}ms")

    # stage 7: idempotency + concurrent duplicates -------------------------
    log("stage 7: idempotency + concurrent duplicates...")
    fid = fraud_pool[5]
    r1 = evaluate_one(rng, fid)
    b1 = r1.json()
    ev = b1["event_id"]
    r2 = evaluate_one(rng, fid, event_id=ev)
    replay = r2.json()

    def dup(_):
        rr = client.post(f"{RISK}/internal/evaluate", json={
            "event_id": ev, "fraud_id": fid,
            "features": {k: v for k, v in b1.items() if k in EVAL_FEATURE_KEYS}})
        try:
            return rr.json()
        except Exception:
            return {"status": rr.status_code}

    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        dups = list(ex.map(dup, range(10)))
    scores = {d.get("risk_score") for d in dups if d.get("risk_score") is not None}
    con = sqlite3.connect(DB_DIR / "risk.db")
    n_rows = con.execute("SELECT COUNT(*) FROM risk_scores WHERE event_id=?", (ev,)).fetchone()[0]
    con.close()
    con_a = sqlite3.connect(DB_DIR / "audit.db")
    n_audit = con_a.execute(
        "SELECT COUNT(*) FROM audit_events WHERE event_id=? AND event_type='score_generated'", (ev,)).fetchone()[0]
    con_a.close()
    idem = {
        "sequential_replay_same_score": b1["risk_score"] == replay["risk_score"],
        "concurrent_duplicate_unique_scores": sorted(scores),
        "risk_rows_for_event": n_rows,
        "audit_events_for_event": n_audit,
        "decision": b1.get("decision"),
    }
    REPORT["idempotency"] = idem
    log(f"  idem: replay_same={idem['sequential_replay_same_score']} "
        f"scores={idem['concurrent_duplicate_unique_scores']} risk_rows={n_rows} audit={n_audit}")

    # stage 8: failure injection -------------------------------------------
    log("stage 8: failure injection...")
    fail = {}

    log("  8a: DB-4 (audit.db) exclusive lock while scoring...")
    con_audit = sqlite3.connect(DB_DIR / "audit.db", timeout=1)
    con_audit.execute("BEGIN EXCLUSIVE")
    lats_locked, errs_locked = [], []
    for _ in range(15):
        t = time.monotonic()
        try:
            r = evaluate_one(rng, fid)
            dt = (time.monotonic() - t) * 1000
            if r.status_code == 200:
                lats_locked.append(dt)
            else:
                errs_locked.append((r.status_code, r.text[:120]))
        except Exception as e:
            errs_locked.append(("EXC", str(e)[:120]))
    con_audit.rollback()
    con_audit.close()
    time.sleep(5)
    con_a = sqlite3.connect(DB_DIR / "audit.db")
    total_audit = con_a.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    con_a.close()
    base_p95 = ramp_eval["16"]["p95"]
    fail["db4_lock"] = {
        "responses_ok_during_lock": len(lats_locked),
        "errors_during_lock": len(errs_locked),
        "latency_p95_during_lock_ms": percentiles(lats_locked).get("p95"),
        "baseline_p95_ms": base_p95,
        "decisions_not_blocked": len(errs_locked) == 0,
        "audit_events_total_after_drain": total_audit,
        "note": "audit append is queued/async; scoring does not block on DB-4, but audit "
                "durability depends on queue drain (writer thread opens its own SQLite session).",
    }
    log(f"  8a: ok={len(lats_locked)} err={len(errs_locked)} p95={percentiles(lats_locked).get('p95')}ms "
        f"(baseline p95 {base_p95}ms) audit_total={total_audit}")

    log("  8b: NaN / extreme feature values...")
    r = evaluate_one(rng, fid)
    feats0 = {k: v for k, v in r.json().items() if k in EVAL_FEATURE_KEYS}
    nan_body = json.dumps({"event_id": gen_event(rng), "fraud_id": fid,
                           "features": {**feats0, "amount_zscore": float("nan")}})
    r_nan = client.post(f"{RISK}/internal/evaluate", content=nan_body)
    nan_res = {}
    if r_nan.status_code == 200:
        j = r_nan.json()
        nan_res = {"status": 200, "risk_score": j.get("risk_score"),
                   "decision": j.get("decision"), "degraded": j.get("degraded"),
                   "silent_allow_warning": bool(j.get("decision") == "allow" and not j.get("degraded"))}
    else:
        nan_res = {"status": r_nan.status_code, "body": r_nan.text[:200]}
    r_x = client.post(f"{RISK}/internal/evaluate", json={
        "event_id": gen_event(rng), "fraud_id": fid,
        "features": {**feats0, "amount_zscore": 1e300, "txn_freq_last_24h": 10 ** 9}})
    fail["nan_extreme"] = {
        "nan_response": nan_res,
        "extreme_response": {"status": r_x.status_code,
                             "body": r_x.json() if r_x.status_code == 200 else r_x.text[:200]},
    }
    log(f"  8b: NaN -> {nan_res}")

    log("  8c: corrupt feature vector (missing required field)...")
    feats_bad = dict(feats0)
    feats_bad.pop("amount_ratio", None)
    r_bad = client.post(f"{RISK}/internal/evaluate", json={
        "event_id": gen_event(rng), "fraud_id": fid, "features": feats_bad})
    fail["missing_feature"] = {"status": r_bad.status_code}
    log(f"  8c: status={r_bad.status_code}")

    log("  8d: risk service down (kill, probe, restart ~60s)...")
    subprocess.run(["taskkill", "/PID", str(risk_pid), "/F"], capture_output=True, text=True)
    time.sleep(2)
    try:
        client.post(f"{RISK}/internal/evaluate", json={
            "event_id": gen_event(rng), "fraud_id": fid, "features": feats0})
        fail["risk_down"] = {"detected": False}
    except Exception as e:
        fail["risk_down"] = {"detected": True, "error": str(e)[:150],
                             "note": "caller gets an explicit connection error (no silent "
                                     "approve/reject); fail-safe must live at the caller"}
    r_ing = ingest_one(rng, fid)
    fail["ingest_while_risk_down"] = {
        "status": r_ing.status_code,
        "note": "ingest does not chain to risk; it succeeds independently - a caller that "
                "skips /internal/evaluate after a risk outage would silently drop scoring"}
    e = dict(os.environ)
    e["DB_DIR"] = str(DB_DIR)
    e["PS14_MODE"] = "development"
    logf = open(DB_DIR / "risk.log", "a", encoding="utf-8", buffering=1)
    p_risk = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.risk_engine.main:app", "--port", "8003"],
                              cwd=str(ROOT), env=e, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    ok = False
    while time.time() - t0 < 150:
        try:
            with urllib.request.urlopen(f"{RISK}/health", timeout=2) as rr:
                ok = rr.status == 200
                break
        except Exception:
            time.sleep(3)
    fail["risk_restarted_healthy"] = ok
    log(f"  8d: detected={fail['risk_down']['detected']} restart_healthy={ok}")

    # stage 9: resources ----------------------------------------------------
    log("stage 9: resources...")
    sizes = {f.name: f.stat().st_size for f in DB_DIR.glob("*.db*") if f.suffix == ".db"}
    REPORT["db_sizes_bytes"] = sizes
    counts = {}
    for dbname, sql in [("risk.db", "risk_scores"),
                        ("features.db", "transaction_features"),
                        ("audit.db", "audit_events")]:
        try:
            c = sqlite3.connect(DB_DIR / dbname)
            counts[f"{dbname}.{sql}"] = c.execute(f"SELECT COUNT(*) FROM {sql}").fetchone()[0]
            c.close()
        except Exception as ex:
            counts[dbname] = str(ex)[:80]
    REPORT["table_growth_counts"] = counts
    REPORT["rss_kb_end"] = rss_kb(p_risk.pid)
    log(f"  db sizes: {sizes}")
    log(f"  counts: {counts}")

    REPORT["runtime_sec"] = round(time.time() - T0, 1)
    with open(OUT, "w") as f:
        json.dump(REPORT, f, indent=1, default=str)
    log(f"DONE -> reports/perf_reliability_audit.json ({REPORT['runtime_sec']}s)")


if __name__ == "__main__":
    main()