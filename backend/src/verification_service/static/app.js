// Hostname-relative so the UI works on LAN/VPS, not just localhost: the
// browser's hostname is used for the Identity Service, and this service
// itself is `window.location.origin` (proxy-safe).
const IDENTITY_URL = `${window.location.origin}/identity`;
const VERIFY_URL   = window.location.origin;     // this Verification Service

const $ = (id) => document.getElementById(id);
let token = localStorage.getItem("ps14_token") || null;
let overview = null;  // last /overview payload: alerts + cases + feedback + chain in one fetch

// API client generated from each service's OpenAPI schema at load time
// (see /static/api.js): every endpoint becomes a callable helper, so a new
// backend endpoint is automatically available here. Auth mode per operation
// (default bearer); tokens are read per request - login/logout need no
// rebuild. `apiReady` resolves once both clients exist.
const AUTH_MODE = {
  register: "none", login: "none", chainStatus: "none", index: "none",
  health: "none", openapi: "none",
  complianceEvents: "compliance", complianceIntegrity: "compliance",
  complianceOverview: "compliance", complianceDeviceGraph: "compliance",
};
let api = null;
const apiReady = Promise.all([
  fetch(IDENTITY_URL + "/openapi.json").then((r) => r.json()),
  fetch(VERIFY_URL + "/openapi.json").then((r) => r.json()),
]).then(([idSpec, vcSpec]) => {
  api = {
    identity: createClient(idSpec, {
      base: IDENTITY_URL,
      authMode: (id) => AUTH_MODE[id] || "bearer",
      getToken: () => token,
    }),
    verify: createClient(vcSpec, {
      base: VERIFY_URL,
      authMode: (id) => AUTH_MODE[id] || "bearer",
      getToken: () => token,
      getComplianceToken: () => $("complianceToken").value.trim(),
    }),
  };
  return api;
}).catch((e) => { console.error("API client build failed:", e); });

async function login() {
  const status = $("loginStatus");
  status.className = "status"; status.textContent = "";
  try {
    const email = $("email").value.trim(), password = $("password").value;
    await apiReady;
    const res = await api.identity.login({ body: { email, password } });
    token = res.access_token;
    localStorage.setItem("ps14_token", token);
    authRecovering = false;
    $("subline").textContent = "Signed in · alerts below";
    showAlerts();
  } catch (e) {
    status.className = "status error";
    status.textContent = "Login failed: " + e.message;
  }
}

// One-click demo: register a throwaway account, sign in, and seed warm-up
// history + an attack event, so the whole flow is visible without knowing
// any credentials.
async function demoLogin() {
  const status = $("loginStatus");
  const btn = $("demoLoginBtn");
  status.className = "status"; status.textContent = "Creating a demo account…";
  btn.disabled = true;
  const stamp = Date.now().toString(36);
  const email = "demo-" + stamp + "@example.com";
  const password = "demo-pass-" + stamp + "!";
  try {
    await apiReady;
    await api.identity.register(
      { body: { full_name: "Demo User", email, phone: "+911234567890", password } });
    const res = await api.identity.login({ body: { email, password } });
    token = res.access_token;
    localStorage.setItem("ps14_token", token);
    authRecovering = false;
    $("subline").textContent = "Demo account ready · seeding activity…";
    showAlerts();
    demo(); // warm-up + attack, then alerts load automatically
  } catch (e) {
    status.className = "status error";
    status.textContent = "Demo failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function fmt(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderAlert(a) {
  const card = document.createElement("div");
  card.className = "card high";

  const head = document.createElement("div");
  head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.alignItems = "center";
  head.innerHTML = `<strong>Unusual activity detected</strong><span class="tag high">Risk: High (${a.risk_score})</span>`;
  card.appendChild(head);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = fmt(a.scored_at) + " · event " + a.event_id;
  card.appendChild(meta);

  const why = document.createElement("div");
  why.innerHTML = "<div class='sub' style='margin-bottom:4px'>Why this was flagged:</div>";
  const ul = document.createElement("ul"); ul.className = "reasons";
  (a.reason_texts || []).forEach((t) => { const li = document.createElement("li"); li.textContent = t; ul.appendChild(li); });
  why.appendChild(ul);
  card.appendChild(why);

  const row = document.createElement("div");
  row.className = "row";
  const yes = document.createElement("button"); yes.className = "btn ok";  yes.textContent = "This was me";
  const no  = document.createElement("button"); no.className = "btn no";   no.textContent = "This wasn't me";
  row.append(yes, no);
  card.appendChild(row);

  const status = document.createElement("div");
  status.className = "status";
  card.appendChild(status);

  const act = (outcome) => {
    yes.disabled = no.disabled = true;
    status.textContent = "Submitting…";
    apiReady.then(() => api.verify.confirmAlert({ event_id: a.event_id, body: { outcome } }))
      .then((res) => {
        status.textContent = res.message;
        row.classList.add("hide");
        // A decision lands on the chain: one refresh keeps alerts, the
        // chain chip, and open history all current.
        loadOverview();
        if (res.recovery_steps && res.recovery_steps.length) {
          const rec = document.createElement("div");
          rec.className = "recovery";
          rec.innerHTML = "<h3>Guided recovery flow</h3><ol>" +
            res.recovery_steps.map((s) => "<li>" + s + "</li>").join("") + "</ol>" +
            "<div class='case'>Case: " + res.case_id + "</div>";
          card.appendChild(rec);
        }
      })
      .catch((e) => { status.className = "status error"; status.textContent = "Failed: " + e.message; yes.disabled = no.disabled = false; });
  };
  yes.onclick = () => act("this_was_me");
  no.onclick  = () => act("this_wasnt_me");

  return card;
}

function showAlerts() {
  $("loginView").classList.add("hide");
  $("alertsView").classList.remove("hide");
  $("logoutBtn").classList.remove("hide");
  loadOverview();
}

// A dead JWT (expired 15-min token, account re-registered after a stack
// restart) makes EVERY authed call fail with 401 and leaves the app stuck
// showing "Failed to load…" errors - the demo button included. On the first
// 401 while logged in, clear the session and return to the login view with
// a clear message instead of leaving the user stranded.
let authRecovering = false;
function handleAuthError(e) {
  if (authRecovering || !e || e.status !== 401 || !token) return false;
  authRecovering = true;
  token = null;
  localStorage.removeItem("ps14_token");
  stopComplianceTimer();
  $("alertsView").classList.add("hide");
  $("complianceView").classList.add("hide");
  $("historyView").classList.add("hide");
  $("logoutBtn").classList.add("hide");
  $("loginView").classList.remove("hide");
  $("loginStatus").className = "status error";
  $("loginStatus").textContent = "Session expired — please sign in again.";
  return true;
}

// One fetch drives the whole alerts screen (BFF pattern): /overview bundles
// /alerts + /cases + /feedback-status + chain status into a single round
// trip, so the UI renders from one payload instead of chained fetches.
//
// `silent` is for the gentle auto-refresh tick: no spinner flash, and
// errors keep the last good state. A generation token makes the newest load
// win, so a slow tick response can never clobber a fresher manual load.
let overviewGen = 0;
function loadOverview(silent) {
  const gen = ++overviewGen;
  const list = $("alertList");
  if (!silent) list.innerHTML = "<div class='spinner'>Loading alerts…</div>";
  apiReady.then(() => api.verify.overview())
    .then((res) => {
      if (gen !== overviewGen) return;  // a newer load superseded this one
      overview = res;
      renderChainChip(res.chain);
      list.innerHTML = "";
      if (!res.alerts.length) {
        list.innerHTML = "<div class='empty'>No open alerts. Everything looks normal ✨<br><span style='font-size:12px'>Try “Simulate demo activity” to see the flow.</span></div>";
      } else {
        res.alerts.forEach((a) => list.appendChild(renderAlert(a)));
      }
      // If history is open, keep it current from the same payload.
      if (!$("caseHistory").classList.contains("hide")) renderCases();
    })
    .catch((e) => {
      if (gen !== overviewGen) return;
      if (silent) return;  // background tick: keep the last good render
      if (handleAuthError(e)) return;
      list.innerHTML = "";
      $("alertStatus").className = "status error";
      $("alertStatus").textContent = "Failed to load alerts: " + e.message;
    });
}

// The alerts-screen chain chip, from the overview payload (the footer badge
// keeps its own public /chain-status poll for the logged-out state).
function renderChainChip(chain) {
  const chip = $("alertsChainChip");
  if (!chip) return;
  const text = chip.querySelector(".text") || chip;
  if (!chain || chain.available === false) {
    chip.className = "chainline";
    text.textContent = "Audit chain status unavailable";
  } else if (chain.ok) {
    chip.className = "chainline ok";
    text.textContent = "✓ Audit chain verified — " + chain.n_entries + " entries";
  } else {
    chip.className = "chainline bad";
    text.textContent = "✗ CHAIN BROKEN — first bad entry at seq " + chain.first_bad_seq;
  }
}

// Post-verification: render the resolved-cases history (what happened after
// the "This was me / wasn't me" decision) plus the feedback-pool status that
// feeds the retraining gate - straight from the last /overview payload, no
// extra fetches (falls back to one fetch if no overview has loaded yet).
function renderCases() {
  const el = $("caseHistory");
  const render = (res) => {
    el.innerHTML = "";
    // Header row: the section label plus the entry point into the account
    // history view (abnormalities across this account's cases).
    const headRow = document.createElement("div");
    headRow.style.display = "flex"; headRow.style.justifyContent = "space-between";
    headRow.style.alignItems = "center"; headRow.style.margin = "12px 0 6px";
    const head = document.createElement("div");
    head.className = "sub"; head.style.margin = "0";
    head.textContent = "Resolved cases";
    const histBtn = document.createElement("button");
    histBtn.type = "button"; histBtn.className = "btn ghost";
    histBtn.style.padding = "4px 10px"; histBtn.style.fontSize = "12px";
    histBtn.textContent = "Account history →";
    histBtn.onclick = showHistory;
    headRow.append(head, histBtn);
    el.appendChild(headRow);
    if (!res.cases.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "No resolved cases yet — confirm or dispute an alert to build history.";
      el.appendChild(empty);
      return;
    }

      // Summary chips, same compact treatment as the compliance trail.
      const confirmed = res.cases.filter((c) => c.outcome === "confirmed").length;
      const chips = document.createElement("div");
      chips.className = "rchip-wrap"; chips.style.marginBottom = "8px";
      chips.innerHTML =
        `<span class="rchip">${res.cases.length} case${res.cases.length === 1 ? "" : "s"}</span>` +
        `<span class="rchip">${confirmed} confirmed</span>` +
        `<span class="rchip">${res.cases.length - confirmed} disputed</span>`;
      el.appendChild(chips);

      res.cases.forEach((c) => {
        const ok = c.outcome === "confirmed";
        const reasons = c.reason_texts || [];
        const card = document.createElement("div");
        card.className = "card"; card.style.padding = "10px 14px"; card.style.cursor = "pointer";
        card.innerHTML =
          `<div style="display:flex;justify-content:space-between;align-items:center;gap:10px">` +
          `<strong style="font-size:12.5px">${esc(c.case_id)}</strong>` +
          `<span style="display:flex;align-items:center;gap:8px">` +
          `<span class="tag ${ok ? "low" : "high"}">${c.outcome}</span>` +
          `<span class="tmore caret" style="font-size:13px">▸</span></span></div>` +
          `<div class="meta" style="margin:6px 0 0">${fmt(c.resolved_at)} · event ${esc(c.event_id)} · ` +
          `risk ${c.risk_band} (${c.risk_score})</div>` +
          `<div class="hide" style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border)">` +
          `<div class="sub" style="margin:0 0 6px">Why this was flagged:</div>` +
          `<div class="rchip-wrap">` +
          (reasons.length
            ? reasons.map((r) => `<span class="rchip">${esc(r)}</span>`).join("")
            : `<span class="rchip none">no flags — nothing unusual</span>`) +
          `</div></div>`;
        const detail = card.querySelector(".hide");
        const caret = card.querySelector(".caret");
        card.onclick = () => {
          const open = !detail.classList.contains("hide");
          detail.classList.toggle("hide", open);
          caret.textContent = open ? "▸" : "▾";
        };
        el.appendChild(card);
      });
      const f = res.feedback;
      if (f) {
        const note = document.createElement("div");
        note.className = "meta"; note.style.marginTop = "10px";
        note.textContent = `Feedback pool: ${f.confirmed} confirmed · ${f.disputed} disputed · ` +
          `retrain trigger at ${f.retrain_disputed_threshold} disputed`;
        el.appendChild(note);
      }
    };
  if (overview) { render(overview); return; }
  apiReady.then(() => api.verify.overview())
    .then((res) => { overview = res; render(res); })
    .catch((e) => { el.innerHTML = ""; const s = document.createElement("div"); s.className = "status error"; s.textContent = "Failed to load cases: " + e.message; el.appendChild(s); });
}

// ---- Account history: abnormalities across the account's cases -----------
// All cases share the same fraud_id, so this page is the trail a reviewer
// uses to spot patterns (repeated reasons, dispute counts, escalation).
// The abnormality analysis itself is computed by the backend (/history).
function showHistory() {
  $("alertsView").classList.add("hide");
  $("historyView").classList.remove("hide");
  loadHistory();
}

function backFromHistory() {
  $("historyView").classList.add("hide");
  $("alertsView").classList.remove("hide");
}

async function loadHistory() {
  const status = $("historyStatus");
  status.className = "status"; status.textContent = "";
  $("historyList").innerHTML = "<div class='spinner'>Loading account history…</div>";
  try {
    await apiReady;
    const h = await api.verify.history();
    renderHistory(h.events || [], h.stats || {});
  } catch (e) {
    if (handleAuthError(e)) return;
    $("historyList").innerHTML = "";
    status.className = "status error";
    status.textContent = "Failed to load history: " + e.message;
  }
}

function renderHistory(events, stats) {
  const list = $("historyList");
  list.innerHTML = "";
  const flagsEl = $("historyFlags");
  if (stats.flags && stats.flags.length) {
    flagsEl.className = "banner bad";
    flagsEl.innerHTML = "<strong>Abnormalities detected:</strong><ul style='margin:6px 0 0 18px'>" +
      stats.flags.map((f) => "<li>" + esc(f) + "</li>").join("") + "</ul>";
  } else {
    flagsEl.className = "banner ok";
    flagsEl.textContent = "No abnormalities detected ✓";
  }

  const b = stats.bands || {};
  $("historyStatsCard").classList.remove("hide");
  $("historyStatsChips").innerHTML =
    `<span class="rchip">${stats.total || 0} scored events</span>` +
    (b.high ? `<span class="rchip">${b.high} high</span>` : "") +
    (b.medium ? `<span class="rchip">${b.medium} medium</span>` : "") +
    (b.low ? `<span class="rchip">${b.low} low</span>` : "") +
    (stats.disputed ? `<span class="rchip">${stats.disputed} disputed</span>` : "") +
    (stats.confirmed ? `<span class="rchip">${stats.confirmed} confirmed</span>` : "");
  const rep = stats.repeated_reasons || [];
  $("historyRepeated").textContent = rep.length
    ? "Repeated reasons: " + rep.map((r) => `${r.text} ×${r.count}`).join(" · ")
    : "No reason code repeats.";

  if (!events.length) {
    list.innerHTML = "<div class='empty'>No scored events yet — activity will appear here as it is evaluated.</div>";
    return;
  }
  events.forEach((e) => {
    const card = document.createElement("div");
    card.className = "card tcard" + (e.risk_band === "low" ? " slim" : "");
    const band = String(e.risk_band || "").toLowerCase();
    const tagCls = band === "high" ? "tag high" : band === "medium" ? "tag medium" : "tag low";
    const head = document.createElement("div");
    head.style.display = "flex"; head.style.justifyContent = "space-between";
    head.style.alignItems = "center"; head.style.gap = "10px";
    head.innerHTML = `<span class="${tagCls}">${e.risk_band} · ${e.risk_score}</span>` +
      (e.outcome ? `<span class="tag ${e.outcome === "confirmed" ? "low" : "high"}">${e.outcome}</span>` : "") +
      `<span class="meta" style="margin:0;white-space:nowrap">${fmt(e.scored_at)}</span>`;
    card.appendChild(head);
    const meta = document.createElement("div");
    meta.className = "meta"; meta.style.margin = "6px 0 2px";
    meta.textContent = "event " + e.event_id;
    card.appendChild(meta);
    const wrap = document.createElement("div");
    wrap.className = "rchip-wrap";
    wrap.innerHTML = (e.reason_texts && e.reason_texts.length)
      ? e.reason_texts.map((r) => `<span class="rchip">${esc(r)}</span>`).join("")
      : `<span class="rchip none">no flags — nothing unusual</span>`;
    card.appendChild(wrap);
    list.appendChild(card);
  });
}

function demo() {
  const sp = $("spinner"); sp.classList.remove("hide");
  apiReady.then(() => api.verify.demoSeed())
    .then((res) => {
      sp.classList.add("hide");
      // 5/6 commits is healthy: the birth event (-hist-0000) steps up on its
      // own new device (NEW_DEVICE) and never enters the baseline by design.
      const n = res.high_risk;
      $("alertStatus").className = "status";
      $("alertStatus").textContent =
        `Seeded ${res.seeded} events · baseline committed ${res.baseline_committed}/6 · ${n} high-risk alert${n === 1 ? "" : "s"}.`;
      lastDemoDevice = res.device_id || null;
      const hint = $("graphHint");
      if (hint && lastDemoDevice) hint.innerHTML = `Demo account seeded — <a href="#" id="graphUseDemo">use its device token</a> in the graph lookup.`;
      if (hint && lastDemoDevice) $("graphUseDemo").onclick = (ev) => {
        ev.preventDefault();
        $("graphDevice").value = lastDemoDevice; $("graphRecipient").value = "";
        loadDeviceGraph();
      };
      loadOverview();
    })
    .catch((e) => {
      sp.classList.add("hide");
      if (handleAuthError(e)) return;
      $("alertStatus").className = "status error";
      $("alertStatus").textContent = "Seed failed: " + e.message;
    });
}

// ---- Compliance view ----------------------------------------------------
function showCompliance() {
  $("loginView").classList.add("hide");
  $("alertsView").classList.add("hide");
  $("complianceView").classList.remove("hide");
  $("logoutBtn").classList.remove("hide");
}

function backFromCompliance() {
  stopComplianceTimer();
  $("complianceView").classList.add("hide");
  if (token) showAlerts();
  else { $("loginView").classList.remove("hide"); $("logoutBtn").classList.add("hide"); }
}

// Compact, scannable trail cards: score events show band + score + model on
// one line with reason chips (collapsed past 3), ingests become slim rows,
// and every card keeps its per-entry hash linkage.
function renderTrailEvent(e) {
  const p = e.payload || {};
  const t = e.event_type;
  const card = document.createElement("div");
  card.className = "card tcard" + (t === "feature_ingested" ? " slim" : "");

  const head = document.createElement("div");
  head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.alignItems = "center"; head.style.gap = "10px";
  head.innerHTML = `<strong style="font-size:12.5px">seq ${e.seq} · ${t}</strong>
    <span class="meta" style="margin:0;white-space:nowrap">${fmt(e.created_at)}</span>`;
  card.appendChild(head);

  if (t === "score_generated") {
    const band = String(p.risk_band || "").toLowerCase();
    const tagCls = band === "high" ? "tag high" : band === "medium" ? "tag medium" : "tag low";
    const line = document.createElement("div");
    line.style.margin = "8px 0 2px";
    line.innerHTML = `<span class="${tagCls}">${band} · ${p.risk_score}</span>` +
      (p.model_version ? `<span class="meta" style="margin-left:8px">model ${p.model_version}</span>` : "");
    card.appendChild(line);
    const reasons = p.reason_texts || p.reason_codes || [];
    const wrap = document.createElement("div");
    wrap.className = "rchip-wrap";
    if (reasons.length) {
      const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      wrap.innerHTML = reasons.map((r) => `<span class="rchip" title="${esc(r)}">${esc(r)}</span>`).join("");
      const chips = [...wrap.querySelectorAll(".rchip")];
      if (chips.length > 3) {
        const extra = chips.slice(3);
        extra.forEach((c) => (c.style.display = "none"));
        const more = document.createElement("button");
        more.type = "button"; more.className = "tmore";
        more.textContent = `…and ${extra.length} more`;
        more.onclick = () => {
          const open = more.textContent.startsWith("…and");
          extra.forEach((c) => (c.style.display = open ? "inline-block" : "none"));
          more.textContent = open ? "show fewer" : `…and ${extra.length} more`;
        };
        wrap.appendChild(more);
      }
    } else {
      // A score with no flagged reasons is not a broken render: say so.
      wrap.innerHTML = `<span class="rchip none">no flags — nothing unusual</span>`;
    }
    card.appendChild(wrap);
  } else if (t === "verification_resolved") {
    const ok = p.outcome === "confirmed";
    const d = document.createElement("div");
    d.style.margin = "8px 0 2px";
    d.innerHTML = `<span class="tag ${ok ? "low" : "high"}">${p.outcome}</span>` +
      (p.case_id ? `<span class="case" style="margin-left:10px;font-size:12.5px">${p.case_id}</span>` : "");
    card.appendChild(d);
  } else if (t === "fraud_id_resolved") {
    // Break-glass: the fraud-id bridge was opened for a compliance reason.
    const d = document.createElement("div");
    d.style.margin = "6px 0 2px";
    d.innerHTML = `<span class="meta">break-glass · actor ${p.actor || "—"} · ${p.reason || ""}</span>`;
    card.appendChild(d);
  } else if (t === "feature_ingested") {
    const d = document.createElement("div");
    d.style.margin = "4px 0 0";
    d.innerHTML = `<span class="meta">event ${p.event_id || "—"}</span>`;
    card.appendChild(d);
  }

  const hash = document.createElement("div");
  hash.className = "hash";
  hash.style.marginTop = "6px";
  hash.title = "entry_hash=" + e.entry_hash + " prev_hash=" + e.prev_hash;
  hash.textContent = "entry " + String(e.entry_hash || "").slice(0, 14) + "… · prev " + String(e.prev_hash || "").slice(0, 14) + "…";
  card.appendChild(hash);
  return card;
}

function renderTrail(events) {
  $("trailList").innerHTML = "";
  if (!events.length) { $("trailList").innerHTML = "<div class='empty'>No audit events yet.</div>"; return; }
  events.forEach((e) => $("trailList").appendChild(renderTrailEvent(e)));
}

function appendTrail(events) {
  events.forEach((e) => $("trailList").appendChild(renderTrailEvent(e)));
}

function updateIntegrityBanner(integrity) {
  const banner = $("integrityBanner");
  if (integrity.ok) {
    banner.className = "banner ok";
    banner.textContent = "✓ Chain verified — " + integrity.n_entries + " entries, genesis " + String(integrity.genesis_hash || "").slice(0, 12) + "…";
  } else {
    banner.className = "banner bad";
    banner.textContent = "✗ CHAIN BROKEN — first bad entry at seq " + integrity.first_bad_seq;
  }
}

// Fuller chain status: entries + genesis + the NEWEST chain entry (seq,
// type, time, hash) so the register-to-audit trail is visible end-to-end.
// `events` is the trail as returned (newest first). Cheap to update on
// every auto-refresh tick without re-rendering the whole list.
function updateChainStatus(integrity, events) {
  const card = $("chainStatusCard");
  if (!integrity) { card.style.display = "none"; return; }
  card.style.display = "";
  const line = $("chainStatusLine");
  if (!integrity.ok) {
    line.textContent = "BROKEN — first bad entry at seq " + integrity.first_bad_seq
      + " (verification of the chain stops there)";
    return;
  }
  const head = events && events.length ? events[0] : null;
  const genesis = String(integrity.genesis_hash || "").slice(0, 16);
  if (!head) {
    line.textContent = integrity.n_entries + " entries · genesis " + genesis + "… · no entries yet";
    return;
  }
  line.textContent = integrity.n_entries + " entries · genesis " + genesis + "… · latest entry seq "
    + head.seq + " · " + head.event_type + " · " + fmt(head.created_at)
    + " · entry " + String(head.entry_hash || "").slice(0, 16) + "…";
}

// Latest verification case: derived by the backend (newest verification_resolved
// joined with its score_generated) so the card tells the whole story - alert ->
// score -> outcome -> case -> chain entry - and stays correct even when the
// newest case falls outside the paged window of events.
function renderLatestCase(latestCase) {
  const card = $("latestCaseCard");
  const body = $("latestCaseBody");
  if (!latestCase) { card.classList.add("hide"); return; }
  const band = String(latestCase.risk_band || "").toLowerCase();
  const tagCls = band === "high" ? "tag high" : band === "medium" ? "tag medium" : "tag low";
  const ok = latestCase.outcome === "confirmed";
  const scoreBit = latestCase.risk_band
    ? ` · <span class="${tagCls}">${latestCase.risk_band} · ${latestCase.risk_score}</span>`
    : "";
  body.innerHTML =
    `<div style="display:flex;justify-content:space-between;align-items:center">` +
      `<span class="case" style="font-size:15px;margin:0">${esc(latestCase.case_id || "—")}</span>` +
      `<span class="tag ${ok ? "low" : "high"}">${latestCase.outcome || "—"}</span>` +
    `</div>` +
    `<div class="meta" style="margin-top:8px">event ${esc(latestCase.event_id || "—")}${scoreBit} · ${fmt(latestCase.resolved_at)}</div>` +
    `<div class="hash" style="margin-top:8px">chain seq ${latestCase.seq} · entry ${String(latestCase.entry_hash || "").slice(0, 16)}…</div>`;
  card.classList.remove("hide");
}

let complianceTimer = null;

// Paging for the trail: first page on load, then "Show older events"
// fetches the next window via ?offset= (the audit endpoints now support it).
const TRAIL_PAGE = 40;
let trailOffset = 0;   // events already rendered
let trailTotal = 0;    // full chain length (from the endpoint's total)
let topSeqAtLoad = 0;  // newest seq the list was rendered with
let lastDemoDevice = null;  // device token from the last /demo/seed, for the graph quick-fill

function startComplianceTimer() {
  if (complianceTimer) clearInterval(complianceTimer);
  complianceTimer = setInterval(refreshCompliance, 20000);
}

function stopComplianceTimer() {
  if (complianceTimer) { clearInterval(complianceTimer); complianceTimer = null; }
}

function updateTrailMore() {
  $("trailMoreBtn").classList.toggle("hide", trailOffset >= trailTotal);
}

async function loadCompliance() {
  const pass = $("complianceToken").value.trim();
  const status = $("complianceStatus");
  status.className = "status"; status.textContent = "";
  if (!pass) { status.className = "status error"; status.textContent = "Enter the compliance passphrase."; return; }
  $("integrityBanner").className = "banner hide";
  $("trailList").innerHTML = "<div class='spinner'>Loading trail…</div>";
  $("trailStaleHint").classList.add("hide");
  status.textContent = "Verifying chain…";
  try {
    await apiReady;
    // One round trip: integrity + latest case + paged events + aggregates.
    const ov = await api.verify.complianceOverview({ limit: TRAIL_PAGE, offset: 0 });
    const integrity = ov.integrity;
    updateIntegrityBanner(integrity);
    const events = ov.events || [];
    trailTotal = ov.total || events.length;
    trailOffset = events.length;
    topSeqAtLoad = events.length ? events[0].seq : 0;
    updateChainStatus(integrity, events);
    renderLatestCase(ov.latest_case);
    renderTrail(events);
    updateTrailMore();
    status.textContent = "";
    // Auto-refresh: keep the banner + trail current as new events land.
    startComplianceTimer();
  } catch (err) {
    status.className = "status error";
    status.textContent = "Failed: " + err.message;
  }
}

// ---- Device graph (compliance role) -------------------------------------
async function loadDeviceGraph() {
  const pass = $("complianceToken").value.trim();
  const status = $("graphStatus");
  status.className = "status"; status.textContent = "";
  if (!pass) { status.className = "status error"; status.textContent = "Enter the compliance passphrase."; return; }
  const device = $("graphDevice").value.trim();
  const recipient = $("graphRecipient").value.trim();
  if (!device && !recipient) { status.className = "status error"; status.textContent = "Enter a device or recipient token."; return; }
  status.textContent = "Resolving graph…";
  try {
    await apiReady;
    // Query params follow the OpenAPI names (snake_case): the generated
    // client passes them through verbatim.
    const res = await api.verify.complianceDeviceGraph({
      device_id: device || undefined,
      recipient_id: recipient || undefined,
    });
    renderDeviceGraph(res);
    status.textContent = "";
  } catch (err) {
    status.className = "status error";
    status.textContent = "Graph lookup failed: " + err.message;
  }
}

function renderDeviceGraph(res) {
  const box = $("graphResult");
  const label = res.device_id ? `device <code>${esc(res.device_id)}</code>` : `recipient <code>${esc(res.recipient_id)}</code>`;
  if (!res.accounts || !res.accounts.length) {
    box.innerHTML = `<div class="meta" style="margin:0">No other accounts share this ${label}.</div>`;
    return;
  }
  const rows = res.accounts.map((a) =>
    `<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-top:1px solid rgba(0,0,0,.06)">` +
      `<span style="font-size:12.5px">${esc(a.fraud_id)} · ${a.events} event${a.events === 1 ? "" : "s"} · last ${a.days_since_last == null ? "—" : a.days_since_last + "d ago"}</span>` +
    `</div>`
  ).join("");
  box.innerHTML =
    `<div class="sub" style="margin:0 0 4px">${res.total_accounts} account${res.total_accounts === 1 ? "" : "s"} on ${label}</div>` +
    rows +
    `<div class="meta" style="margin-top:6px">Shared ${label} is the link-analysis (mule-ring) signature — a ring shows several accounts converging on one device or sink recipient.</div>`;
}

async function refreshCompliance() {
  // Silent poll: update banner/cards every tick; re-render the trail only
  // when a NEW event lands at the top (so scrolling isn't reset each poll).
  // If the user paged deeper, surface a hint instead of clobbering the list.
  const pass = $("complianceToken").value.trim();
  if (!pass) return;
  try {
    await apiReady;
    const ov = await api.verify.complianceOverview({ limit: TRAIL_PAGE, offset: 0 });
    const integrity = ov.integrity;
    updateIntegrityBanner(integrity);
    const events = ov.events || [];
    trailTotal = ov.total || events.length;
    const top = events.length ? events[0].seq : 0;
    updateChainStatus(integrity, events);
    renderLatestCase(ov.latest_case);
    if (top !== topSeqAtLoad) {
      topSeqAtLoad = top;
      if (trailOffset === TRAIL_PAGE) { trailOffset = events.length; renderTrail(events); }
      else { $("trailStaleHint").classList.remove("hide"); }
    }
    updateTrailMore();
  } catch (err) {
    // silent - keep the last good state; the footer badge still reports
  }
}

// Dev default from src/settings.py (overridable via COMPLIANCE_TOKEN).
// Pre-filled so the compliance view opens with one click in the prototype;
// production replaces this with real RBAC + mTLS (section 3).
$("complianceToken").value = "ps14-dev-compliance-token-change-me";
$("complianceBtn").onclick = () => { showCompliance(); loadCompliance(); };
$("complianceBackBtn").onclick = backFromCompliance;
$("complianceLoadBtn").onclick = loadCompliance;
$("trailMoreBtn").onclick = async () => {
  const pass = $("complianceToken").value.trim();
  if (!pass) return;
  const more = $("trailMoreBtn");
  more.disabled = true; more.textContent = "Loading older…";
  try {
    await apiReady;
    const page = await api.verify.complianceEvents({ limit: TRAIL_PAGE, offset: trailOffset });
    const events = page.events || [];
    trailTotal = page.total || trailOffset + events.length;
    trailOffset += events.length;
    appendTrail(events);
    updateTrailMore();
  } catch (err) {
    $("complianceStatus").className = "status error";
    $("complianceStatus").textContent = "Failed to load older events: " + err.message;
  } finally {
    more.disabled = false; more.textContent = "Show older events";
  }
};
$("complianceToken").addEventListener("keydown", (e) => { if (e.key === "Enter") loadCompliance(); });
$("graphBtn").onclick = loadDeviceGraph;
$("graphDevice").addEventListener("keydown", (e) => { if (e.key === "Enter") loadDeviceGraph(); });
$("graphRecipient").addEventListener("keydown", (e) => { if (e.key === "Enter") loadDeviceGraph(); });

$("historyBackBtn").onclick = backFromHistory;
$("historyRefreshBtn").onclick = loadHistory;

$("loginBtn").onclick = login;
$("demoLoginBtn").onclick = demoLogin;
$("password").addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
$("demoBtn").onclick = demo;
$("refreshBtn").onclick = loadOverview;
$("historyBtn").onclick = () => {
  const el = $("caseHistory");
  if (el.classList.contains("hide")) { el.classList.remove("hide"); renderCases(); }
  else el.classList.add("hide");
};
$("logoutBtn").onclick = () => { stopComplianceTimer(); token = null; localStorage.removeItem("ps14_token"); location.reload(); };

// ---- Live chain-integrity badge (footer + alerts chip) ------------------
// Polls /chain-status (proxied to the Audit Service with the internal
// token; every poll is itself logged to the audit access log) and renders
// the same state into the footer badge and the alerts-screen chip.
function setChainState(ok, n, genesis, firstBad, unavailable) {
  const cls = (id) => (id === "chainBadge" ? "footer" : "chainline");
  [$("chainBadge"), $("alertsChainChip")].filter(Boolean).forEach((el) => {
    const text = el.querySelector(".text") || el;
    if (unavailable) {
      el.className = cls(el.id);
      text.textContent = "Audit chain status unavailable";
    } else if (ok) {
      el.className = cls(el.id) + " ok";
      text.textContent = "✓ Audit chain verified — " + n + " entries · genesis " + String(genesis || "").slice(0, 12) + "…";
    } else {
      el.className = cls(el.id) + " bad";
      text.textContent = "✗ CHAIN BROKEN — first bad entry at seq " + firstBad;
    }
  });
}

function refreshChainBadge() {
  apiReady.then(() => api.verify.chainStatus())
    .then((d) => setChainState(d.ok, d.n_entries, d.genesis_hash, d.first_bad_seq, false))
    .catch(() => setChainState(false, 0, "", null, true));
}
refreshChainBadge();
setInterval(refreshChainBadge, 60000);

// Gentle auto-refresh: while the alerts screen is open, re-fetch /overview
// every 30s so new alerts appear without clicking Refresh. Skips when
// logged out or on another view, and never flashes the spinner (silent).
// The server-side /overview cache has a 10s TTL, so each tick still gets
// fresh data.
const OVERVIEW_REFRESH_MS = 30000;
setInterval(() => {
  if (!token) return;
  if ($("alertsView").classList.contains("hide")) return;  // login/compliance views
  loadOverview(true);
}, OVERVIEW_REFRESH_MS);

if (token) showAlerts();
