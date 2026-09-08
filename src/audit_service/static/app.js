const BASE = window.location.origin;
const $ = (id) => document.getElementById(id);

// Dev default from src/settings.py (overridable via COMPLIANCE_TOKEN).
// Pre-filled so the viewer opens with one click in the prototype; production
// replaces this with real RBAC + mTLS (section 3).
$("pass").value = "ps14-dev-compliance-token-change-me";

// Paging: first window on load, then "Show older events" fetches the next
// window via ?offset= (the audit endpoints support it and report total).
const TRAIL_PAGE = 40;
let trailOffset = 0;  // events already rendered
let trailTotal = 0;   // full chain length (from the endpoint's total)
let genesisHash = null; // out-of-band genesis (for the tail row's link check)
let lastRowEl = null; // DOM element of the last (oldest) rendered row
let lastRowE = null;  // its event - the boundary whose link the next page resolves

// Event-type filter chips: the audit endpoint filters on a single event_type.
const EVENT_FILTERS = [
  { label: "All", value: "" },
  { label: "Scores", value: "score_generated" },
  { label: "Verifications", value: "verification_resolved" },
  { label: "Ingests", value: "feature_ingested" },
  { label: "Break-glass", value: "fraud_id_resolved" },
  { label: "Retrain triggers", value: "retrain_trigger" },
  { label: "Drift alerts", value: "drift_alert" },
];
let eventFilter = ""; // active filter value ("" = all)
let loadSeq = 0;       // guard: overlapping load() calls (rapid chip clicks)
                       // let only the newest one render its results

function api(path, pass) {
  return fetch(BASE + path, { headers: { "X-Compliance-Token": pass } })
    .then(async (r) => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
      return data;
    });
}

function fmt(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function short(h) { return h ? String(h).slice(0, 12) + "…" : "—"; }

// Newest-first trail; each row's prev_hash must equal the OLDER sibling's
// entry_hash (the row below it). The last row of a page has no older sibling
// yet: pending until the next page resolves it, unless it is the final row
// of the whole chain, where prev must equal the out-of-band genesis hash.
function renderEvent(e, expectedPrev, genesis) {
  const p = e.payload || {};
  const t = e.event_type;
  const card = document.createElement("div");
  card.className = "card tcard" + (t === "feature_ingested" ? " slim" : "");

  const head = document.createElement("div");
  head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.alignItems = "center"; head.style.gap = "10px";
  head.innerHTML = `<span class="tag event">seq ${e.seq} · ${t}</span>
    <span class="meta" style="margin:0;white-space:nowrap">${fmt(e.created_at)}</span>`;
  card.appendChild(head);

  const body = document.createElement("div");
  if (t === "score_generated") {
    const band = String(p.risk_band || "").toLowerCase();
    const tagCls = band === "high" ? "tag high" : band === "medium" ? "tag medium" : "tag low";
    const row = document.createElement("div");
    row.style.margin = "8px 0 2px";
    row.innerHTML = `<span class="${tagCls}">${band} · ${p.risk_score}</span>` +
      (p.model_version ? `<span class="meta" style="margin-left:8px">model ${p.model_version}</span>` : "");
    body.appendChild(row);
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
    body.appendChild(wrap);
  } else if (t === "verification_resolved") {
    const ok = p.outcome === "confirmed";
    const row = document.createElement("div");
    row.style.margin = "8px 0 2px";
    row.innerHTML = `<span class="tag ${ok ? "low" : "high"}">${p.outcome}</span>` +
      (p.case_id ? `<span class="case" style="margin-left:10px;font-size:12.5px">${p.case_id}</span>` : "");
    body.appendChild(row);
  } else if (t === "fraud_id_resolved") {
    // Break-glass: the bridge was opened for a compliance reason. PII never
    // leaves DB-1; the chain records actor + reason only.
    const row = document.createElement("div");
    row.style.margin = "6px 0 2px";
    row.innerHTML = `<span class="meta">break-glass · actor ${p.actor || "—"} · ${p.reason || ""}</span>`;
    body.appendChild(row);
  } else if (t === "feature_ingested") {
    // Privacy Layer: a transaction was transformed into the feature store.
    const row = document.createElement("div");
    row.style.margin = "4px 0 0";
    row.innerHTML = `<span class="meta">event ${p.event_id || "—"}</span>`;
    body.appendChild(row);
  }
  card.appendChild(body);

  const hash = document.createElement("div");
  hash.className = "hashrow";
  hash.title = "prev_hash=" + e.prev_hash + "\nentry_hash=" + e.entry_hash;
  // Row-to-row link verdicts only hold on the unfiltered trail: under an
  // event-type filter, consecutive rows are not chain-adjacent (other event
  // types sit between them), so the verdict is suppressed. The integrity
  // banner still verifies the whole chain.
  let verdict = "";
  if (!eventFilter) {
    verdict = expectedPrev === null
      ? `<span class="link pending">· link resolves when older events load</span>`
      : (e.prev_hash === expectedPrev
          ? `<span class="link ok">✓ linked</span>`
          : `<span class="link bad">✗ BROKEN LINK</span>`);
  }
  hash.innerHTML = `prev ${short(e.prev_hash)} → entry ${short(e.entry_hash)}` + (verdict ? " · " + verdict : "");
  card.appendChild(hash);
  return card;
}

function updateMoreBtn() {
  $("trailMoreBtn").classList.toggle("hide", trailOffset >= trailTotal);
}

function renderFilterRow() {
  const row = $("filterRow");
  row.className = "chips filterrow";
  row.innerHTML = EVENT_FILTERS.map((f) =>
    `<span class="chip${f.value === eventFilter ? " active" : ""}" data-value="${f.value}" role="button" tabindex="0">${f.label}</span>`
  ).join("");
  row.querySelectorAll(".chip").forEach((c) => {
    const apply = () => {
      const v = c.getAttribute("data-value");
      if (v === eventFilter) return;
      eventFilter = v;
      renderFilterRow();
      load(); // resets paging to page 0 under the new filter
    };
    c.addEventListener("click", apply);
    c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); apply(); } });
  });
}

// Render one API window (newest-first) into the trail. `final` marks the
// window that reaches the chain's tail, where the last row's prev must be
// the genesis hash.
function renderWindow(events, genesis, final) {
  const list = $("trail");
  events.forEach((e, i) => {
    const isLast = i === events.length - 1;
    const expected = isLast ? (final ? genesis : null) : events[i + 1].entry_hash;
    const card = renderEvent(e, expected, genesis);
    list.appendChild(card);
    if (isLast) { lastRowEl = card; lastRowE = e; }
  });
}

// The boundary row of the previous window now has an older sibling: resolve
// its pending link verdict against the first appended event's entry_hash.
function resolveBoundary(firstNew) {
  if (!lastRowEl || !lastRowE || !firstNew) return;
  const ok = lastRowE.prev_hash === firstNew.entry_hash;
  const span = lastRowEl.querySelector(".link");
  if (span) {
    span.className = "link " + (ok ? "ok" : "bad");
    span.textContent = ok ? "✓ linked" : "✗ BROKEN LINK";
  }
}

// Chain-wide summary chips: counts come from the endpoint's by_type +
// no_flag_scores aggregates (not the loaded window), so the numbers are
// stable and accurate no matter how far the user has paged.
function renderSummary(total, byType, noFlag) {
  const chips = $("chips");
  chips.className = "chips";
  const label = eventFilter
    ? (EVENT_FILTERS.find((f) => f.value === eventFilter) || {}).label
    : null;
  chips.innerHTML =
    (label ? `<span class="chip active">filtered: ${label}</span>` : "") +
    `<span class="chip">${total} ${label ? label.toLowerCase() : "events"} total</span>` +
    (noFlag ? `<span class="chip" title="score events with no flagged reasons">no flags ×${noFlag}</span>` : "") +
    Object.entries(byType || {}).map(([t, n]) => `<span class="chip">${t} ×${n}</span>`).join("");
}

async function load() {
  const mySeq = ++loadSeq; // this load is now the newest; older ones must yield
  const pass = $("pass").value.trim();
  const status = $("status");
  status.className = "status"; status.textContent = "";
  if (!pass) { status.className = "status error"; status.textContent = "Enter the compliance passphrase."; return; }
  $("integrityBanner").className = "banner hide";
  $("chips").className = "chips hide";
  $("trail").innerHTML = "<div class='spinner'>Loading trail…</div>";
  $("genesis").className = "genesis hide";
  $("trailStaleHint").classList.add("hide");
  status.textContent = "Verifying chain…";
  try {
    const integrity = await api("/compliance/integrity", pass);
    if (mySeq !== loadSeq) return; // superseded by a newer load()
    const banner = $("integrityBanner");
    if (integrity.ok) {
      banner.className = "banner ok";
      banner.textContent = `✓ Chain verified — ${integrity.n_entries} entries, genesis ${short(integrity.genesis_hash)}`;
    } else {
      banner.className = "banner bad";
      banner.textContent = `✗ CHAIN BROKEN — first bad entry at seq ${integrity.first_bad_seq}`;
    }

    renderFilterRow();
    const q = "/compliance/events?limit=" + TRAIL_PAGE + "&offset=0" +
      (eventFilter ? "&event_type=" + encodeURIComponent(eventFilter) : "");
    const page = await api(q, pass);
    if (mySeq !== loadSeq) return; // superseded by a newer load()
    const events = page.events || [];
    trailTotal = page.total || events.length;
    trailOffset = events.length;
    genesisHash = integrity.genesis_hash || null;
    lastRowEl = lastRowE = null;
    renderSummary(trailTotal, page.by_type, page.no_flag_scores || 0);

    const list = $("trail");
    list.innerHTML = "";
    if (!events.length) {
      list.innerHTML = "<div class='empty'>No audit events yet.</div>";
    } else {
      // newest-first; the final window ends at the chain tail (prev = genesis)
      const final = trailOffset >= trailTotal;
      renderWindow(events, integrity.genesis_hash, final);
      $("genesis").className = "genesis";
      $("genesis").textContent = "genesis " + integrity.genesis_hash;
    }
    updateMoreBtn();
    status.textContent = "";
    refreshBadge(); // footer stays current right after a load/filter change
  } catch (err) {
    if (mySeq !== loadSeq) return; // a stale failure must not clobber a newer load
    status.className = "status error";
    status.textContent = "Failed: " + err.message;
  }
}

// Live chain-status footer badge: polls /compliance/integrity every 60s
// (same cadence as the :8004 footer). Each poll is itself logged to
// audit_access_log - the badge is on the chain, too.
function refreshBadge() {
  const pass = $("pass").value.trim();
  const el = $("chainBadge");
  const text = el.querySelector(".text");
  if (!pass) {
    el.className = "chainline";
    text.textContent = "Audit chain: enter the passphrase to check";
    return;
  }
  api("/compliance/integrity", pass)
    .then((d) => {
      if (d.ok) {
        el.className = "chainline ok";
        text.textContent = "✓ Audit chain verified — " + d.n_entries + " entries · genesis " + short(d.genesis_hash);
      } else {
        el.className = "chainline bad";
        text.textContent = "✗ CHAIN BROKEN — first bad entry at seq " + d.first_bad_seq;
      }
    })
    .catch(() => {
      el.className = "chainline";
      text.textContent = "Audit chain: unavailable";
    });
}

$("trailMoreBtn").onclick = async () => {
  const pass = $("pass").value.trim();
  if (!pass) return;
  const mySeq = loadSeq; // a filter change (new load()) must abandon this append
  const more = $("trailMoreBtn");
  more.disabled = true; more.textContent = "Loading older…";
  try {
    const q = "/compliance/events?limit=" + TRAIL_PAGE + "&offset=" + trailOffset +
      (eventFilter ? "&event_type=" + encodeURIComponent(eventFilter) : "");
    const page = await api(q, pass);
    if (mySeq !== loadSeq) return; // superseded by a filter change
    const events = page.events || [];
    trailTotal = page.total || trailOffset + events.length;
    if (events.length) {
      resolveBoundary(events[0]);
      const final = trailOffset + events.length >= trailTotal;
      renderWindow(events, genesisHash, final);
      trailOffset += events.length;
    }
    updateMoreBtn();
    $("trailStaleHint").classList.add("hide");
  } catch (err) {
    $("status").className = "status error";
    $("status").textContent = "Failed to load older events: " + err.message;
  } finally {
    more.disabled = false; more.textContent = "Show older events";
  }
};

$("loadBtn").onclick = load;
$("pass").addEventListener("keydown", (e) => { if (e.key === "Enter") load(); });

// One click: the dev passphrase is pre-filled, so the trail loads on open.
load();
refreshBadge();
setInterval(refreshBadge, 60000);
