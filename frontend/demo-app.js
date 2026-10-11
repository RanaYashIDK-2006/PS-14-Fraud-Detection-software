/* PS-14 landing page — Live Demo (Phase 20: demo usability & presentation)
 *
 * Everything this file shows comes from the front service's demo API:
 *   GET  /demo/readiness              -> per-service liveness/readiness
 *   POST /demo/synthetic-transaction  -> the REAL Privacy -> Risk response
 *
 * Rules this UI follows:
 *   - never invents a score, band, decision or reason code;
 *   - never renders stack traces, tokens or internal URLs;
 *   - labels every result as synthetic demo output, not real-world validation;
 *   - shows a failure as a failure (no success-looking fallbacks).
 */
(function () {
  'use strict';

  var FRAUD_ID_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no 0/O/1/I
  var STORAGE_KEY = 'ps14_demo_fraud_id';

  var PRESETS = [
    { id: 'everyday', label: 'Everyday purchase', amount: 85, hour_of_day: 14,
      device_id: 'demo-device-01', location_id: 'L-DEMO-A', recipient_id: 'R-DEMO-A',
      failed_auth_count_24h: 0 },
    { id: 'new-device', label: 'Large amount on a new device', amount: 1250, hour_of_day: 3,
      device_id: 'demo-device-02', location_id: 'L-DEMO-B', recipient_id: 'R-DEMO-B',
      failed_auth_count_24h: 1 },
    { id: 'high-risk', label: 'High amount + repeated failed logins', amount: 5000, hour_of_day: 2,
      device_id: 'demo-device-03', location_id: 'L-DEMO-C', recipient_id: 'R-DEMO-C',
      failed_auth_count_24h: 4 }
  ];

  // Plain-language glossary. Keys match the fields the backend actually
  // returns; unknown keys are still shown (raw name + value) rather than hidden.
  var REASON_TEXT = {
    DAILY_SPEND_EXCEEDED: 'Spent more today than the account baseline allows',
    AMOUNT_UNUSUAL: 'Amount is unusual for this account',
    AMOUNT_HIGH: 'Amount is high in absolute terms',
    AMOUNT_EXTREME: 'Amount is extreme in absolute terms',
    NEW_DEVICE: 'Seen from a device never used by this account before',
    NEW_ACCOUNT_LARGE: 'Large amount on a very new account',
    MULE_RING: 'Account links to a shared device/recipient cluster',
    MULE_RING_STRONG: 'Strong shared device/recipient clustering',
    MULE_RING_SCORE: 'Cluster behaviour score crossed the mule threshold',
    SHARED_DEVICE: 'Device is shared with several other accounts',
    SHARED_RECIPIENT: 'Recipient is shared with several other accounts',
    FREQUENCY_ABNORMAL: 'Unusually many transactions in the last 24 hours',
    BEHAVIOR_DEVIATION: 'Behaviour deviates from the learned baseline',
    DOMAIN_SHIFT: 'Input looks unlike the training distribution',
    VELOCITY_HIGH_DAILY_SPEND: 'Daily spend velocity limit crossed (soft)',
    VELOCITY_EXTREME_DAILY_SPEND: 'Daily spend velocity limit crossed (hard-range)',
    VELOCITY_SPEND_NEW_DEVICE: 'Spending quickly from an unfamiliar device',
    FAILED_AUTH_BURST: 'Repeated failed authentications in the last 24 hours',
    UNUSUAL_LOCATION: 'Location is unusual for this account',
    UNUSUAL_RECIPIENT: 'Recipient is unusual for this account'
  };

  var FIELD_HELP = {
    risk_score: 'Blended 0-100 risk score produced by the Risk Engine.',
    risk_band: 'Score grouped into a band (low / medium / high / critical).',
    decision: 'What the pipeline recommends: allow, review, verify or decline.',
    ml_score: 'Model output only — the calibrated ML probability, 0..1.',
    rule_score: 'Rules output only — how strongly the rule engine fired, 0..1.',
    reason_codes: 'Why the score is what it is, as short machine codes.',
    fired_rules: 'Individual rule names that fired (rule output, not ML).',
    degraded: 'True means ML was unavailable and the decision fell back to rules.',
    calibrated: 'True means the ML probability passed through the calibrator.',
    odds: 'Model odds (p / 1-p), clamped for very large values.',
    limits: 'Velocity/policy limits evaluated alongside the score.',
    uncertainty: 'Model dispersion estimates behind the score.',
    feature_version: 'Version tag of the feature contract.',
    rule_version: 'Version tag of the rules configuration.'
  };

  var DECISION_STYLE = {
    allow: 'bg-green-500/10 text-green-400 border-green-500/30',
    review: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    verify: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
    decline: 'bg-red-500/10 text-red-400 border-red-500/30',
    deny: 'bg-red-500/10 text-red-400 border-red-500/30'
  };

  function $(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function demoAccountId() {
    var existing = null;
    try { existing = localStorage.getItem(STORAGE_KEY); } catch (e) { existing = null; }
    if (existing && /^F[A-Z2-9]{15}$/.test(existing)) return existing;
    var out = 'F';
    var bytes = new Uint8Array(15);
    (window.crypto || window.msCrypto).getRandomValues(bytes);
    for (var i = 0; i < 15; i++) out += FRAUD_ID_ALPHABET[bytes[i] % FRAUD_ID_ALPHABET.length];
    try { localStorage.setItem(STORAGE_KEY, out); } catch (e) { /* private mode: keep in-page only */ }
    return out;
  }

  // ---------------------------------------------------------------- readiness
  function chip(label, cls) {
    return '<span class="px-2 py-0.5 rounded-md border text-xs font-medium ' + cls + '">' + esc(label) + '</span>';
  }

  function renderReadiness(data) {
    var host = $('demo-readiness');
    var badge = $('system-status-badge');
    if (!host) return;
    if (!data || !data.services) {
      host.innerHTML = '<span class="text-slate-500 text-xs">Readiness could not be read from /demo/readiness.</span>';
      if (badge) badge.className = 'w-2 h-2 rounded-full bg-slate-500';
      var unknownText = $('system-status-text');
      if (unknownText) unknownText.textContent = 'Status unknown';
      return;
    }
    var rows = Object.keys(data.services).map(function (name) {
      var s = data.services[name];
      var liveness = s.liveness === 'alive'
        ? chip('alive', 'bg-green-500/10 text-green-400 border-green-500/30')
        : chip('unreachable', 'bg-red-500/10 text-red-400 border-red-500/30');
      var readiness = s.readiness
        ? chip('readiness: ' + s.readiness, s.readiness === 'ready'
            ? 'bg-green-500/10 text-green-400 border-green-500/30'
            : 'bg-amber-500/10 text-amber-400 border-amber-500/30')
        : '<span class="text-slate-500">readiness: not published</span>';
      var model = s.model_readiness
        ? chip('model: ' + s.model_readiness, s.model_readiness === 'loaded'
            ? 'bg-green-500/10 text-green-400 border-green-500/30'
            : 'bg-amber-500/10 text-amber-400 border-amber-500/30')
        : '';
      var latency = (s.latency_ms === null || s.latency_ms === undefined)
        ? '' : '<span class="text-slate-600 text-xs">' + esc(s.latency_ms) + ' ms</span>';
      return '<div class="flex flex-wrap items-center gap-2 py-1.5 border-b border-slate-800/40 last:border-0">'
        + '<span class="w-20 text-xs font-semibold text-slate-300">' + esc(name) + '</span>'
        + liveness + readiness + model + latency
        + (s.detail ? '<span class="text-xs text-slate-500">' + esc(s.detail) + '</span>' : '')
        + '</div>';
    }).join('');
    host.innerHTML = rows
      + '<div class="mt-3 text-xs text-slate-500">' + esc(data.note || '') + '</div>';

    var runBtn = $('demo-run');
    if (runBtn) runBtn.disabled = !data.demo_path_ready;
    var warn = $('demo-not-ready-warning');
    if (warn) warn.classList.toggle('hidden', !!data.demo_path_ready);

    if (badge) {
      var text = $('system-status-text');
      if (data.demo_path_ready) {
        if (text) text.textContent = 'Demo path ready';
        badge.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse';
      } else {
        if (text) text.textContent = data.all_services_alive ? 'Demo path degraded' : 'Services unavailable';
        badge.className = 'w-2 h-2 rounded-full bg-amber-500 animate-pulse';
      }
    }
  }

  async function refreshReadiness() {
    try {
      var r = await fetch('/demo/readiness', { headers: { 'Accept': 'application/json' } });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      renderReadiness(await r.json());
    } catch (e) {
      renderReadiness(null);
    }
  }

  // ------------------------------------------------------------------ presets
  function applyPreset() {
    var select = $('demo-preset');
    if (!select) return;
    var preset = PRESETS.filter(function (p) { return p.id === select.value; })[0] || PRESETS[0];
    $('demo-amount').value = preset.amount;
    $('demo-hour').value = preset.hour_of_day;
    $('demo-device').value = preset.device_id;
    $('demo-location').value = preset.location_id;
    $('demo-recipient').value = preset.recipient_id;
    $('demo-failed').value = preset.failed_auth_count_24h;
  }

  // ------------------------------------------------------------------ results
  function metric(label, value, help) {
    return '<div class="glass rounded-xl p-4">'
      + '<div class="text-[11px] text-slate-500 uppercase tracking-wider">' + esc(label) + '</div>'
      + '<div class="mt-1 text-lg font-bold text-white">' + esc(value) + '</div>'
      + (help ? '<div class="mt-1 text-[11px] text-slate-500 leading-snug">' + esc(help) + '</div>' : '')
      + '</div>';
  }

  function renderResult(data) {
    var host = $('demo-result');
    var d = data.decision || {};
    var decisionCls = DECISION_STYLE[String(d.decision || '').toLowerCase()] || 'bg-slate-500/10 text-slate-300 border-slate-500/30';
    var reasons = (d.reason_codes || []).map(function (code) {
      var why = REASON_TEXT[code] || 'No glossary entry — code shown as returned by the backend';
      return '<li class="flex items-start gap-2 py-1">'
        + '<span class="px-1.5 py-0.5 rounded bg-slate-800/70 text-[11px] font-mono text-slate-300">' + esc(code) + '</span>'
        + '<span class="text-xs text-slate-400">' + esc(why) + '</span></li>';
    }).join('') || '<li class="text-xs text-slate-500">No reason codes returned.</li>';

    var triggers = ((d.limits && d.limits.triggers) || []).map(function (t) {
      return '<li class="text-xs text-slate-400">• ' + esc(t.reason_code) + ' (' + esc(t.level) + '): ' + esc(t.detail) + '</li>';
    }).join('');

    var flags = '';
    if (d.degraded === true) flags += chip('degraded: ML unavailable, rules-only fallback', 'bg-amber-500/10 text-amber-400 border-amber-500/30');
    if (d.degraded === false) flags += ' ' + chip('degraded: false (ML path used)', 'bg-green-500/10 text-green-400 border-green-500/30');
    if (d.calibrated === true) flags += ' ' + chip('calibrated probability', 'bg-blue-500/10 text-blue-400 border-blue-500/30');
    if (d.calibrated === false) flags += ' ' + chip('not calibrated', 'bg-amber-500/10 text-amber-400 border-amber-500/30');

    var ingest = data.ingest || {};
    var signals = ['amount_ratio', 'txn_amount_bucket', 'txn_freq_last_24h', 'new_device_flag',
      'unusual_location_flag', 'unusual_recipient_flag', 'failed_auth_count_24h',
      'known_device_count', 'account_tenure_days', 'mule_ring_score']
      .filter(function (k) { return k in ingest; })
      .map(function (k) {
        return '<div class="flex justify-between gap-3 py-0.5 border-b border-slate-800/30 last:border-0">'
          + '<span class="text-[11px] font-mono text-slate-400">' + esc(k) + '</span>'
          + '<span class="text-[11px] text-slate-300">' + esc(ingest[k]) + '</span></div>';
      }).join('');

    host.innerHTML =
      '<div class="flex flex-wrap items-center justify-between gap-3 mb-4">'
      + '<div class="flex items-center gap-3">'
      + '<span class="px-3 py-1.5 rounded-xl border text-sm font-bold uppercase tracking-wide ' + decisionCls + '">decision: ' + esc(d.decision || 'not returned') + '</span>'
      + '<span class="text-2xl font-black text-white">' + esc(d.risk_score) + '</span>'
      + '<span class="text-sm text-slate-400">' + esc(d.risk_band || '') + '</span>'
      + '</div>'
      + '<div class="text-[11px] text-slate-500 font-mono">event_id: ' + esc(data.event_id) + '</div>'
      + '</div>'
      + '<div class="flex flex-wrap gap-2 mb-4">' + flags + '</div>'
      + '<div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">'
      + metric('ML score (model output)', d.ml_score, FIELD_HELP.ml_score)
      + metric('Rule score (rules output)', d.rule_score, FIELD_HELP.rule_score)
      + metric('Limits / policy', (d.limits && d.limits.pass === false) ? 'triggered' : 'pass', FIELD_HELP.limits)
      + '</div>'
      + (triggers ? '<div class="mb-4 text-xs"><div class="text-slate-500 uppercase tracking-wider mb-1">Limit triggers</div><ul>' + triggers + '</ul></div>' : '')
      + '<div class="mb-4"><div class="text-slate-500 uppercase tracking-wider mb-1 text-xs">Reason codes</div><ul class="divide-y divide-slate-800/30">' + reasons + '</ul></div>'
      + (signals ? '<div class="mb-4"><div class="text-slate-500 uppercase tracking-wider mb-1 text-xs">Derived privacy signals (Privacy Layer)</div>' + signals + '</div>' : '')
      + '<details class="glass rounded-xl p-4"><summary class="text-xs text-slate-400 cursor-pointer">Show the raw backend response (exactly as returned)</summary>'
      + '<pre class="mt-3 text-[11px] text-slate-400 whitespace-pre-wrap break-all">' + esc(JSON.stringify(data, null, 2)) + '</pre></details>'
      + '<div class="mt-4 text-[11px] text-slate-500">' + esc(data.note || '') + '</div>';
    host.classList.remove('hidden');
  }

  function renderError(status, detail, service) {
    var host = $('demo-result');
    var hint = '';
    if (status === 503) hint = 'The demo needs the Privacy Layer and Risk Engine running. Use the readiness list above to see which one is missing.';
    if (status === 429) hint = 'The demo endpoint is rate-limited on purpose. Try again in about a minute.';
    host.innerHTML =
      '<div class="border border-red-500/30 bg-red-500/5 rounded-xl p-4">'
      + '<div class="flex items-center gap-2 mb-1">'
      + '<span class="text-red-400 font-semibold text-sm">Demo run failed</span>'
      + '<span class="text-xs text-slate-500">HTTP ' + esc(status) + (service ? ' · ' + esc(service) : '') + '</span>'
      + '</div>'
      + '<div class="text-sm text-slate-300">' + esc(detail || 'No detail returned.') + '</div>'
      + (hint ? '<div class="mt-2 text-xs text-slate-400">' + esc(hint) + '</div>' : '')
      + '<div class="mt-2 text-[11px] text-slate-500">No risk decision was produced, and none is being shown in its place.</div>'
      + '</div>';
    host.classList.remove('hidden');
  }

  async function runDemo() {
    var btn = $('demo-run');
    var out = $('demo-result');
    var body = {
      fraud_id: $('demo-account').value.trim() || null,
      amount: parseFloat($('demo-amount').value),
      hour_of_day: parseInt($('demo-hour').value, 10),
      device_id: $('demo-device').value.trim() || null,
      location_id: $('demo-location').value.trim() || null,
      recipient_id: $('demo-recipient').value.trim() || null,
      failed_auth_count_24h: parseInt($('demo-failed').value, 10) || 0
    };
    if (!(body.amount > 0)) {
      renderError(422, 'Amount must be a positive number.');
      return;
    }
    btn.disabled = true;
    btn.textContent = 'Running…';
    out.innerHTML = '<div class="glass rounded-xl p-6 text-sm text-slate-400 animate-pulse">Scoring a synthetic transaction through the Privacy Layer and Risk Engine…</div>';
    out.classList.remove('hidden');
    try {
      var r = await fetch('/demo/synthetic-transaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'demo' },
        body: JSON.stringify(body)
      });
      var data = null;
      try { data = await r.json(); } catch (e) { data = null; }
      if (!r.ok) {
        renderError(r.status, data && data.detail, data && data.detail ? '' : '');
        return;
      }
      renderResult(data);
    } catch (e) {
      renderError(0, 'The front service did not answer. Is it still running on port 8000?');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run synthetic transaction';
      refreshReadiness();
    }
  }

  function newAccount() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    var id = demoAccountId();
    $('demo-account').value = id;
    $('demo-account-note').textContent = 'Repeat runs reuse this synthetic account, so the Privacy Layer builds up history across runs.';
  }

  document.addEventListener('DOMContentLoaded', function () {
    if (!$('demo-run')) return;
    $('demo-account').value = demoAccountId();
    var select = $('demo-preset');
    PRESETS.forEach(function (p) {
      var opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.label;
      select.appendChild(opt);
    });
    select.addEventListener('change', applyPreset);
    applyPreset();
    $('demo-run').addEventListener('click', runDemo);
    $('demo-new-account').addEventListener('click', newAccount);
    refreshReadiness();
    setInterval(refreshReadiness, 10000);
  });
})();
