/* PS-14 Demo & Reports — Phase 20
 *
 * Talks only to the front service's own API:
 *   GET  /demo/readiness
 *   POST /demo/batch/run
 *   GET  /demo/batch/progress
 *   GET  /demo/reports
 *   GET  /demo/reports/{id}
 *   GET  /demo/reports/{id}/export
 *
 * Rules: never invent a number, never render a stack trace or token, never
 * present a failed/partial run as a completed report.
 */
(function () {
  'use strict';

  var POLL_MS = 1500;
  var REFRESH_MS = 10000;
  var state = { active: null, reports: [], report: null, pollTimer: null, chart: null };

  function $(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function fmtInt(n) {
    return (n === null || n === undefined || isNaN(n)) ? '—' : Number(n).toLocaleString('en-US');
  }

  function fmtPct(x, digits) {
    return (x === null || x === undefined || isNaN(x)) ? '—' : (100 * Number(x)).toFixed(digits === undefined ? 1 : digits) + '%';
  }

  function fmtNum(x, digits) {
    return (x === null || x === undefined || isNaN(x)) ? '—' : Number(x).toFixed(digits === undefined ? 3 : digits);
  }

  function fmtTime(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleString(); } catch (e) { return esc(iso); }
  }

  async function getJSON(url) {
    var r = await fetch(url, { headers: { 'Accept': 'application/json' } });
    var body = null;
    try { body = await r.json(); } catch (e) { body = null; }
    if (!r.ok) {
      var err = new Error((body && body.detail) ? String(body.detail) : ('HTTP ' + r.status));
      err.status = r.status;
      throw err;
    }
    return body;
  }

  // ─────────────────────────────────────────────────────────── tabs
  function showTab(name) {
    var overview = name === 'overview';
    $('view-overview').classList.toggle('hidden', !overview);
    $('view-reports').classList.toggle('hidden', overview);
    $('tab-overview').setAttribute('aria-selected', String(overview));
    $('tab-reports').setAttribute('aria-selected', String(!overview));
    if (!overview) refreshReports();
  }

  // ─────────────────────────────────────────────────────── readiness
  async function refreshReadiness() {
    var chip = $('readiness-chip');
    var host = $('readiness-list');
    try {
      var data = await getJSON('/demo/readiness');
      var services = data.services || {};
      var rows = Object.keys(services).map(function (name) {
        var s = services[name];
        var cls = s.liveness === 'alive' ? 'chip-ok' : 'chip-bad';
        var readiness = s.readiness ? ('readiness: ' + s.readiness) : 'readiness not published';
        var model = s.model_readiness ? (' · model: ' + s.model_readiness) : '';
        return '<div style="display:flex;gap:8px;align-items:center;padding:5px 0;border-bottom:1px solid var(--border)">'
          + '<span class="mono" style="width:70px">' + esc(name) + '</span>'
          + '<span class="chip ' + cls + '"><span class="dot"></span>' + esc(s.liveness) + '</span>'
          + '<span class="faint">' + esc(readiness + model) + '</span>'
          + '</div>';
      }).join('');
      host.innerHTML = rows || '<span class="faint">No service data.</span>';

      if (data.demo_path_ready) {
        chip.className = 'chip chip-ok';
        chip.innerHTML = '<span class="dot"></span> Demo path ready';
      } else {
        chip.className = 'chip chip-warn';
        chip.innerHTML = '<span class="dot"></span> ' + (data.all_services_alive ? 'Demo path degraded' : 'Services unavailable');
      }
      $('run-button').disabled = false;
      if (!data.demo_path_ready) {
        host.insertAdjacentHTML('beforeend',
          '<div class="notice notice-warn" style="margin-top:10px"><strong>Demo path not fully ready</strong>' +
          'Offline mode still works (it loads the model in its own process). Service-backed mode needs the Risk Engine reachable.</div>');
      }
    } catch (e) {
      chip.className = 'chip chip-bad';
      chip.innerHTML = '<span class="dot"></span> Status unknown';
      host.innerHTML = '<span class="faint">Readiness could not be read from /demo/readiness.</span>';
      $('run-button').disabled = false;
    }
  }

  // ─────────────────────────────────────────────────────────── run
  async function startRun() {
    var mode = $('run-mode').value;
    var btn = $('run-button');
    btn.disabled = true;
    $('run-error').classList.add('hidden');
    $('run-progress').classList.remove('hidden');
    setProgress(0, 'Submitting run…');
    try {
      var r = await fetch('/demo/batch/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'demo' },
        body: JSON.stringify({ mode: mode, rows: 25000, seed: 42 })
      });
      var body = null;
      try { body = await r.json(); } catch (e) { body = null; }
      if (!r.ok) {
        throw new Error(body && body.detail ? String(body.detail) : ('HTTP ' + r.status));
      }
      state.active = body;
      beginPolling();
    } catch (e) {
      showRunError(e.message);
      btn.disabled = false;
    }
  }

  function showRunError(message) {
    $('run-error').classList.remove('hidden');
    $('run-error-detail').textContent = message;
    $('run-progress').classList.add('hidden');
  }

  function setProgress(pct, text) {
    var fill = $('progress-fill');
    fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
    $('progress-bar-wrap').setAttribute('aria-valuenow', String(Math.round(pct)));
    $('run-status').textContent = text;
  }

  function beginPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollProgress, POLL_MS);
    pollProgress();
  }

  async function pollProgress() {
    try {
      var data = await getJSON('/demo/batch/progress');
      if (data.active) {
        var a = data.active;
        var pct = a.total ? (100 * a.processed / a.total) : 0;
        setProgress(pct, (a.state === 'scoring'
          ? ('Scoring ' + fmtInt(a.processed) + ' / ' + fmtInt(a.total) + ' transactions')
          : esc(a.state)) + ' · ' + esc(a.mode) + ' mode · report ' + esc(a.report_id));
        $('run-button').disabled = true;
        return;
      }
      // No active run: stop polling and reconcile the latest report.
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      $('run-button').disabled = false;
      $('run-progress').classList.add('hidden');
      var latest = data.latest;
      if (latest && latest.state === 'completed') {
        await openReport(latest.report_id, true);
      } else if (latest && latest.state === 'failed') {
        showRunError('The run failed before producing a report. See the report history for details.');
      }
      refreshReports();
      refreshLatestPanel();
    } catch (e) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      $('run-button').disabled = false;
      showRunError(e.message);
    }
  }

  // ─────────────────────────────────────────────────────── reports
  async function refreshReports() {
    try {
      var data = await getJSON('/demo/reports');
      state.reports = data.reports || [];
      renderHistory();
      if (!state.report && state.reports.length && state.reports[0].state === 'completed') {
        await openReport(state.reports[0].report_id, false);
      }
    } catch (e) {
      $('history-rows').innerHTML = '<tr><td colspan="7" class="faint">' + esc(e.message) + '</td></tr>';
    }
  }

  function renderHistory() {
    var rows = state.reports.map(function (r) {
      var stateChip = r.state === 'completed' ? 'chip-ok' : (r.state === 'failed' ? 'chip-bad' : 'chip-warn');
      return '<tr>'
        + '<td class="mono">' + esc(r.report_id) + '</td>'
        + '<td><span class="chip ' + stateChip + '"><span class="dot"></span>' + esc(r.state || 'unknown') + '</span></td>'
        + '<td>' + esc(r.execution_mode || '—') + '</td>'
        + '<td class="n">' + fmtInt(r.rows_scored !== undefined ? r.rows_scored : r.rows) + '</td>'
        + '<td class="n">' + (r.elapsed_s ? fmtNum(r.elapsed_s, 1) + ' s' : '—') + '</td>'
        + '<td class="faint">' + fmtTime(r.started_at) + '</td>'
        + '<td><button class="btn btn-sm" data-report="' + esc(r.report_id) + '">Open</button></td>'
        + '</tr>';
    }).join('');
    $('history-rows').innerHTML = rows || '<tr><td colspan="7" class="faint">No reports yet.</td></tr>';
    Array.prototype.forEach.call($('history-rows').querySelectorAll('button[data-report]'), function (btn) {
      btn.addEventListener('click', function () { openReport(btn.getAttribute('data-report'), false); });
    });
  }

  async function refreshLatestPanel() {
    var host = $('latest-report');
    try {
      var data = await getJSON('/demo/reports');
      var latest = (data.reports || [])[0];
      if (!latest) {
        host.innerHTML = '<p class="faint" style="margin:0">No reports yet. Run the demo to create one.</p>';
        return;
      }
      var decisions = latest.decision_counts
        ? Object.keys(latest.decision_counts).map(function (k) { return k + ' ' + fmtInt(latest.decision_counts[k]); }).join(' · ')
        : '—';
      host.innerHTML =
        '<div class="mono faint" style="margin-bottom:6px">' + esc(latest.report_id) + '</div>'
        + '<div class="muted" style="font-size:13px;margin-bottom:8px">' + esc(decisions) + '</div>'
        + '<button class="btn btn-sm" id="open-latest">Open report</button>';
      var open = $('open-latest');
      if (open) open.addEventListener('click', function () { openReport(latest.report_id, true); });
    } catch (e) {
      host.innerHTML = '<p class="faint" style="margin:0">Latest report could not be read.</p>';
    }
  }

  async function openReport(reportId, switchTab) {
    try {
      var report = await getJSON('/demo/reports/' + encodeURIComponent(reportId));
      state.report = report;
      renderReport(report);
      if (switchTab) showTab('reports');
    } catch (e) {
      if (e.status === 409) { setTimeout(function () { openReport(reportId, switchTab); }, 2000); return; }
      $('report-empty').classList.remove('hidden');
      $('report-body').classList.add('hidden');
      $('report-empty').innerHTML = '<div class="notice notice-error"><strong>Report unavailable</strong>' + esc(e.message) + '</div>';
    }
  }

  function renderReport(r) {
    $('report-empty').classList.add('hidden');
    $('report-body').classList.remove('hidden');

    $('rep-id').textContent = r.report_id || '—';
    var stateChip = $('rep-state');
    stateChip.className = 'chip ' + (r.state === 'completed' ? 'chip-ok' : (r.state === 'failed' ? 'chip-bad' : 'chip-warn'));
    stateChip.innerHTML = '<span class="dot"></span> ' + esc(r.state || 'unknown');

    var exportBase = '/demo/reports/' + encodeURIComponent(r.report_id) + '/export';
    $('rep-export-json').href = exportBase + '?format=json';
    $('rep-export-csv').href = exportBase + '?format=csv';
    $('rep-rerun').onclick = function () { showTab('overview'); startRun(); };

    $('rep-mode').textContent = r.execution_mode_label || r.execution_mode || '—';

    var ds = r.dataset || {};
    var sel = ds.selection || {};
    var val = r.validation || {};
    $('rep-dataset').textContent = (ds.type === 'synthetic' ? 'Synthetic (repository generator)' : String(ds.type || '—'))
      + (ds.label_provenance ? ' · labels: ' + ds.label_provenance : '');
    $('rep-rows').textContent = fmtInt(sel.rows_selected) + ' / '
      + fmtInt((r.summary || {}).rows_scored) + ' / ' + fmtInt(val.rows_skipped_invalid);
    $('rep-times').textContent = fmtTime(r.started_at) + ' → ' + fmtTime(r.completed_at);
    $('rep-elapsed').textContent = r.elapsed_s ? fmtNum(r.elapsed_s, 2) + ' s' : '—';
    $('rep-hash').textContent = (ds.sha256 || '—');
    $('rep-seed').textContent = (ds.generator_params ? ('seed ' + ds.generator_params.seed + ' · noise ' + ds.generator_params.noise) : '—')
      + ' · ' + (ds.schema_version || '—');
    var repro = r.reproducibility || {};
    $('rep-git').textContent = repro.git_sha || '—';
    var eng = (r.execution || {}).engine;
    $('rep-engine').textContent = eng ? (eng.engine_class + ' · ' + (eng.artifacts_fingerprint || '—')) : (r.execution_mode === 'live' ? 'Risk Engine service' : '—');

    var summary = r.summary || {};
    $('stat-processed').textContent = fmtInt(summary.rows_scored);
    $('stat-processed-sub').textContent = fmtInt(val.rows_skipped_invalid) + ' skipped · '
      + fmtInt(val.rows_with_missing_features) + ' rows had missing features';
    $('stat-throughput').textContent = (r.execution && r.execution.throughput_rows_per_s)
      ? fmtInt(r.execution.throughput_rows_per_s) + '/s' : '—';
    $('stat-degraded').textContent = fmtInt(summary.degraded_rows);
    $('stat-degraded-sub').textContent = summary.degraded_rows ? 'decisions fell back to rules-only' : 'no degraded rows in this run';

    renderDecisions(summary, r.metrics || {});
    renderMetrics(r.metrics || {}, summary);
    renderOps(r);
    renderReasons(summary.top_reason_codes || {});

    var limits = (r.limitations || []).concat(r.warnings ? [] : []);
    var items = (r.limitations || []).map(function (l) { return '<li>' + esc(l) + '</li>'; });
    if (r.warnings && r.warnings.length) {
      r.warnings.forEach(function (w) { items.push('<li><strong>Warning:</strong> ' + esc(w) + '</li>'); });
    }
    $('limitation-list').innerHTML = items.join('') || '<li>None recorded.</li>';
  }

  function renderDecisions(summary, metrics) {
    var counts = summary.decision_counts || {};
    var share = summary.decision_share || {};
    var perDecision = (metrics.per_decision) || {};
    var keys = Object.keys(counts);
    var rows = keys.map(function (k) {
      var pd = perDecision[k] || {};
      return '<tr><td class="mono">' + esc(k) + '</td>'
        + '<td class="n">' + fmtInt(counts[k]) + '</td>'
        + '<td class="n">' + fmtPct(share[k], 2) + '</td>'
        + '<td class="n">' + (pd.synthetic_fraud_rate === undefined ? '—' : fmtPct(pd.synthetic_fraud_rate, 2)) + '</td></tr>';
    }).join('');
    var total = keys.reduce(function (acc, k) { return acc + counts[k]; }, 0);
    rows += '<tr><td class="muted">total</td><td class="n muted">' + fmtInt(total) + '</td>'
      + '<td class="n muted">' + (total ? '100%' : '—') + '</td><td class="n muted"></td></tr>';
    $('decision-rows').innerHTML = rows || '<tr><td colspan="4" class="faint">No decisions recorded.</td></tr>';

    var canvas = $('decision-chart');
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    if (!keys.length || typeof Chart === 'undefined') { canvas.parentElement.classList.add('hidden'); return; }
    canvas.parentElement.classList.remove('hidden');
    state.chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: keys,
        datasets: [{ label: 'Rows', data: keys.map(function (k) { return counts[k]; }),
          backgroundColor: '#14b8a6', borderColor: '#0f766e', borderWidth: 1 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: function (ctx) { return fmtInt(ctx.parsed.y) + ' rows'; } } } },
        scales: {
          x: { ticks: { color: '#93a1ae' }, grid: { display: false } },
          y: { ticks: { color: '#93a1ae', callback: function (v) { return fmtInt(v); } },
               grid: { color: '#1f2933' }, beginAtZero: true }
        }
      }
    });
  }

  function renderMetrics(metrics, summary) {
    var host = $('metrics-body');
    if (!metrics.available) {
      host.innerHTML = '<div class="notice notice-warn"><strong>Metrics unavailable</strong>'
        + esc(metrics.reason || 'no reason recorded') + '</div>';
      return;
    }
    var rank = metrics.ranking || {};
    var c = metrics.confusion || {};
    var rankBlock = rank.available
      ? '<div class="grid grid-2" style="margin-bottom:12px">'
        + '<div class="stat"><div class="label">ROC-AUC</div><div class="value num">' + fmtNum(rank.roc_auc, 4) + '</div></div>'
        + '<div class="stat"><div class="label">PR-AUC</div><div class="value num">' + fmtNum(rank.pr_auc, 4) + '</div></div>'
        + '</div>'
      : '<div class="notice notice-warn"><strong>Ranking metrics unavailable</strong>' + esc(rank.reason || '') + '</div>';

    host.innerHTML = rankBlock
      + '<div class="grid grid-3" style="margin-bottom:12px">'
      + '<div class="stat"><div class="label">Precision</div><div class="value num">' + fmtNum(metrics.precision, 3) + '</div></div>'
      + '<div class="stat"><div class="label">Recall</div><div class="value num">' + fmtNum(metrics.recall, 3) + '</div></div>'
      + '<div class="stat"><div class="label">F1</div><div class="value num">' + fmtNum(metrics.f1, 3) + '</div></div>'
      + '</div>'
      + '<div class="table-scroll"><table><thead><tr><th></th><th class="n">Predicted flagged</th><th class="n">Predicted allow</th></tr></thead>'
      + '<tbody>'
      + '<tr><td>Synthetic fraud</td><td class="n">' + fmtInt(c.tp) + '</td><td class="n">' + fmtInt(c.fn) + '</td></tr>'
      + '<tr><td>Synthetic legit</td><td class="n">' + fmtInt(c.fp) + '</td><td class="n">' + fmtInt(c.tn) + '</td></tr>'
      + '</tbody></table></div>'
      + '<dl class="kv" style="margin-top:12px">'
      + '<dt>Labels</dt><dd>' + esc(metrics.labels || '—') + '</dd>'
      + '<dt>Flagged means</dt><dd>' + esc(metrics.flagged_definition || '—') + '</dd>'
      + '</dl>'
      + '<div class="notice notice-info" style="margin-top:10px"><strong>Circularity check</strong>'
      + esc(metrics.circularity_note || '') + '</div>'
      + '<p class="faint">Split: the whole synthetic corpus was scored; there is no held-out split inside this demo run.</p>';
  }

  function renderOps(r) {
    var exec = r.execution || {};
    var val = r.validation || {};
    var rows = [
      ['Evaluation mode', r.execution_mode_label || r.execution_mode || '—'],
      ['Execution elapsed', exec.elapsed_s ? fmtNum(exec.elapsed_s, 2) + ' s' : '—'],
      ['Throughput', exec.throughput_rows_per_s ? fmtInt(exec.throughput_rows_per_s) + ' rows/s' : '—'],
      ['Rows in dataset', fmtInt(val.rows_in_file)],
      ['Rows usable', fmtInt(val.rows_usable)],
      ['Rows skipped (invalid)', fmtInt(val.rows_skipped_invalid)],
      ['Rows with missing features', fmtInt(val.rows_with_missing_features)],
      ['Duplicate event ids', val.duplicate_event_ids === null || val.duplicate_event_ids === undefined ? '—' : fmtInt(val.duplicate_event_ids)],
      ['Domain-incompatible rows', fmtInt((r.summary || {}).domain_incompatible_rows)],
      ['Service chunk failures', exec.failed_rows === undefined ? '—' : fmtInt(exec.failed_rows)]
    ];
    $('ops-kv').innerHTML = rows.map(function (kv) {
      return '<dt>' + esc(kv[0]) + '</dt><dd class="num">' + esc(kv[1]) + '</dd>';
    }).join('');
    var warnings = r.warnings || [];
    $('ops-warnings').innerHTML = warnings.length
      ? '<div class="notice notice-warn"><strong>Warnings</strong><ul class="tight">'
        + warnings.map(function (w) { return '<li>' + esc(w) + '</li>'; }).join('') + '</ul></div>'
      : '<p class="faint" style="margin:0">No warnings recorded.</p>';
  }

  function renderReasons(counts) {
    var keys = Object.keys(counts);
    $('reason-rows').innerHTML = keys.length
      ? keys.map(function (k) {
          return '<tr><td class="mono">' + esc(k) + '</td><td class="n">' + fmtInt(counts[k]) + '</td></tr>';
        }).join('')
      : '<tr><td colspan="2" class="faint">No reason codes recorded.</td></tr>';
  }

  // ───────────────────────────────────────────────────────── boot
  function boot() {
    $('tab-overview').addEventListener('click', function () { showTab('overview'); });
    $('tab-reports').addEventListener('click', function () { showTab('reports'); });
    $('run-button').addEventListener('click', startRun);
    refreshReadiness();
    refreshLatestPanel();
    refreshReports();
    setInterval(refreshReadiness, REFRESH_MS);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
