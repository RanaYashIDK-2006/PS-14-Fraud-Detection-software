var C={green:'#22c55e',red:'#ef4444',accent:'#3b82f6',purple:'#a855f7',yellow:'#eab308',dim:'#64748b',grid:'#1e293b',orange:'#f97316'};

/* Admin-authenticated fetch: the monitor is admin-gated server-side, so every
   call must carry the admin Bearer token (sessionStorage from the admin login).
   On 401, bounce to the login page. */
function authFetch(url, opts) {
  opts = opts || {};
  var token = sessionStorage.getItem('admin_token');
  opts.headers = Object.assign({}, opts.headers || {}, token ? {'Authorization': 'Bearer ' + token} : {});
  return fetch(url, opts).then(function(r) {
    if (r.status === 401) { window.location.href = '/login-page?next=/monitor-page'; throw new Error('unauthorized'); }
    return r;
  });
}

function fmt(s){var h=Math.floor(s/3600),m=Math.floor((s%3600)/60);return h>0?h+'h '+m+'m':m+'m';}

/* LIVE METRICS */
function renderLiveMetrics(data){
  var m=data.metrics||{};
  var l=data.latency||{};
  var svc=data.services||{};
  var uptime=data.uptime_seconds||0;
  var recent=data.recent_events||[];

  document.getElementById('mEval').textContent=m.total_evaluations||0;
  document.getElementById('mFraud').textContent=m.fraud_detected||0;
  document.getElementById('mFP').textContent=m.legit_flagged||0;
  document.getElementById('mDet').textContent=(m.detection_rate_pct||0)+'%';
  document.getElementById('mFPR').textContent=(m.fpr_pct||0)+'%';
  document.getElementById('mUp').textContent=fmt(uptime);

  var svcHtml='';
  for(var name in svc){
    var st=svc[name].status||'unknown';
    var lat=svc[name].latency_ms?svc[name].latency_ms+'ms':'—';
    svcHtml+='<div class="service-card"><div class="service-dot '+st+'"></div><div><div class="service-name">'+name+'</div><div class="service-info">'+(st==='up'?'Healthy · '+lat:st==='down'?'DOWN':'Unknown')+'</div></div></div>';
  }
  document.getElementById('svcGrid').innerHTML=svcHtml;

  document.getElementById('lAvg').textContent=l.avg_ms||0;
  document.getElementById('lP50').textContent=l.p50_ms||0;
  document.getElementById('lP95').textContent=l.p95_ms||0;
  document.getElementById('lP99').textContent=l.p99_ms||0;
  document.getElementById('lSamp').textContent=l.samples||0;

  var events=[].concat(recent).reverse();
  var tbody='';
  for(var i=0;i<events.length;i++){
    var e=events[i];
    var t=new Date(e.ts).toLocaleTimeString();
    var sc=e.score||0;
    var cls=e.fraud?'fraud':e.flagged?'fraud':'legit';
    var label=e.fraud?'FRAUD':e.flagged?'FLAGGED':'OK';
    tbody+='<tr><td>'+t+'</td><td style="font-weight:600;color:'+(sc>=80?C.red:sc>=50?C.orange:sc>=30?C.yellow:C.green)+'">'+sc+'</td><td><span class="badge '+cls+'">'+label+'</span></td><td>'+(e.latency_ms?e.latency_ms+'ms':'—')+'</td></tr>';
  }
  document.getElementById('eventsBody').innerHTML=tbody;
}

/* PHASES */
function loadPhases(){
  authFetch('/monitor/phases').then(function(r){return r.json()}).then(function(data){
    var phases=data.phases||[];
    var tbody='';
    for(var i=0;i<phases.length;i++){
      var p=phases[i];
      var cls=p.classification||'UNKNOWN';
      var badgeClass='info';
      if(cls.indexOf('BLOCKED')>=0||cls.indexOf('NO_ROBUST')>=0||cls.indexOf('ARTIFACT')>=0) badgeClass='warn';
      else if(cls.indexOf('CLEAN')>=0||cls.indexOf('READY')>=0) badgeClass='pass';
      else if(cls.indexOf('FAIL')>=0||cls.indexOf('METHODOLOGY')>=0) badgeClass='fail';
      var fw=p.firewall||{};
      var fwHtml='<span style="color:'+(fw.final_test_accessed===false?C.green:C.red)+'">FT:'+(fw.final_test_accessed===false?'✓':'✗')+'</span>';
      tbody+='<tr><td><b style="color:'+C.accent+'">'+p.phase+'</b></td><td>'+p.title+'</td><td><span class="badge '+badgeClass+'">'+cls+'</span></td><td style="font-size:10px">'+fwHtml+'</td></tr>';
    }
    document.getElementById('phaseBody').innerHTML=tbody;
  }).catch(function(e){
    document.getElementById('phaseBody').innerHTML='<tr><td colspan="4" style="color:'+C.red+'">Failed to load phases</td></tr>';
  });
}

/* TEST RESULTS */
function loadTestResults(){
  authFetch('/monitor/test-results').then(function(r){return r.json()}).then(function(d){
    var el=document.getElementById('testResults');
    if(!d||!d.timestamp){el.innerHTML='<div style="color:'+C.dim+';font-size:12px">No test results yet — first run pending</div>';return;}
    var ts=new Date(d.timestamp).toLocaleString();
    var allPass=d.all_passed;
    var dotColor=allPass?C.green:C.red;
    var statusText=allPass?'ALL PASSED':'FAILURES DETECTED';
    var html='<div class="test-header"><div class="test-status"><div class="dot" style="background:'+dotColor+'"></div><div class="text" style="color:'+dotColor+'">'+statusText+'</div></div><div style="color:'+C.dim+';font-size:11px;margin-left:auto">'+ts+'</div></div>';
    html+='<div class="test-counts"><span class="p">✓ '+d.passed+' passed</span>';
    if(d.failed>0)html+=' <span class="f">✗ '+d.failed+' failed</span>';
    html+=' <span style="color:'+C.dim+'">'+d.total+' total</span></div>';
    if(d.failed_tests&&d.failed_tests.length>0){
      html+='<div style="margin-top:8px">';
      for(var i=0;i<d.failed_tests.length;i++)html+='<span class="test-badge fail">✗ '+d.failed_tests[i]+'</span> ';
      html+='</div>';
    }
    if(d.passed_tests&&d.passed_tests.length>0){
      html+='<div class="test-badges">';
      for(var j=0;j<d.passed_tests.length;j++)html+='<span class="test-badge pass">✓ '+d.passed_tests[j]+'</span>';
      html+='</div>';
    }
    if(d.error)html+='<div style="color:'+C.red+';font-size:11px;margin-top:8px">'+d.error+'</div>';
    el.innerHTML=html;
  }).catch(function(e){
    document.getElementById('testResults').innerHTML='<div style="color:'+C.red+';font-size:12px">Failed to load test results</div>';
  });
}

/* RUN TESTS */
document.getElementById('runBtn').addEventListener('click',function(){
  var btn=this;
  btn.disabled=true;btn.textContent='⏳ Running…';btn.classList.add('running');
  authFetch('/monitor/run-tests',{method:'POST',headers:{'X-Requested-With':'fetch'}}).then(function(r){return r.json()}).then(function(){
    btn.disabled=false;btn.textContent='▶ Run Tests';btn.classList.remove('running');
    loadTestResults();
  }).catch(function(){btn.disabled=false;btn.textContent='▶ Run Tests';btn.classList.remove('running');});
});

/* MAIN LOOP */
function refresh(){
  authFetch('/monitor/metrics').then(function(r){return r.json()}).then(function(data){
    renderLiveMetrics(data);
  }).catch(function(e){console.error('Metrics load failed:',e);});
}

// Initial load
refresh();
loadPhases();
loadTestResults();

// Auto-refresh
setInterval(refresh,5000);
setInterval(loadPhases,30000);
setInterval(loadTestResults,60000);
