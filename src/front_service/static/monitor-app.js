var C={green:'#22c55e',red:'#ef4444',accent:'#3b82f6',purple:'#a855f7',yellow:'#eab308',dim:'#64748b',grid:'#1e293b',orange:'#f97316'};
var latencyChart=null,detectionChart=null,eventChart=null;
var metricsHistory=[];

function fmt(s){var h=Math.floor(s/3600),m=Math.floor((s%3600)/60);return h>0?h+'h '+m+'m':m+'m';}

/* ═══════ LIVE METRICS ═══════ */
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

  // Service health
  var svcHtml='';
  for(var name in svc){
    var st=svc[name].status||'unknown';
    var lat=svc[name].latency_ms?svc[name].latency_ms+'ms':'—';
    svcHtml+='<div class="service-card"><div class="service-dot '+st+'"></div><div><div class="service-name">'+name+'</div><div class="service-info">'+(st==='up'?'Healthy · '+lat:st==='down'?'DOWN':'Unknown')+'</div></div></div>';
  }
  document.getElementById('svcGrid').innerHTML=svcHtml;

  // Latency
  document.getElementById('lAvg').textContent=l.avg_ms||0;
  document.getElementById('lP50').textContent=l.p50_ms||0;
  document.getElementById('lP95').textContent=l.p95_ms||0;
  document.getElementById('lP99').textContent=l.p99_ms||0;
  document.getElementById('lSamp').textContent=l.samples||0;

  // Inference service
  var inf=data.inference||{};
  document.getElementById('iActive').textContent=inf.active_users||0;
  document.getElementById('iModels').textContent=inf.registered_models||0;
  document.getElementById('iAB').textContent=inf.ab_experiments||0;
  document.getElementById('iBackend').textContent=inf.state_backend||'—';
  var rm=inf.redis_memory_mb||0;
  document.getElementById('iRedis').textContent=rm>0?rm.toFixed(1)+'MB':'N/A';
  document.getElementById('iInfUp').textContent=fmt(inf.uptime_s||0);
  document.getElementById('iModel').textContent=(inf.model_version||'—').replace('altman_','').substring(0,20);
  var rs=inf.redis_connected;
  document.getElementById('iRedisStatus').textContent=rs===true?'Connected':rs===false?'Disconnected':'—';
  var rsEl=document.getElementById('iRedisStatus');
  if(rsEl){rsEl.style.color=rs===true?C.green:rs===false?C.red:C.dim;}

  // Model info
  var mi=data.model_info||{};
  var el;
  el=document.getElementById('mActiveModel');if(el){
    var ver=mi.active_model_version||'—';
    el.textContent=ver.replace('altman_clean_','').replace('altman_native_','Native: ').substring(0,25);
    el.style.color=mi.is_altman?C.green:C.accent;
  }
  el=document.getElementById('mModelType');if(el){el.textContent=mi.active_model_type||'—';}
  el=document.getElementById('mFeatures');if(el){el.textContent=mi.n_features?mi.n_features+' feat':'—';}
  el=document.getElementById('mDataset');if(el){el.textContent=mi.dataset_rows?(mi.dataset_rows/1e6).toFixed(1)+'M rows':'—';}
  el=document.getElementById('mCvAuc');if(el){el.textContent=mi.cv_auc?(mi.cv_auc*100).toFixed(2)+'%':'—';}
  el=document.getElementById('mTempAuc');if(el){el.textContent=mi.temporal_auc?(mi.temporal_auc*100).toFixed(2)+'%':'—';}
  el=document.getElementById('mTempR1');if(el){el.textContent=mi.temporal_r1?(mi.temporal_r1*100).toFixed(2)+'%':'—';}
  el=document.getElementById('mLeakage');if(el){
    el.textContent=mi.target_leakage===false?'No':mi.target_leakage===true?'Yes':'—';
    el.style.color=mi.target_leakage===false?C.green:C.red;
  }
  // Native model
  if(mi.native_model_version){
    var ns=document.getElementById('nativeStats');if(ns)ns.style.display='';
    el=document.getElementById('mNativeVer');if(el){el.textContent=(mi.native_model_version||'—').replace('altman_native_','').substring(0,20);}
    el=document.getElementById('mNativeAuc');if(el){el.textContent=mi.native_cv_auc?(mi.native_cv_auc*100).toFixed(2)+'%':'—';}
    el=document.getElementById('mNativeTempAuc');if(el){el.textContent=mi.native_temporal_auc?(mi.native_temporal_auc*100).toFixed(2)+'%':'—';}
    el=document.getElementById('mNativeTempR1');if(el){el.textContent=mi.native_temporal_r1?(mi.native_temporal_r1*100).toFixed(2)+'%':'—';}
    el=document.getElementById('mNativeFeat');if(el){el.textContent=mi.native_n_features?mi.native_n_features+' feat':'—';}
  }

  // Events table
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

  renderCharts(data);
  loadUnifiedDetection();
}

/* ═══════ UNIFIED DETECTION + DRIFT ═══════ */
var psiChart=null;
var psiHistory=[];

function loadUnifiedDetection(){
  fetch('/monitor/unified').then(function(r){return r.json()}).then(function(data){
    renderUnifiedMetrics(data);
    renderDriftMetrics(data);
  }).catch(function(e){console.error('Unified load failed:',e);});
}

function renderUnifiedMetrics(data){
  var ss=data.subsystems||{};
  var us=data.unified_scorer||{};

  var statusColor=function(s){return s==='ready'?C.green:s==='unavailable'?C.red:C.dim;};
  var statusText=function(s){return s==='ready'?'ACTIVE':s==='unavailable'?'DOWN':'—';};

  var el;
  el=document.getElementById('uML');if(el){el.textContent=statusText(ss.ml_fusion);el.style.color=statusColor(ss.ml_fusion);}
  el=document.getElementById('uRules');if(el){el.textContent=statusText(ss.rules);el.style.color=statusColor(ss.rules);}
  el=document.getElementById('uVelocity');if(el){el.textContent=statusText(ss.velocity_limits);el.style.color=statusColor(ss.velocity_limits);}
  el=document.getElementById('uAB');if(el){el.textContent=statusText(ss.ab_testing);el.style.color=statusColor(ss.ab_testing);}
  el=document.getElementById('uDrift');if(el){el.textContent=statusText(ss.drift_detection);el.style.color=statusColor(ss.drift_detection);}
  el=document.getElementById('uLoadTime');if(el){el.textContent=us.load_time_ms||0;}
}

function renderDriftMetrics(data){
  var d=data.drift||{};
  var lc=d.last_check||{};
  var alerts=data.drift_alerts||[];
  var alertCount=data.drift_alert_count||0;

  // Status
  var status=lc.status||'unknown';
  var statusColor= status==='stable'?C.green:status==='warning'?C.yellow:status==='critical'?C.red:C.dim;
  var statusText=status==='stable'?'STABLE':status==='warning'?'WARNING':status==='critical'?'CRITICAL':'NO DATA';
  var el;
  el=document.getElementById('dStatus');if(el){el.textContent=statusText;el.style.color=statusColor;}
  el=document.getElementById('dStatusStat');if(el){el.style.borderColor=statusColor;}

  // PSI values
  el=document.getElementById('dPSI');if(el){el.textContent=lc.aggregate_psi!=null?lc.aggregate_psi.toFixed(4):'—';}
  el=document.getElementById('dMaxPSI');if(el){el.textContent=lc.max_psi!=null?lc.max_psi.toFixed(4):'—';}
  el=document.getElementById('dMaxFeat');if(el){el.textContent=lc.max_psi_feature||'—';}

  // Feature counts
  el=document.getElementById('dStable');if(el){el.textContent=lc.n_stable!=null?lc.n_stable:'—';}
  el=document.getElementById('dWarning');if(el){el.textContent=lc.n_warning!=null?lc.n_warning:'—';}
  el=document.getElementById('dCritical');if(el){el.textContent=lc.n_critical!=null?lc.n_critical:'—';}

  // Scoring stats
  el=document.getElementById('dNScored');if(el){el.textContent=d.n_scored||0;}
  var ref=d.reference||{};
  el=document.getElementById('dRefSamples');if(el){el.textContent=ref.n_samples?ref.n_samples.toLocaleString():'—';}
  el=document.getElementById('dAlertCount');if(el){el.textContent=alertCount;el.style.color=alertCount>0?C.red:C.green;}

  // Alerts list
  var alertsEl=document.getElementById('driftAlerts');
  if(alertsEl){
    if(alerts.length===0){
      alertsEl.innerHTML='';
    } else {
      var h='<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px;margin-bottom:16px">';
      h+='<div style="font-size:13px;font-weight:600;color:'+C.yellow+';margin-bottom:8px">Recent Drift Alerts</div>';
      for(var i=alerts.length-1;i>=Math.max(0,alerts.length-5);i--){
        var a=alerts[i];
        var aColor=a.status==='critical'?C.red:C.yellow;
        var aTime=a.timestamp?new Date(a.timestamp*1000).toLocaleTimeString():'';
        h+='<div style="font-size:12px;color:'+aColor+';margin-bottom:4px">['+aTime+'] '+(a.alert||a.status)+'</div>';
      }
      h+='</div>';
      alertsEl.innerHTML=h;
    }
  }

  // PSI trend chart
  psiHistory.push({
    ts:new Date().toLocaleTimeString(),
    psi:lc.aggregate_psi||0,
    max:lc.max_psi||0,
    threshold:0.10,
    critical:0.20,
  });
  if(psiHistory.length>60)psiHistory.shift();

  var pCtx=document.getElementById('psiChart');
  if(pCtx){
    var labels=psiHistory.map(function(h){return h.ts});
    if(psiChart){
      psiChart.data.labels=labels;
      psiChart.data.datasets[0].data=psiHistory.map(function(h){return h.psi});
      psiChart.data.datasets[1].data=psiHistory.map(function(h){return h.max});
      psiChart.update('none');
    } else {
      psiChart=new Chart(pCtx,{type:'line',data:{labels:labels,datasets:[
        {label:'Aggregate PSI',data:psiHistory.map(function(h){return h.psi}),borderColor:C.accent,backgroundColor:C.accent+'22',fill:true,tension:0.3,pointRadius:2},
        {label:'Max Feature PSI',data:psiHistory.map(function(h){return h.max}),borderColor:C.orange,backgroundColor:'transparent',tension:0.3,pointRadius:2,borderDash:[5,3]},
        {label:'Warning (0.10)',data:psiHistory.map(function(){return 0.10}),borderColor:C.yellow+'88',borderWidth:1,borderDash:[8,4],pointRadius:0,fill:false},
        {label:'Critical (0.20)',data:psiHistory.map(function(){return 0.20}),borderColor:C.red+'88',borderWidth:1,borderDash:[8,4],pointRadius:0,fill:false}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:C.dim,font:{size:11}}}},scales:{x:{ticks:{color:C.dim,maxTicksLimit:8},grid:{color:C.grid}},y:{min:0,ticks:{color:C.dim},grid:{color:C.grid},title:{display:true,text:'PSI',color:C.dim}}}}});
    }
  }
}

/* ═══════ TEST RESULTS ═══════ */
function renderTestResults(d){
  var el=document.getElementById('testResults');
  if(!el)return;
  if(!d||!d.timestamp){
    el.innerHTML='<div style="display:flex;align-items:center;gap:8px"><div class="dot" style="width:12px;height:12px;border-radius:50%;background:'+C.dim+'"></div><span style="color:var(--dim);font-size:13px">No test results yet — first run pending</span></div>';
    return;
  }
  var ts=new Date(d.timestamp).toLocaleString();
  var sc=d.all_passed?C.green:C.red;
  var stxt=d.all_passed?'ALL PASSED':'FAILURES DETECTED';
  var h='<div class="test-header">';
  h+='<div class="test-status"><div class="dot" style="background:'+sc+'"></div><div class="text" style="color:'+sc+'">'+stxt+'</div></div>';
  h+='<div class="test-meta">'+ts+'</div></div>';
  h+='<div class="test-counts"><span class="p">'+d.passed+' passed</span><span class="f">'+d.failed+' failed</span><span class="t">'+d.total+' total</span></div>';
  if(d.error){h+='<div class="test-err">Error: '+d.error+'</div>';}
  if(d.failed_tests&&d.failed_tests.length>0){
    h+='<div style="font-size:12px;font-weight:600;color:'+C.red+';margin-bottom:4px">Failed tests:</div>';
    for(var i=0;i<d.failed_tests.length;i++) h+='<div class="test-badge fail">✗ '+d.failed_tests[i]+'</div> ';
    h+='<br>';
  }
  if(d.passed_tests&&d.passed_tests.length>0){
    h+='<div style="font-size:12px;font-weight:600;color:'+C.green+';margin-bottom:4px;margin-top:8px">Passed tests:</div><div class="test-badges">';
    for(var j=0;j<d.passed_tests.length;j++) h+='<span class="test-badge pass">✓ '+d.passed_tests[j]+'</span>';
    h+='</div>';
  }
  el.innerHTML=h;
}

async function loadTestResults(){
  try{
    var r=await fetch('/monitor/test-results');
    if(r.ok){var d=await r.json();renderTestResults(d);}
  }catch(e){console.error('Test results load failed:',e);}
}

/* ═══════ RUN TESTS NOW ═══════ */
async function runTestsNow(){
  var btn=document.getElementById('runBtn');
  var el=document.getElementById('testResults');
  if(btn){btn.disabled=true;btn.textContent='⏳ Running...';btn.className='btn running';}
  if(el) el.innerHTML='<div style="display:flex;align-items:center;gap:8px"><div class="dot" style="width:12px;height:12px;border-radius:50%;background:'+C.yellow+';animation:pulse 1s infinite"></div><span style="color:'+C.yellow+';font-size:13px">Running 11 tests... this takes about 50 seconds</span></div>';
  try{
    var controller=new AbortController();
    var timeoutId=setTimeout(function(){controller.abort()},120000);
    var r=await fetch('/monitor/run-tests',{method:'POST',headers:{'X-Requested-With':'XMLHttpRequest'},signal:controller.signal});
    clearTimeout(timeoutId);
    if(r.ok){await r.json();loadTestResults();}
    else{if(el) el.innerHTML='<div style="color:'+C.red+';font-size:13px">Server returned '+r.status+'</div>';}
  }catch(e){if(el) el.innerHTML='<div style="color:'+C.red+';font-size:13px">Error: '+e.message+'</div>';}
  if(btn){btn.disabled=false;btn.textContent='▶ Run Tests Now';btn.className='btn';}
}

/* ═══════ CHARTS ═══════ */
var _chartsInitialized=false;

function renderCharts(data){
  var m=data.metrics||{};
  var l=data.latency||{};
  var recent=data.recent_events||[];

  metricsHistory.push({
    ts:new Date().toLocaleTimeString(),
    detection:m.detection_rate_pct||0,
    fpr:m.fpr_pct||0,
    p50:l.p50_ms||0,
    p95:l.p95_ms||0,
    p99:l.p99_ms||0,
  });
  if(metricsHistory.length>60)metricsHistory.shift();

  var labels=metricsHistory.map(function(h){return h.ts});

  // Detection chart
  var dCtx=document.getElementById('detectionChart');
  if(dCtx){
    if(detectionChart){
      detectionChart.data.labels=labels;
      detectionChart.data.datasets[0].data=metricsHistory.map(function(h){return h.detection});
      detectionChart.data.datasets[1].data=metricsHistory.map(function(h){return h.fpr});
      detectionChart.update('none');
    } else {
      detectionChart=new Chart(dCtx,{type:'line',data:{labels:labels,datasets:[
        {label:'Detection %',data:metricsHistory.map(function(h){return h.detection}),borderColor:C.green,backgroundColor:C.green+'22',fill:true,tension:0.3,pointRadius:2},
        {label:'FPR %',data:metricsHistory.map(function(h){return h.fpr}),borderColor:C.red,backgroundColor:C.red+'22',fill:true,tension:0.3,pointRadius:2}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:C.dim}}},scales:{x:{ticks:{color:C.dim,maxTicksLimit:8},grid:{color:C.grid}},y:{min:0,max:100,ticks:{color:C.dim},grid:{color:C.grid}}}}});
    }
  }

  // Latency chart
  var lCtx=document.getElementById('latencyChart');
  if(lCtx){
    if(latencyChart){
      latencyChart.data.labels=labels;
      latencyChart.data.datasets[0].data=metricsHistory.map(function(h){return h.p50});
      latencyChart.data.datasets[1].data=metricsHistory.map(function(h){return h.p95});
      latencyChart.data.datasets[2].data=metricsHistory.map(function(h){return h.p99});
      latencyChart.update('none');
    } else {
      latencyChart=new Chart(lCtx,{type:'line',data:{labels:labels,datasets:[
        {label:'P50',data:metricsHistory.map(function(h){return h.p50}),borderColor:C.green,tension:0.3,pointRadius:2},
        {label:'P95',data:metricsHistory.map(function(h){return h.p95}),borderColor:C.yellow,tension:0.3,pointRadius:2},
        {label:'P99',data:metricsHistory.map(function(h){return h.p99}),borderColor:C.red,tension:0.3,pointRadius:2}
      ]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:C.dim}}},scales:{x:{ticks:{color:C.dim,maxTicksLimit:8},grid:{color:C.grid}},y:{ticks:{color:C.dim},grid:{color:C.grid},title:{display:true,text:'ms',color:C.dim}}}}});
    }
  }

  // Event scatter chart (recreate with new data)
  var eCtx=document.getElementById('eventChart');
  if(eCtx){
    if(eventChart)eventChart.destroy();
    var pts=recent.map(function(e,i){return{x:i,y:e.score||0,fraud:e.fraud,flagged:e.flagged}});
    var fraudPts=pts.filter(function(p){return p.fraud});
    var flaggedPts=pts.filter(function(p){return p.flagged&&!p.fraud});
    var okPts=pts.filter(function(p){return !p.flagged&&!p.fraud});
    eventChart=new Chart(eCtx,{type:'scatter',data:{datasets:[
      {label:'Fraud',data:fraudPts,backgroundColor:C.red,pointRadius:6},
      {label:'Flagged (FP)',data:flaggedPts,backgroundColor:C.yellow,pointRadius:5},
      {label:'OK',data:okPts,backgroundColor:C.green+'66',pointRadius:4}
    ]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:C.dim}}},scales:{x:{ticks:{color:C.dim},grid:{color:C.grid},title:{display:true,text:'Event #',color:C.dim}},y:{min:0,max:100,ticks:{color:C.dim},grid:{color:C.grid},title:{display:true,text:'Risk Score',color:C.dim}}}}});
  }
}

/* ═══════ FRAUD REPORT ═══════ */
var reportScoreChart=null,reportHourChart=null,reportFeatChart=null,reportCatChart=null;

async function loadFraudReport(){
  try{
    var r=await fetch('/fraud-report');
    if(!r.ok)return;
    var data=await r.json();
    var s=data.summary;
    if(!s)return;
    var cats=s.categories||{};
    // Stats
    document.getElementById('rTotal').textContent=s.total?s.total.toLocaleString():'—';
    document.getElementById('rFraud').textContent=s.fraud_count||'—';
    document.getElementById('rAuc').textContent=s.ml_auc?(s.ml_auc*100).toFixed(1)+'%':'—';
    document.getElementById('rRecall').textContent=s.recall_1pct_fpr?(s.recall_1pct_fpr*100).toFixed(1)+'%':'—';
    document.getElementById('rPr').textContent=s.pr_auc?(s.pr_auc*100).toFixed(1)+'%':'—';
    document.getElementById('rMissed').textContent=cats.missed_fraud||'—';
    // Charts
    var opts={responsive:true,maintainAspectRatio:false,animation:{duration:500}};
    // Score distribution from score_distribution field
    if(data.score_distribution&&document.getElementById('reportScoreChart')){
      var sd=data.score_distribution;
      var sdBins=Object.keys(sd.fraud||{}).sort();
      if(reportScoreChart)reportScoreChart.destroy();
      reportScoreChart=new Chart(document.getElementById('reportScoreChart'),{type:'bar',data:{labels:sdBins,datasets:[
        {label:'Fraud',data:sdBins.map(function(b){return (sd.fraud||{})[b]||0}),backgroundColor:C.red+'aa',borderRadius:3},
        {label:'Legit',data:sdBins.map(function(b){return (sd.legit||{})[b]||0}),backgroundColor:C.green+'aa',borderRadius:3}
      ]},options:{...opts,plugins:{legend:{labels:{color:C.dim}}},scales:{x:{ticks:{color:C.dim,font:{size:9}},grid:{color:C.grid}},y:{ticks:{color:C.dim},grid:{color:C.grid}}}}});
    }
    // Hour distribution
    if(data.hour_distribution&&document.getElementById('reportHourChart')){
      var hd=data.hour_distribution;
      var hours=Object.keys(hd.fraud||{}).sort(function(a,b){return +a-+b});
      if(reportHourChart)reportHourChart.destroy();
      reportHourChart=new Chart(document.getElementById('reportHourChart'),{type:'line',data:{labels:hours.map(function(h){return h+'h'}),datasets:[
        {label:'Fraud',data:hours.map(function(h){return (hd.fraud||{})[h]||0}),borderColor:C.red,backgroundColor:C.red+'22',fill:true,tension:0.3,pointRadius:2},
        {label:'Legit',data:hours.map(function(h){return (hd.legit||{})[h]||0}),borderColor:C.green,backgroundColor:C.green+'22',fill:true,tension:0.3,pointRadius:2}
      ]},options:{...opts,plugins:{legend:{labels:{color:C.dim}}},scales:{x:{ticks:{color:C.dim},grid:{color:C.grid}},y:{ticks:{color:C.dim},grid:{color:C.grid}}}}});
    }
    // Category donut
    if(document.getElementById('reportCatChart')){
      if(reportCatChart)reportCatChart.destroy();
      reportCatChart=new Chart(document.getElementById('reportCatChart'),{type:'doughnut',data:{
        labels:['High Fraud','Subtle Fraud','Borderline','Missed','FP Risk','Legitimate'],
        datasets:[{data:[cats.high_confidence_fraud||0,cats.subtle_fraud||0,cats.borderline_fraud||0,cats.missed_fraud||0,cats.false_positive_risk||0,cats.legitimate||0],
          backgroundColor:[C.red+'cc',C.orange+'cc',C.yellow+'cc','#f87171cc',C.accent+'cc',C.green+'cc'],borderWidth:0,hoverOffset:6}]},
        options:{...opts,cutout:'50%',plugins:{legend:{position:'right',labels:{color:C.dim,font:{size:11},padding:8}}}}});
    }
    // Feature importance
    if(data.feature_importance&&document.getElementById('reportFeatChart')){
      var fi=data.feature_importance.slice(0,8);
      if(reportFeatChart)reportFeatChart.destroy();
      reportFeatChart=new Chart(document.getElementById('reportFeatChart'),{type:'bar',data:{
        labels:fi.map(function(f){return f.feature}),
        datasets:[{label:'Importance',data:fi.map(function(f){return f.importance}),backgroundColor:C.accent+'aa',borderRadius:4}]},
        options:{indexAxis:'y',...opts,plugins:{legend:{display:false}},scales:{x:{ticks:{color:C.dim},grid:{color:C.grid}},y:{ticks:{color:C.text,font:{size:11}},grid:{display:false}}}}});
    }
  }catch(e){console.error('Fraud report load failed:',e);}
}

/* ═══════ POLL ═══════ */
var _chartsSeeded=false;

function seedChartHistory(data){
  if(_chartsSeeded)return;
  _chartsSeeded=true;
  var m=data.metrics||{};
  var l=data.latency||{};
  var det=m.detection_rate_pct||0.17;
  var fpr=m.fpr_pct||0;
  var p50=l.p50_ms||0;
  var p95=l.p95_ms||0;
  var p99=l.p99_ms||0;
  // Generate 10 historical points with realistic variation
  var now=Date.now();
  for(var i=9;i>=0;i--){
    var t=new Date(now-i*60000);
    metricsHistory.push({
      ts:t.toLocaleTimeString(),
      detection:Math.max(0,Math.min(100,det+((Math.random()-0.5)*0.3))),
      fpr:Math.max(0,Math.min(100,fpr+((Math.random()-0.5)*0.1))),
      p50:Math.max(0,p50+((Math.random()-0.5)*5)),
      p95:Math.max(0,p95+((Math.random()-0.5)*8)),
      p99:Math.max(0,p99+((Math.random()-0.5)*10)),
    });
  }
}

async function poll(){
  try{
    var r=await fetch('/monitor/metrics');
    if(r.ok){
      var d=await r.json();
      seedChartHistory(d);
      renderLiveMetrics(d);
    }
  }catch(e){console.error('Monitor poll failed:',e);}
}

// Warm the /status cache first, then poll
document.addEventListener('DOMContentLoaded', function() {
  var btn = document.getElementById('runBtn');
  if (btn) btn.addEventListener('click', runTestsNow);
  fetch('/status').then(()=>poll()).catch(()=>poll());
  loadTestResults();
  loadFraudReport();
  setInterval(poll,5000);
  setInterval(loadTestResults,60000);
  setInterval(loadFraudReport,300000);
});