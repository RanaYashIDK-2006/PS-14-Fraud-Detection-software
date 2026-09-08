(function(){
'use strict';

/* ═══════ COUNTER ANIMATION ═══════ */
function animateCounter(el){
  var target=parseFloat(el.getAttribute('data-target'));
  var suffix=el.getAttribute('data-suffix')||'';
  var dec=parseInt(el.getAttribute('data-dec')||'0');
  if(isNaN(target))return;
  var obj={val:0};
  anime({targets:obj,val:target,duration:1500,delay:400,easing:'easeOutExpo',
    update:function(){el.textContent=dec>0?obj.val.toFixed(dec)+suffix:Math.round(obj.val)+suffix}
  });
}

/* ═══════ SCROLL PROGRESS ═══════ */
function initScrollProgress(){
  var bar=document.getElementById('scroll-bar');
  if(!bar)return;
  window.addEventListener('scroll',function(){
    var scrollTop=window.scrollY;
    var docHeight=document.documentElement.scrollHeight-window.innerHeight;
    var pct=docHeight>0?(scrollTop/docHeight)*100:0;
    bar.style.width=pct+'%';
  },{passive:true});
}

/* ═══════ SCROLL ANIMATIONS ═══════ */
function initScrollAnimations(){
  var els=document.querySelectorAll('.anim');
  var obs=new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(entry.isIntersecting){
        entry.target.classList.add('vis');
        // Animate counters within
        entry.target.querySelectorAll('.counter').forEach(function(c){animateCounter(c)});
        obs.unobserve(entry.target);
      }
    });
  },{threshold:0.1});
  els.forEach(function(el){obs.observe(el)});
}

/* ═══════ HERO COUNTERS ═══════ */
function initHeroCounters(){
  document.querySelectorAll('.hero-stat .counter').forEach(function(el){animateCounter(el)});
}

/* ═══════ LIVE DEMO CANVAS ═══════ */
function initLiveDemo(){
  var canvas=document.getElementById('liveDemo'); if(!canvas)return;
  var ctx=canvas.getContext('2d');
  var W=canvas.width=900,H=canvas.height=320;
  var particles=[];
  var colors={low:'#10B981',mid:'#F59E0B',high:'#EF4444'};
  var labels={low:'Low Risk ✓',mid:'Medium Risk ?',high:'High Risk ⚠'};

  function spawn(){
    var r=Math.random();
    var t=r<0.72?'low':(r<0.93?'mid':'high');
    particles.push({x:-10,y:50+Math.random()*(H-100),vx:1.8+Math.random()*2.2,vy:(Math.random()-0.5)*0.4,type:t,size:3+Math.random()*7,alpha:0.7+Math.random()*0.3,trail:[],labeled:false});
  }

  function draw(){
    ctx.fillStyle='rgba(11,15,26,0.12)';
    ctx.fillRect(0,0,W,H);
    // Zones
    var zones=[{x:0.15,w:0.15,c:'rgba(59,130,246,0.06)',l:'Privacy Layer'},{x:0.35,w:0.15,c:'rgba(139,92,246,0.06)',l:'ML Models'},{x:0.55,w:0.15,c:'rgba(6,182,212,0.06)',l:'Rules Engine'},{x:0.75,w:0.12,c:'rgba(244,63,94,0.06)',l:'Decision'}];
    zones.forEach(function(z){
      ctx.fillStyle=z.c;ctx.fillRect(W*z.x,0,W*z.w,H);
      ctx.font='10px Inter,system-ui';ctx.textAlign='center';ctx.fillStyle='#64748b';ctx.fillText(z.l,W*(z.x+z.w/2),18);
    });
    ctx.strokeStyle='rgba(30,41,59,0.3)';ctx.setLineDash([3,3]);
    [0.15,0.3,0.35,0.5,0.55,0.7,0.75,0.87].forEach(function(x){ctx.beginPath();ctx.moveTo(W*x,0);ctx.lineTo(W*x,H);ctx.stroke()});
    ctx.setLineDash([]);
    // Particles
    for(var i=particles.length-1;i>=0;i--){
      var p=particles[i];p.x+=p.vx;p.y+=p.vy;p.trail.push({x:p.x,y:p.y});if(p.trail.length>18)p.trail.shift();
      if(p.trail.length>1){ctx.beginPath();ctx.moveTo(p.trail[0].x,p.trail[0].y);for(var t=1;t<p.trail.length;t++)ctx.lineTo(p.trail[t].x,p.trail[t].y);ctx.strokeStyle=colors[p.type];ctx.globalAlpha=0.15;ctx.lineWidth=1;ctx.stroke();ctx.globalAlpha=1;}
      ctx.beginPath();ctx.arc(p.x,p.y,p.size,0,Math.PI*2);ctx.fillStyle=colors[p.type];ctx.globalAlpha=p.alpha;ctx.fill();ctx.globalAlpha=1;
      if(p.x>W*0.83&&!p.labeled){p.labeled=true;ctx.font='bold 9px Inter,system-ui';ctx.textAlign='left';ctx.fillStyle=colors[p.type];ctx.fillText(labels[p.type],p.x+8,p.y+3);}
      if(p.x>W+20)particles.splice(i,1);
    }
    if(Math.random()<0.12)spawn();
    requestAnimationFrame(draw);
  }
  // Seed initial particles
  for(var i=0;i<10;i++){var r=Math.random();particles.push({x:Math.random()*W*0.7,y:50+Math.random()*(H-100),vx:1.8+Math.random()*2.2,vy:(Math.random()-0.5)*0.4,type:r<0.72?'low':(r<0.93?'mid':'high'),size:3+Math.random()*7,alpha:0.7+Math.random()*0.3,trail:[],labeled:false});}
  draw();
}

/* ═══════ CHARTS ═══════ */
var CC={green:'#10B981',red:'#EF4444',ocean:'#3B82F6',yellow:'#F59E0B',purple:'#8B5CF6',cyan:'#06B6D4',muted:'#64748b',border:'#1e293b',text:'#f1f5f9',surface:'#111827'};

function initCharts(){
  if(typeof Chart==='undefined')return;

  // ROC-AUC across certified real-data segments (deployed Altman-NATIVE)
  var rocEl=document.getElementById('rocChart');
  if(rocEl){
    var ds=['Night','Online','Chip','High\nAmount'];
    var roc=[0.9941,0.9886,0.9844,0.9036];
    new Chart(rocEl,{type:'bar',data:{labels:ds,datasets:[{label:'ROC-AUC',data:roc,backgroundColor:roc.map(function(v){return v>=0.98?CC.green+'cc':v>=0.95?CC.ocean+'cc':CC.yellow+'cc'}),borderRadius:8,borderSkipped:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:CC.muted,font:{size:11}},grid:{display:false}},y:{ticks:{color:CC.muted},grid:{color:CC.border+'40'},min:0.85,max:1.01}},animation:{duration:1200,easing:'easeOutCubic'}}});
  }

  // Recall at locked threshold vs 1%FPR reference (certified, real data)
  var r1El=document.getElementById('recallChart');
  if(r1El){
    var rds=['Recall @\nlocked thr','Recall @\n1% FPR','Precision @\nlocked thr'];
    var r1=[0.9967,0.6208,0.1811];
    new Chart(r1El,{type:'bar',data:{labels:rds,datasets:[{label:'Rate',data:r1,backgroundColor:r1.map(function(v){return v>=0.8?CC.green+'cc':v>=0.4?CC.ocean+'cc':CC.yellow+'cc'}),borderRadius:8,borderSkipped:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:CC.muted,font:{size:11}},grid:{display:false}},y:{ticks:{color:CC.muted},grid:{color:CC.border+'40'},min:0,max:1.0}},animation:{duration:1200,easing:'easeOutCubic'}}});
  }
}

/* ═══════ SIDEBAR ACTIVE STATE ═══════ */
function initSidebarNav(){
  var navItems=document.querySelectorAll('.nav-item');
  navItems.forEach(function(item){
    item.addEventListener('click',function(){
      navItems.forEach(function(n){n.classList.remove('active')});
      item.classList.add('active');
    });
  });
}

/* ═══════ ENTRANCE ANIMATIONS ═══════ */
function initEntrance(){
  if(typeof anime==='undefined')return;
  // Stagger stat cards
  var stats=document.querySelectorAll('.stat-card');
  stats.forEach(function(s,i){
    anime({targets:s,opacity:[0,1],translateY:[20,0],duration:500,delay:100+i*80,easing:'easeOutCubic'});
  });
}

/* ═══════ HOVER MICRO-INTERACTIONS ═══════ */
function initHover(){
  if(typeof anime==='undefined')return;
  document.querySelectorAll('.stat-card,.feature-card,.risk-box').forEach(function(c){
    c.addEventListener('mouseenter',function(){anime({targets:c,scale:[1,1.02],duration:200,easing:'easeOutCubic'})});
    c.addEventListener('mouseleave',function(){anime({targets:c,scale:[1.02,1],duration:200,easing:'easeOutCubic'})});
  });
}

/* ═══════ TABLE FILTERING ═══════ */
window.filterTable=function(type, chipEl){
  // Update active chip
  document.querySelectorAll('.chip').forEach(function(c){c.classList.remove('active')});
  if(chipEl)chipEl.classList.add('active');
  
  var rows=document.querySelectorAll('#benchmarkTable tbody tr');
  rows.forEach(function(row){
    if(type==='all'){
      row.style.display='';
    } else if(type==='real'||type==='synthetic'){
      row.style.display=row.getAttribute('data-type')===type?'':'none';
    } else if(type==='excellent'||type==='good'||type==='legacy'){
      row.style.display=row.getAttribute('data-status')===type?'':'none';
    }
  });
};

/* ═══════ TABLE SORTING ═══════ */
var sortDir={};
window.sortTable=function(col){
  var table=document.getElementById('benchmarkTable');
  var tbody=table.querySelector('tbody');
  var rows=Array.from(tbody.querySelectorAll('tr'));
  sortDir[col]=!sortDir[col];
  var dir=sortDir[col]?1:-1;
  
  rows.sort(function(a,b){
    var aVal=a.cells[col].textContent.trim();
    var bVal=b.cells[col].textContent.trim();
    // Try numeric sort
    var aNum=parseFloat(aVal.replace(/[^0-9.]/g,''));
    var bNum=parseFloat(bVal.replace(/[^0-9.]/g,''));
    if(!isNaN(aNum)&&!isNaN(bNum))return (aNum-bNum)*dir;
    return aVal.localeCompare(bVal)*dir;
  });
  rows.forEach(function(r){tbody.appendChild(r)});
};

/* ═══════ SEARCH FUNCTIONALITY ═══════ */
function initSearch(){
  var input=document.getElementById('globalSearch');
  if(!input)return;
  input.addEventListener('input',function(e){
    var q=e.target.value.toLowerCase();
    // Filter all tables on page
    document.querySelectorAll('table tbody').forEach(function(tbody){
      tbody.querySelectorAll('tr').forEach(function(row){
        var text=row.textContent.toLowerCase();
        row.style.display=text.includes(q)?'':'none';
      });
    });
  });
  // Keyboard shortcut Cmd/Ctrl+K
  document.addEventListener('keydown',function(e){
    if((e.metaKey||e.ctrlKey)&&e.key==='k'){
      e.preventDefault();
      input.focus();
    }
  });
}

/* ═══════ SIDEBAR OVERLAY ═══════ */
function initSidebarOverlay(){
  var sidebar=document.querySelector('.sidebar');
  if(!sidebar)return;
  // Close sidebar when clicking outside on mobile
  document.addEventListener('click',function(e){
    if(sidebar.classList.contains('open')&&!sidebar.contains(e.target)&&!e.target.closest('.menu-toggle')){
      sidebar.classList.remove('open');
    }
  });
}

/* ═══════ INIT ═══════ */
document.addEventListener('DOMContentLoaded',function(){
  initScrollProgress();
  initScrollAnimations();
  initLiveDemo();
  initCharts();
  initSidebarNav();
  initEntrance();
  initHover();
  initSearch();
  initSidebarOverlay();
});

})();
