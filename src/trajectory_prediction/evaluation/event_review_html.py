"""Self-contained offline GT viewer; editable notes never become event truth."""
import json

from trajectory_prediction.evaluation.event_review_data import (
    COLUMNS,
    LOCKED,
    MAX_CELL_CODEPOINTS,
    REVIEW_WHITESPACE,
)


def script_source():
    header = 'const COLUMNS='+json.dumps(COLUMNS)+'; const LOCKED='+json.dumps(LOCKED)+';\n'
    header += 'const REVIEW_WHITESPACE='+json.dumps(REVIEW_WHITESPACE)+';\n'
    header += 'const MAX_CELL_CODEPOINTS='+str(MAX_CELL_CODEPOINTS)+';\n'
    return header+r'''
function hasText(value){return Array.from(value).some(c=>!REVIEW_WHITESPACE.includes(c));}
function plotBounds(points,equalAspect){
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
 let x0=Math.min(0,...xs),x1=Math.max(1,...xs),y0=Math.min(0,...ys),y1=Math.max(1,...ys);
 if(equalAspect){const unit=Math.max((x1-x0)/540,(y1-y0)/185),cx=(x0+x1)/2,cy=(y0+y1)/2;
  x0=cx-unit*270;x1=cx+unit*270;y0=cy-unit*92.5;y1=cy+unit*92.5;}
 return {x0,x1,y0,y1};
}
function csvText(rows){
 const esc=v=>/[",\r\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v;
 return [COLUMNS,...rows.map(r=>COLUMNS.map(k=>r[k]))].map(r=>r.map(esc).join(',')).join('\r\n')+'\r\n';
}
function parseCsv(text){
 text=text.replace(/^\ufeff/,'');let rows=[],row=[],cell='',quoted=false,closed=false;
 const end=()=>{row.push(cell);if(row.length>1||row[0]!=='')rows.push(row);row=[];cell='';closed=false;};
 for(let i=0;i<text.length;i++){
  let c=text[i];
  if(quoted){if(c==='"'){if(text[i+1]==='"'){cell+='"';i++;}else{quoted=false;closed=true;}}else cell+=c;continue;}
  if(closed&&c!==','&&c!=='\r'&&c!=='\n')throw Error('引号后有非法字符');
  if(c==='"'&&cell===''&&!closed){quoted=true;continue;}
  if(c===','){row.push(cell);cell='';closed=false;}
  else if(c==='\r'||c==='\n'){if(c==='\r'&&text[i+1]==='\n')i++;end();}
  else cell+=c;
 }
 if(quoted)throw Error('CSV引号未闭合');
 if(row.length||cell!==''||closed)end();return rows;
}
function validateRows(matrix,baseline){
 if(!matrix.length||JSON.stringify(matrix[0])!==JSON.stringify(COLUMNS))throw Error('工作表列不匹配');
 const byId=new Map(baseline.map(r=>[r.task_id,r])),seen=new Set(),out=[];
 for(const cells of matrix.slice(1)){
  if(cells.length!==COLUMNS.length)throw Error('工作表列数错误');
  const r=Object.fromEntries(COLUMNS.map((k,i)=>[k,cells[i]])),b=byId.get(r.task_id);
  if(!b||seen.has(r.task_id)||LOCKED.some(k=>r[k]!==b[k]))throw Error('任务来源或锁定字段不匹配');
  if(!['pending','in_progress','reviewed'].includes(r.review_status))throw Error('未知阅读状态');
  if(Object.values(r).some(v=>Array.from(v).length>MAX_CELL_CODEPOINTS))throw Error('单元格超过8192个Unicode字符');
  if(r.review_status==='reviewed'&&(!hasText(r.reviewer)||!hasText(r.notes)))throw Error('已阅读须填写复核人和备注');
  seen.add(r.task_id);out.push(r);
 }
 if(seen.size!==baseline.length)throw Error('工作表缺少任务');return out;
}
function boot(){
 const data=JSON.parse(document.getElementById('review-data').textContent),$=id=>document.getElementById(id);
 const blocks=new Map(data.blocks.map(b=>[b.block_id,b])),events=new Map(data.events.map(e=>[e.event_id,e]));
 const diagnostics=new Map(data.diagnostics.map(d=>[d.diagnostic_id,d]));
 let working=new Map(data.queue.map(r=>[r.task_id,{...r}])),current=null;
 const el=(tag,attrs={},text=null)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==null)e.textContent=text;return e;};
 function save(){if(current&&working.has(current)){const r=working.get(current);r.review_status=$('status').value;r.reviewer=$('reviewer').value;r.notes=$('notes').value;}}
 function message(text){$('message').textContent=text;}
 function plot(id,points,xlabel,ylabel,boundaries=[],equalAspect=false){
  const svg=$(id);svg.replaceChildren();const {x0,x1,y0,y1}=plotBounds(points,equalAspect);
  const X=x=>55+540*(x-x0)/(x1-x0),Y=y=>220-185*(y-y0)/(y1-y0);
  svg.append(el('path',{d:'M55 20V220H610',fill:'none',stroke:'#64748b'}));
  svg.append(el('polyline',{points:points.map(p=>X(p[0])+','+Y(p[1])).join(' '),fill:'none',stroke:'#2563eb','stroke-width':2}));
  for(const p of points)svg.append(el('circle',{cx:X(p[0]),cy:Y(p[1]),r:2.2,fill:'#2563eb'}));
  for(const b of boundaries){if(b.x<x0||b.x>x1)continue;svg.append(el('line',{x1:X(b.x),x2:X(b.x),y1:20,y2:220,stroke:b.color,'stroke-dasharray':'5 4'}));const right=X(b.x)>500;svg.append(el('text',{x:X(b.x)+(right?-4:3),y:30+b.level*15,fill:b.color,'font-size':11,'text-anchor':right?'end':'start'},b.label));}
  svg.append(el('text',{x:290,y:250,'font-size':12},xlabel));svg.append(el('text',{x:8,y:15,'font-size':12},ylabel));
  svg.append(el('text',{x:56,y:235,'font-size':10},x0.toFixed(2)));svg.append(el('text',{x:575,y:235,'font-size':10},x1.toFixed(2)));
  svg.append(el('text',{x:5,y:45,'font-size':10},y1.toFixed(2)));svg.append(el('text',{x:5,y:217,'font-size':10},y0.toFixed(2)));
 }
 function render(id){
  save();current=id;const task=working.get(id);if(!task)return;
  for(const k of ['status','reviewer','notes'])$(k).disabled=false;
  const block=blocks.get(task.block_id),event=events.get(task.event_id);
  $('status').value=task.review_status;$('reviewer').value=task.reviewer;$('notes').value=task.notes;
  $('heading').textContent=task.task_kind==='event_check'?'候选事件核对':task.task_kind==='block_scan'?'完整分析块扫描':'不可评价区间检查';
  $('identity').textContent=task.role+' · '+task.instance+' · '+(event?event.event_type:task.task_kind);
  const context=block?{block_id:block.block_id,category:block.category,scene:block.scene,log:block.log,
   first_annotation:block.first_annotation,last_annotation:block.last_annotation,
   event_records:block.event_ids.map(k=>events.get(k)),diagnostics:block.diagnostic_ids.map(k=>diagnostics.get(k))}:diagnostics.get(task.diagnostic_id);
  $('context').textContent=JSON.stringify(event?{selected_event:event,context}:context,null,2);
  const links=event?data.window_links.filter(r=>r.event_id===event.event_id):[];
  $('coverage').textContent=event?`已有窗口关联 ${links.length} 条，其中pilot ${links.filter(r=>r.phase3a_pilot_member).length} 条。关联不代表新增预测。`:'本任务包含完整GT上下文；无候选不等于人工阴性。';
  if(block){const xy=block.xy,base=xy[0],t0=block.time_us[0];
   plot('trajectory',xy.map(p=>[p[0]-base[0],p[1]-base[1]]),'相对首点 ΔX (m)','ΔY (m)',[],true);
   const speed=[];for(let i=0;i<block.speed_mps.length;i++){speed.push([(block.time_us[i]-t0)/1e6,block.speed_mps[i]],[(block.time_us[i+1]-t0)/1e6,block.speed_mps[i]]);}
   const boundary=event?[{x:(event.onset_us-t0)/1e6,color:'#dc2626',label:'onset',level:0},
    {x:(event.confirmation_us-t0)/1e6,color:'#047857',label:'confirmation',level:1},
    {x:(event.end_us-t0)/1e6,color:'#7c3aed',label:event.right_censored?'end（右删失）':'end',level:2}]:[];
   if(speed.length)plot('speed',speed,'相对分析块起点 t (s)','原始区间速度 (m/s)',boundary);
   else{$('speed').replaceChildren(el('text',{x:130,y:125,'font-size':18,fill:'#64748b'},'单点：无有效速度区间'));}
   $('source').textContent=`${xy.length} 个原始GT点；世界坐标首点 (${base[0].toFixed(3)}, ${base[1].toFixed(3)}) m。${speed.length?'':'单点，无有效速度区间。'}`;
  }else{$('trajectory').replaceChildren();$('speed').replaceChildren();$('source').textContent='此诊断位于分析块边界，未虚构轨迹上下文。';}
  document.body.dataset.reviewReady='true';document.body.dataset.selectedTask=id;document.body.dataset.noMatches='false';
 }
 function list(){save();const role=$('role-filter').value,kind=$('kind-filter').value,q=$('search').value.toLowerCase();
  const rows=data.queue.filter(r=>(!role||r.role===role)&&(!kind||r.task_kind===kind)&&[r.task_id,r.instance,r.event_id,r.block_id].join(' ').toLowerCase().includes(q));
  $('tasks').replaceChildren();for(const r of rows){const o=document.createElement('option');o.value=r.task_id;o.textContent=(r.task_kind==='event_check'?'事件':'块扫描')+' · '+r.instance.slice(0,10)+(r.event_id?' · '+events.get(r.event_id).event_type:'');$('tasks').append(o);}
  $('count').textContent=`筛选 ${rows.length} / ${data.queue.length} 条任务；不代表独立事件数量`;
  const wanted=rows.find(r=>r.task_id===current)||rows[0];$('tasks').disabled=!wanted;
  if(wanted){$('tasks').value=wanted.task_id;render(wanted.task_id);}
  else{current=null;for(const k of ['status','reviewer','notes']){$(k).disabled=true;$(k).value=k==='status'?'pending':'';}
   $('heading').textContent='无匹配任务';for(const k of ['identity','source','coverage','context'])$(k).textContent='';
   $('trajectory').replaceChildren();$('speed').replaceChildren();document.body.dataset.noMatches='true';document.body.dataset.selectedTask='';}
 }
 $('tasks').addEventListener('change',()=>render($('tasks').value));
 for(const id of ['role-filter','kind-filter','search'])$(id).addEventListener('input',list);
 for(const id of ['status','reviewer','notes'])$(id).addEventListener('input',save);
 $('export-working').addEventListener('click',()=>{try{save();const rows=[...working.values()];validateRows(parseCsv(csvText(rows)),data.queue);
  const url=URL.createObjectURL(new Blob(['\ufeff'+csvText(rows)],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='review_working.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message('已导出阅读记录；未导入或确认任何事件GT。');}catch(e){message(e.message);}});
 $('import-working').addEventListener('change',async ev=>{try{const file=ev.target.files[0];if(!file)return;
  const rows=validateRows(parseCsv(await file.text()),data.queue);working=new Map(rows.map(r=>[r.task_id,r]));current=null;list();message('已加载同源阅读记录；静态GT与规则候选未改变。');}catch(e){message('导入拒绝：'+e.message);}finally{ev.target.value='';}});
 $('totals').textContent=`${data.summary.block_scan_tasks} 个分析块 · ${data.summary.event_check_tasks} 条候选记录（含复合配对） · ${data.summary.diagnostic_rows} 条诊断`;
 try{current=decodeURIComponent(location.hash.slice(1));}catch(e){current=null;}list();
}
if(typeof module!=='undefined'&&module.exports)module.exports={csvText,parseCsv,validateRows,plotBounds};else boot();
'''


def render_html(data):
    payload = json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    payload = payload.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>轨迹事件人工复核资料</title><style>
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;margin:0;color:#1e293b;background:#f1f5f9}
header{padding:18px 24px;background:#0f172a;color:white}header h1{font-size:21px;margin:0 0 8px}header p{margin:5px 0;font-size:13px}
main{display:grid;grid-template-columns:320px 1fr;gap:16px;padding:16px}aside,section{background:white;border-radius:8px;padding:16px}
label{display:block;font-size:13px;margin:10px 0 4px}input,select,textarea,button{font:inherit;padding:8px;border:1px solid #cbd5e1;border-radius:4px;max-width:100%}
aside select,aside input,#tasks{width:100%}#tasks{height:400px;font-size:12px}#count,#source,#coverage,#identity{font-size:12px;color:#475569;overflow-wrap:anywhere}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:10px}svg{width:100%;border:1px solid #e2e8f0}h2{font-size:19px;margin:0 0 8px}
pre{font-size:11px;white-space:pre-wrap;max-height:260px;overflow:auto;background:#f8fafc;padding:12px}textarea{width:100%;min-height:80px}
button{background:#1d4ed8;color:white;cursor:pointer;margin-top:10px}#message{color:#b45309;min-height:24px;font-size:13px}
@media(max-width:950px){main{grid-template-columns:1fr}.charts{grid-template-columns:1fr}}
</style></head><body><header><h1>轨迹事件人工复核资料</h1><p id="totals"></p>
<p>仅展示GT、规则候选与既有窗口覆盖。阅读状态不等于人工事件标签；未触发规则不等于阴性。</p></header>
<main><aside><label for="role-filter">数据角色</label><select id="role-filter"><option value="">全部角色</option><option>independent_fit</option><option>independent_select</option></select>
<label for="kind-filter">任务范围</label><select id="kind-filter"><option value="">全部任务</option><option value="block_scan">完整分析块扫描</option><option value="event_check">候选事件核对</option><option value="diagnostic_check">无块诊断</option></select>
<label for="search">搜索对象或来源ID</label><input id="search" placeholder="instance / block / event ID"><p id="count"></p><select id="tasks" size="16" aria-label="复核任务"></select>
<label for="import-working">继续已有工作表（CSV）</label><input type="file" id="import-working" accept=".csv"><button id="export-working">下载本次阅读记录 CSV</button><p id="message" role="status"></p></aside>
<section><h2 id="heading"></h2><p id="identity"></p><div class="charts"><svg id="trajectory" viewBox="0 0 650 260" aria-label="GT轨迹"></svg><svg id="speed" viewBox="0 0 650 260" aria-label="真实时间速度与事件边界"></svg></div>
<p id="source"></p><p id="coverage"></p><label for="status">阅读进度</label><select id="status"><option>pending</option><option>in_progress</option><option>reviewed</option></select>
<label for="reviewer">复核人</label><input id="reviewer"><label for="notes">观察与疑点（reviewed须填写）</label><textarea id="notes"></textarea>
<details><summary>原始事件与诊断证据</summary><pre id="context"></pre></details><p><a href="coverage.csv">查看继承的覆盖统计 CSV</a></p></section></main>
<script type="application/json" id="review-data">'''+payload+'''</script><script>'''+script_source()+'''</script></body></html>'''
