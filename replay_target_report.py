"""Offline paired-target report, separate from the archived first replay."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from replay_core import records


def build_report(run, output):
    manifest = json.loads((run / "run_manifest.json").read_text())
    comparison = pd.read_csv(run / "forecast_actual_comparison.csv")
    summary = pd.read_csv(run / "model_summary.csv")
    fields = ["decision_date", "target_end", "symbol", "model_name", "target_variant",
              "predicted_return", "actual_return_raw", "actual_return_model_target",
              "prediction_error_raw", "direction_hit", "evaluation_status", "missing_reason"]
    payload = {"manifest": manifest, "rows": records(comparison[fields]), "summary": records(summary)}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    report = run / "report"
    report.mkdir(exist_ok=True)
    (report / "index.html").write_text(HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", data))

    models = ["enet_all", "enet_market", "extra_all", "extra_market"]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for variant, offset, color, label in [("legacy_clipped", -.18, "#5578b8", "Clipped target"),
                                          ("raw", .18, "#e39b4d", "Raw target")]:
        values = summary[summary.target_variant.eq(variant)].set_index("model_name").loc[models]
        axes[0].bar(x + offset, values.mae_raw, .36, color=color, label=label)
        axes[1].bar(x + offset, values.prediction_abs_median, .36, color=color, label=label)
    axes[0].axhline(summary.zero_prediction_mae_raw.iloc[0], color="#444", ls="--", label="Always predict zero")
    axes[0].set_title("All evaluated assets: MAE versus raw realized return")
    axes[1].set_title("Median absolute forecast: did forecasts get larger?")
    for ax in axes:
        ax.set_xticks(x, models, rotation=15)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Same reviewed data and features; only training targets differ", fontsize=12)
    fig.tight_layout()
    fig.savefig(report / "target_comparison.png", dpi=170)
    plt.close(fig)
    stable = output / "report"
    stable.mkdir(exist_ok=True)
    for file in report.iterdir():
        shutil.copy2(file, stable / file.name)


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练目标对照 · Ansel Crypto Rank</title><script>__PLOTLY__</script><style>
:root{color-scheme:dark;font:14px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#e7eefc;background:#0b101b}*{box-sizing:border-box}body{margin:0}main{max-width:1480px;margin:auto;padding:30px}h1{font-size:30px;margin:10px 0}h2{font-size:18px;margin:0 0 10px}.muted,small{color:#a7b6ce}a{color:#87b7ff}.panel{background:#141e30;border:1px solid #2b3952;border-radius:12px;padding:20px;margin-top:18px}.notice{background:#30271b;color:#f2d9af;border:1px solid #705432;padding:16px;border-radius:10px}.filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end}label{display:flex;flex-direction:column;gap:5px}select,input,button{font:inherit;background:#0d1728;color:#e7eefc;border:1px solid #42536d;border-radius:6px;padding:8px}button{cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart{height:380px}.scroll{overflow:auto;max-height:520px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:10px;border-bottom:1px solid #2b3952;text-align:right}td:first-child,th:first-child{text-align:left}thead{position:sticky;top:0;background:#18283f}code{color:#bdd6ff}.count{font-size:18px}details{margin-top:12px}@media(max-width:850px){main{padding:15px}.grid{grid-template-columns:1fr}h1{font-size:24px}.chart{height:330px}}
</style></head><body><main>
<small>ANSEL · PAIRED TARGET EXPERIMENT</small><h1>取消收益率截断，预测会改善吗？</h1>
<p class="muted" id="period"></p><p><a href="/historical-replay">原始归档报告</a> · <a href="/">项目首页</a></p>
<p class="notice">两版使用同一份经过修正与隔离的数据、同样的训练样本和输入特征。只改变四个模型学习的收益目标。这里的“原目标”是用本次共同数据重跑的基准，不等于此前未经核验的归档结果。未核实区间不补零；数据仍有来源限制，本页是受限样本研究对照。</p>
<section class="panel"><div class="filters"><label>币种<select id="coin"><option value="">全部币种</option></select></label><label>模型<select id="model"><option value="">全部四个模型</option></select></label><label>决策日从<input id="from" type="date"></label><label>至<input id="to" type="date"></label><button id="reset">重置</button><button id="export">下载当前对照 CSV</button><button id="manifest">下载运行记录</button></div><p id="coverage" class="count"></p><p class="muted">同一条币种周分别由两版预测；两版条数不能作为独立样本翻倍计算。未结束周不计入指标。</p></section>
<section class="panel"><h2>先看同一批实际收益上的误差</h2><p class="muted">数值为百分点；差值 = 取消截断版 − 原目标版，负值表示误差减少。零预测基准每周都预测收益为 0。</p><div class="scroll"><table><thead><tr><th>模型</th><th>每版有效条数</th><th>原目标 MAE</th><th>取消截断 MAE</th><th>差值</th><th>零预测 MAE</th><th>原目标方向准确率</th><th>取消截断方向准确率</th></tr></thead><tbody id="summary"></tbody></table></div><p id="finding"></p></section>
<div class="grid"><section class="panel"><h2>平均绝对误差对照</h2><div id="maechart" class="chart"></div></section><section class="panel"><h2>预测幅度分布</h2><p class="muted">预测绝对值的中位数；变大并不自动意味着更准。</p><div id="amplitude" class="chart"></div></section></div>
<section class="panel"><h2>逐周预测与实际</h2><p class="muted">选择全部币种时展示每周可评价共同样本的均值；虚线为取消截断版。明细保留每个币种的原始结果。</p><div id="timeline" class="chart"></div></section>
<section class="panel"><h2>同一训练目标口径下的附加检查</h2><p class="muted">两版也同时对照截断目标。真实收益误差和截断目标误差分别列出，避免把口径变化当作预测改善。</p><div class="scroll"><table><thead><tr><th>模型 / 版本</th><th>真实收益 MAE（百分点）</th><th>截断目标 MAE（百分点）</th><th>真实收益 RMSE（百分点）</th><th>预测落在 ±2%</th><th>平均周排名相关系数</th></tr></thead><tbody id="details"></tbody></table></div></section>
<section class="panel"><h2>逐币种预测记录</h2><p id="tablecount" class="muted"></p><div class="scroll"><table><thead><tr><th>决策日</th><th>目标周结束</th><th>币种</th><th>模型</th><th>训练目标</th><th>预测收益</th><th>实际收益</th><th>误差（百分点）</th><th>状态</th></tr></thead><tbody id="records"></tbody></table></div><button id="prev">上一页</button> <span id="page"></span> <button id="next">下一页</button></section>
<section class="panel"><h2>哪些资料被修正或隔离？</h2><div id="rules"></div><p>APE 三天的异常 OHLC 用已缓存的 Coinbase USD 日线修正，原始文件保留。QNT 约 305% 的周涨幅经 Coinbase 验证后保留。其他核验结果见运行记录。未确认的拆分不猜测复权比例。</p><details><summary>试验边界</summary><ul id="limits"></ul><p>新闻特征沿用同周已有正文清洗与 VADER 情绪结果；all 模型继续使用新闻特征。算法、参数、52 周窗口和输入特征定义均未修改。历史基线特征保留原值；新增周按原公式在共同修正数据上重算。</p></details></section>
<p class="muted">历史模拟 · 未保存当年的实时数据版本 · 研究对照不等于可执行策略业绩</p>
</main><script type="application/json" id="payload">__DATA__</script><script id="experiment-script">
const D=JSON.parse(document.getElementById('payload').textContent),M=D.manifest,all=D.rows,$=id=>document.getElementById(id);
const models=['enet_all','enet_market','extra_all','extra_market'],variants=['legacy_clipped','raw'],names={legacy_clipped:'原目标',raw:'取消截断'},colors=['#72aaff','#4ed7c1','#c49af7','#ffc16e'];
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null,fmt=(x,n=2)=>x==null||!Number.isFinite(x)?'—':x.toFixed(n),pct=x=>x==null?'—':fmt(100*x)+'%',pp=x=>x==null?'—':fmt(100*x),median=a=>{if(!a.length)return null;const x=[...a].sort((a,b)=>a-b),i=Math.floor(x.length/2);return x.length%2?x[i]:(x[i-1]+x[i])/2};
const status={evaluated:'已评价',pending:'待到期',missing_price:'缺失价格',quarantined_price:'数据隔离'};let filtered=[],page=0;
function add(select,value,text){const o=document.createElement('option');o.value=value;o.textContent=text;select.appendChild(o)}
[...new Set(all.map(x=>x.symbol))].sort().forEach(s=>add($('coin'),s,s));models.forEach(m=>add($('model'),m,m));$('from').value=M.start;$('to').value=M.end;
$('period').textContent=M.start+' → '+M.end+' UTC · '+M.decision_weeks+' 个决策周 · 四模型 × 两种训练目标';
const layout={paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#b9c7dc',size:11},margin:{l:60,r:15,t:35,b:60},xaxis:{gridcolor:'#2a3950'},yaxis:{gridcolor:'#2a3950',tickformat:'.1%'},legend:{orientation:'h',y:1.14}};
const draw=(id,data,extra={})=>Plotly.react(id,data,{...layout,...extra},{responsive:true,displaylogo:false});
function tdrow(parent,values){const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});parent.appendChild(tr)}
function update(){filtered=all.filter(r=>(!$('coin').value||r.symbol===$('coin').value)&&(!$('model').value||r.model_name===$('model').value)&&r.decision_date>=$('from').value&&r.decision_date<=$('to').value);page=0;render()}
function render(){const valid=filtered.filter(r=>r.evaluation_status==='evaluated');$('coverage').textContent='每版 '+(valid.length/2).toLocaleString()+' 条可评价预测 · 共 '+filtered.length.toLocaleString()+' 条记录（含两版及未评价记录）';$('summary').replaceChildren();$('details').replaceChildren();const active=models.filter(m=>!$('model').value||m===$('model').value),stats={};let improved=0;
 active.forEach(m=>{stats[m]={};variants.forEach(v=>{const g=valid.filter(r=>r.model_name===m&&r.target_variant===v),s={n:g.length,mae:mean(g.map(r=>Math.abs(r.predicted_return-r.actual_return_raw))),clipped:mean(g.filter(r=>r.actual_return_model_target!=null).map(r=>Math.abs(r.predicted_return-r.actual_return_model_target))),rmse:Math.sqrt(mean(g.map(r=>(r.predicted_return-r.actual_return_raw)**2))??NaN),zero:mean(g.map(r=>Math.abs(r.actual_return_raw))),hit:mean(g.map(r=>r.direction_hit)),amp:median(g.map(r=>Math.abs(r.predicted_return))),near:mean(g.map(r=>Math.abs(r.predicted_return)<=.02?1:0))};stats[m][v]=s;const full=D.summary.find(r=>r.model_name===m&&r.target_variant===v);tdrow($('details'),[m+' / '+names[v],pp(s.mae),pp(s.clipped),pp(s.rmse),pct(s.near),$('coin').value||$('from').value!==M.start||$('to').value!==M.end?'见全量 CSV':fmt(full?.mean_weekly_rank_ic,4)])});const a=stats[m].legacy_clipped,b=stats[m].raw,delta=a.n?b.mae-a.mae:null;if(delta<0)improved++;tdrow($('summary'),[m,a.n,pp(a.mae),pp(b.mae),pp(delta),pp(a.zero),pct(a.hit),pct(b.hit)])});
 $('finding').textContent=valid.length?'当前筛选中，'+improved+' / '+active.length+' 个模型的真实收益 MAE 减少。是否有预测价值，还需与零预测基准和排名表现共同判断。':'当前筛选没有已结束且价格可核验的共同样本。';
 const bars=variants.map((v,i)=>({type:'bar',x:active,y:active.map(m=>stats[m][v].mae),name:names[v],marker:{color:i?'#e7a154':'#6c9cdd'}}));bars.push({type:'scatter',mode:'lines',x:active,y:active.map(m=>stats[m].legacy_clipped.zero),name:'零预测基准',line:{color:'#e5edf7',dash:'dot'}});draw('maechart',bars,{barmode:'group'});
 draw('amplitude',variants.map((v,i)=>({type:'bar',x:active,y:active.map(m=>stats[m][v].amp),name:names[v],marker:{color:i?'#e7a154':'#6c9cdd'}})),{barmode:'group'});
 const byDate=new Map(),actual=new Map(),traces=[];valid.forEach(r=>{const key=r.target_end+'|'+r.symbol;if(!actual.has(key))actual.set(key,r)});[...actual.values()].forEach(r=>{if(!byDate.has(r.target_end))byDate.set(r.target_end,[]);byDate.get(r.target_end).push(r.actual_return_raw)});const dates=[...byDate.keys()].sort();traces.push({x:dates,y:dates.map(d=>mean(byDate.get(d))),name:$('coin').value?'实际收益':'共同样本实际均值',mode:'lines',line:{color:'#e8eef8',width:2.5}});
 active.forEach(m=>variants.forEach(v=>{const map=new Map();valid.filter(r=>r.model_name===m&&r.target_variant===v).forEach(r=>{if(!map.has(r.target_end))map.set(r.target_end,[]);map.get(r.target_end).push(r.predicted_return)});const x=[...map.keys()].sort();traces.push({x,y:x.map(d=>mean(map.get(d))),name:m+' / '+names[v],mode:'lines',line:{color:colors[models.indexOf(m)],dash:v==='raw'?'dash':'solid',width:1.4}})}));draw('timeline',traces,{hovermode:'x unified'});table()}
function table(){const max=Math.max(1,Math.ceil(filtered.length/100));page=Math.max(0,Math.min(page,max-1));$('records').replaceChildren();const rows=[...filtered].sort((a,b)=>b.decision_date.localeCompare(a.decision_date)||a.symbol.localeCompare(b.symbol)||a.model_name.localeCompare(b.model_name)||a.target_variant.localeCompare(b.target_variant));rows.slice(page*100,(page+1)*100).forEach(r=>tdrow($('records'),[r.decision_date,r.target_end,r.symbol,r.model_name,names[r.target_variant],pct(r.predicted_return),pct(r.actual_return_raw),pp(r.prediction_error_raw),status[r.evaluation_status]||r.evaluation_status]));$('page').textContent=(page+1)+' / '+max;$('tablecount').textContent=filtered.length.toLocaleString()+' 条记录；每页 100 条';$('prev').disabled=page===0;$('next').disabled=page>=max-1}
function save(text,name,type){const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
$('export').onclick=()=>{const cols=['decision_date','target_end','symbol','model_name','target_variant','predicted_return','actual_return_raw','actual_return_model_target','prediction_error_raw','evaluation_status','missing_reason'],cell=v=>'"'+String(v??'').replaceAll('"','""')+'"';save('\ufeff'+cols.join(',')+'\n'+filtered.map(r=>cols.map(c=>cell(r[c])).join(',')).join('\n'),'target_comparison_filtered.csv','text/csv;charset=utf-8')};$('manifest').onclick=()=>save(JSON.stringify(M,null,2),'target_experiment_manifest.json','application/json');$('prev').onclick=()=>{page--;table()};$('next').onclick=()=>{page++;table()};['coin','model','from','to'].forEach(id=>$(id).onchange=update);$('reset').onclick=()=>{$('coin').value='';$('model').value='';$('from').value=M.start;$('to').value=M.end;update()};
M.preparation.quality_rules.rules.forEach(r=>{const p=document.createElement('p');p.textContent=r.symbol+' · '+(r.start||'起始')+' → '+(r.end||'研究截止')+'：'+r.reason;$('rules').appendChild(p)});M.preparation.limitations.forEach(s=>{const li=document.createElement('li');li.textContent=s;$('limits').appendChild(li)});update();
</script></body></html>'''
