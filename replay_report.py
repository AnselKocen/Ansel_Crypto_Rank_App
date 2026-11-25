"""Offline interactive report and exportable figures for a completed replay."""
from __future__ import annotations

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

COLORS = {"enet_all": "#63a5ff", "enet_market": "#43d6c1",
          "extra_all": "#c79aff", "extra_market": "#ffbc70"}


def build_report(run_dir: Path, out: Path, prices: pd.DataFrame):
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    comparison = pd.read_csv(run_dir / "forecast_actual_comparison.csv")
    metrics = pd.read_csv(run_dir / "weekly_model_metrics.csv")
    strategies = pd.read_csv(run_dir / "strategy_returns.csv")
    coverage = pd.read_csv(out / "data_coverage.csv")
    fields = ["decision_date", "target_start", "target_end", "model_name", "symbol",
              "predicted_return", "actual_return_raw", "prediction_error_raw",
              "direction_hit", "predicted_rank", "evaluation_status"]
    payload = {"manifest": manifest, "comparison": records(comparison[fields]),
               "metrics": records(metrics), "strategies": records(strategies),
               "coverage": records(coverage)}
    report = run_dir / "report"
    report.mkdir(exist_ok=True)
    script_data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", script_data)
    (report / "index.html").write_text(page, encoding="utf-8")
    stable = out / "report"
    stable.mkdir(exist_ok=True)
    (stable / "index.html").write_text(page, encoding="utf-8")
    render_figures(comparison, metrics, strategies, report)
    for figure in report.glob("*.png"):
        shutil.copy2(figure, stable / figure.name)


def render_figures(comparison, metrics, strategies, report):
    plt.style.use("seaborn-v0_8-whitegrid")
    btc = comparison[comparison.symbol == "BTC"].copy()
    btc["target_end"] = pd.to_datetime(btc.target_end)
    evaluated = btc[btc.evaluation_status == "evaluated"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    actual = evaluated.drop_duplicates("target_end").sort_values("target_end")
    axes[0].plot(actual.target_end, actual.actual_return_raw, color="#182b47", lw=2.2, label="Actual")
    for model, group in btc.groupby("model_name"):
        group = group.sort_values("target_end")
        axes[0].plot(group.target_end, group.predicted_return, color=COLORS[model], lw=1.2, label=model)
        valid = group[group.evaluation_status == "evaluated"]
        axes[1].plot(valid.target_end, valid.prediction_error_raw, color=COLORS[model], lw=1, label=model)
    for ax in axes:
        ax.axhline(0, color="grey", lw=.7)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("BTC: next-week return forecasts vs actual USD returns")
    axes[1].set_title("Forecast minus actual (percentage points)")
    axes[0].legend(ncol=5, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(report / "btc_prediction_vs_actual.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model, group in evaluated.groupby("model_name"):
        axes[0].scatter(group.predicted_return, group.actual_return_raw, s=18, alpha=.6,
                        color=COLORS[model], label=model)
    values = evaluated[["predicted_return", "actual_return_raw"]].to_numpy().flatten()
    values = values[np.isfinite(values)]
    if len(values):
        low, high = values.min(), values.max()
        axes[0].plot([low, high], [low, high], "--", color="grey")
    axes[0].set(xlabel="Predicted weekly return", ylabel="Actual weekly return", title="BTC forecast calibration")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for model, group in metrics.groupby("model_name"):
        group = group.dropna(subset=["mae"]).sort_values("target_end")
        axes[1].plot(pd.to_datetime(group.target_end), group.mae, color=COLORS[model], label=model)
    axes[1].set_title("All covered assets: weekly mean absolute error")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(report / "calibration_and_weekly_error.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 5))
    for (model, strategy, segment), group in strategies.groupby(["model_name", "strategy", "segment_id"]):
        if strategy != "EW" or not segment or model not in COLORS:
            continue
        group = group.sort_values("target_end")
        ax.plot(pd.to_datetime(group.target_end), group.cumulative_value, color=COLORS[model],
                label=f"{model} segment {int(segment)}", lw=1.5)
    ax.set(title="Top-20 equal weight: actual gross NAV (each complete segment starts at 1)",
           ylabel="NAV, no transaction costs")
    if ax.lines:
        ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(report / "strategy_nav.png", dpi=160)
    plt.close(fig)


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>历史预测回放 · Ansel Crypto Rank</title>
<script>__PLOTLY__</script>
<style>
:root{color-scheme:dark;--bg:#0b101b;--panel:#131d2e;--line:#28364e;--muted:#a3b2c9;--text:#eef4ff;--accent:#63a5ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}
main{max-width:1500px;margin:auto;padding:32px}a{color:var(--accent)}h1{font-size:32px;line-height:1.3;margin:10px 0}h2{font-size:18px;margin:0}
.eyebrow{letter-spacing:3px;color:#79b2ff;font-size:11px}.muted{color:var(--muted)}.header{display:flex;justify-content:space-between;gap:24px;align-items:center}
.badge{border:1px solid #42644f;color:#b1e8c1;padding:6px 12px;border-radius:30px;white-space:nowrap}.notice{background:#282319;border:1px solid #655130;color:#ead9b1;padding:14px 18px;border-radius:12px;margin:24px 0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{border:1px solid var(--line);background:var(--panel);border-radius:14px}.card{padding:18px}.card b{display:block;font-size:27px;line-height:1.5}.card small{color:var(--muted)}.panel{padding:20px;margin-top:18px;min-width:0}
.filters{display:flex;flex-wrap:wrap;gap:14px;align-items:end}.filters label{display:flex;flex-direction:column;gap:5px;min-width:140px;color:var(--muted)}
select,input,button{font:inherit;border:1px solid #3b4d69;border-radius:7px;padding:9px 11px;background:#0e1829;color:var(--text)}button{cursor:pointer}button:hover{border-color:var(--accent)}button.primary{background:#254e85}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart{height:350px}.wide{height:400px}
.panelhead{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:8px}.tablewrap{overflow:auto;max-height:560px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}thead{position:sticky;top:0;background:#17263d}td.pos{color:#8ee6c0}td.neg{color:#ffa8a8}.pagination{display:flex;gap:12px;align-items:center;justify-content:end;margin-top:12px}.empty{padding:40px;text-align:center;color:var(--muted)}details{margin-top:14px}summary{cursor:pointer}.foot{margin-top:28px;font-size:12px;color:var(--muted)}
@media(max-width:850px){main{padding:16px}.grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}.header{align-items:start}.badge{font-size:11px}h1{font-size:25px}.chart{height:320px}.filters label{flex:1;min-width:120px}}
</style></head><body><main>
<header class="header"><div><div class="eyebrow">ANSEL · HISTORICAL REPLAY</div><h1>过去每一周，模型会怎么预测？</h1><div class="muted" id="subtitle"></div><a href="/" style="font-size:12px">返回项目首页</a></div><div class="badge" id="runstatus">正在读取结果</div></header>
<div class="notice" id="notice"></div>
<section class="cards"><div class="card"><small>当前筛选 · 可评价预测</small><b id="count">—</b><small id="coverage"></small></div><div class="card"><small>平均绝对误差</small><b id="mae">—</b><small>预测与实际周收益之差</small></div><div class="card"><small>方向准确率</small><b id="hit">—</b><small>上涨／下跌／零变化方向一致</small></div><div class="card"><small>待到期 / 缺失价格</small><b id="missing">—</b><small>不进入有效评分</small></div></section>
<section class="panel"><div class="filters">
<label>币种<select id="coin" aria-label="币种"></select></label>
<label>模型<select id="model" aria-label="模型"><option value="">全部四个模型</option></select></label>
<label>决策周从<input id="from" type="date"></label><label>决策周至<input id="to" type="date"></label>
<label>评分状态<select id="state"><option value="">全部状态</option><option value="evaluated">已评价</option><option value="pending">待到期</option><option value="missing_price">缺失价格</option></select></label>
<button id="reset">重置为 BTC</button><button class="primary" id="export">导出当前对照 CSV</button>
</div></section>
<section class="panel"><div class="panelhead"><h2>预测与实际周收益</h2><span class="muted">横轴：目标周结束日 · 双击图表重置缩放</span></div><div id="timeline" class="chart wide"></div></section>
<div class="grid"><section class="panel"><h2>逐周偏差</h2><div class="muted">预测 − 实际；0 表示完全一致</div><div id="error" class="chart"></div></section><section class="panel"><h2>预测与现实的距离</h2><div class="muted">虚线为 y=x；最多绘制 3,000 个点，指标使用全部样本</div><div id="scatter" class="chart"></div></section></div>
<div class="grid"><section class="panel"><h2>每周平均绝对误差</h2><div id="weekly" class="chart"></div></section><section class="panel"><h2>四模型逐周方向准确率</h2><div id="heat" class="chart"></div></section></div>
<section class="panel"><div class="panelhead"><h2>按原规则持有，实际会怎样？</h2><select id="strategy"><option value="EW">Top 20 等权做多</option><option value="LS">Top 20 / Bottom 20 多空</option><option value="baseline_EW">当期预测币池等权基准</option></select></div><div class="muted">使用全部组合资产，币种筛选不影响此图。毛收益未计手续费、滑点和资金费；多空总敞口为 200%。缺失持仓收益时不补零，每段连续可评价区间从净值 1 重新开始。</div><div id="nav" class="chart wide"></div></section>
<section class="panel"><div class="panelhead"><h2>预测与实际对照明细</h2><span class="muted" id="tablecount"></span></div><div class="tablewrap"><table><thead><tr><th>决策日</th><th>目标周结束</th><th>模型</th><th>币种</th><th>预测收益</th><th>实际收益</th><th>误差（百分点）</th><th>预测排名</th><th>状态</th></tr></thead><tbody id="table"></tbody></table></div><div class="pagination"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div></section>
<section class="panel"><h2>资料覆盖与可追溯记录</h2><div id="sources" class="muted"></div><details><summary>查看缺失或覆盖不完整的币种</summary><div class="tablewrap"><table><thead><tr><th>币种</th><th>状态</th><th>起始</th><th>截止</th><th>说明</th></tr></thead><tbody id="unavailable"></tbody></table></div></details><button id="manifest" style="margin-top:16px">下载运行配置 JSON</button></section>
<div class="foot">历史模拟，不是当年留存的实时预测。四模型沿用原固定参数与 52 周训练规则。旧回测结果未在本报告中重算或拼接。</div>
</main><script type="application/json" id="payload">__DATA__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),M=D.manifest;
const colors={enet_all:'#63a5ff',enet_market:'#43d6c1',extra_all:'#c79aff',extra_market:'#ffbc70'};
const models=Object.keys(colors),all=D.comparison,el=id=>document.getElementById(id);
const pct=x=>x==null||!Number.isFinite(x)?'—':(x*100).toFixed(2)+'%';
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const labels={evaluated:'已评价',pending:'待到期',missing_price:'缺失价格'};
let filtered=[],page=0;
const base={paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#b8c7df',size:11},margin:{l:56,r:18,t:15,b:50},legend:{orientation:'h',y:1.15},xaxis:{gridcolor:'#27364d'},yaxis:{gridcolor:'#27364d',tickformat:'.0%',zerolinecolor:'#637086'},hovermode:'closest'};
function draw(id,traces,extra={}){Plotly.react(id,traces,{...base,...extra},{responsive:true,displaylogo:false,modeBarButtonsToRemove:['sendDataToCloud','share'],toImageButtonOptions:{format:'png',scale:2}})}
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o)}
option(el('coin'),'','全部币种');[...new Set(all.map(r=>r.symbol))].sort().forEach(s=>option(el('coin'),s,s));
models.forEach(m=>option(el('model'),m,m));el('coin').value='BTC';el('from').value=M.start;el('to').value=M.end;
el('subtitle').textContent=M.start+' → '+M.end+' UTC · '+M.decision_weeks+' 个决策周 · 四模型逐周重新训练';
el('runstatus').textContent=M.status==='complete'?'回放完成':'回放完成 · 部分输入缺失';
const prep=M.prepared_data;
el('notice').textContent='数据口径：'+(prep.price_source==='yahoo'?'公开 Yahoo USD 行情；未覆盖币种仅在已有日期复用项目 CoinDesk 缓存。Yahoo 基础币成交量由美元成交量与收盘价估算。':'CoinDesk CADLI USD 行情。')+' 新闻优先使用已保存原数据，缺失周使用公开出版方文章。采用起点固定币池，不等同于每周历史 Top 200。完整限制见下方运行配置。';
function update(){filtered=all.filter(r=>(!el('coin').value||r.symbol===el('coin').value)&&(!el('model').value||r.model_name===el('model').value)&&r.decision_date>=el('from').value&&r.decision_date<=el('to').value&&(!el('state').value||r.evaluation_status===el('state').value));page=0;render()}
function render(){
 const valid=filtered.filter(r=>r.evaluation_status==='evaluated'),pending=filtered.filter(r=>r.evaluation_status==='pending').length,missing=filtered.filter(r=>r.evaluation_status==='missing_price').length;
 el('count').textContent=valid.length.toLocaleString();el('coverage').textContent='共 '+filtered.length.toLocaleString()+' 条预测';
 el('mae').textContent=pct(mean(valid.map(r=>Math.abs(r.prediction_error_raw))));el('hit').textContent=pct(mean(valid.map(r=>r.direction_hit)));el('missing').textContent=pending+' / '+missing;
 const active=models.filter(m=>!el('model').value||m===el('model').value),timeline=[],error=[],scatter=[],weekly=[];
 const groups=new Map();valid.forEach(r=>{const k=r.target_end+'|'+r.model_name;if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r)});
 const actual=new Map();valid.forEach(r=>{const k=r.target_end+'|'+r.symbol;if(!actual.has(k))actual.set(k,r)});
 const byDate=new Map();[...actual.values()].forEach(r=>{if(!byDate.has(r.target_end))byDate.set(r.target_end,[]);byDate.get(r.target_end).push(r.actual_return_raw)});
 const dates=[...byDate.keys()].sort();timeline.push({x:dates,y:dates.map(d=>mean(byDate.get(d))),name:el('coin').value?'实际收益':'样本实际收益均值',mode:'lines+markers',line:{color:'#eef4ff',width:2.5},marker:{size:4},connectgaps:false});
 active.forEach(m=>{
  const rows=filtered.filter(r=>r.model_name===m),weeks=[...new Set(rows.map(r=>r.target_end))].sort(),agg=new Map();
  rows.forEach(r=>{if(!agg.has(r.target_end))agg.set(r.target_end,[]);agg.get(r.target_end).push(r)});
  timeline.push({x:weeks,y:weeks.map(d=>mean(agg.get(d).map(r=>r.predicted_return))),name:m,mode:'lines',line:{color:colors[m],width:1.5}});
  const good=weeks.filter(d=>groups.has(d+'|'+m));error.push({x:good,y:good.map(d=>mean(groups.get(d+'|'+m).map(r=>r.prediction_error_raw))),name:m,mode:'lines',line:{color:colors[m]}});
  weekly.push({x:good,y:good.map(d=>mean(groups.get(d+'|'+m).map(r=>Math.abs(r.prediction_error_raw)))),name:m,mode:'lines',line:{color:colors[m]}});
  const points=valid.filter(r=>r.model_name===m),stride=Math.max(1,Math.ceil(points.length/750)),sample=points.filter((_,i)=>i%stride===0);
  scatter.push({x:sample.map(r=>r.predicted_return),y:sample.map(r=>r.actual_return_raw),text:sample.map(r=>r.symbol+' · '+r.target_end),name:m,mode:'markers',marker:{color:colors[m],size:5,opacity:.65}});
 });
 const vals=valid.flatMap(r=>[r.predicted_return,r.actual_return_raw]);if(vals.length){const low=vals.reduce((a,b)=>Math.min(a,b),Infinity),high=vals.reduce((a,b)=>Math.max(a,b),-Infinity);scatter.push({x:[low,high],y:[low,high],mode:'lines',name:'y=x',line:{color:'#7b8ba3',dash:'dash'},hoverinfo:'skip'})}
 draw('timeline',timeline,{hovermode:'x unified'});draw('error',error);draw('weekly',weekly);
 draw('scatter',scatter,{xaxis:{...base.xaxis,title:'预测周收益',tickformat:'.0%'},yaxis:{...base.yaxis,title:'实际周收益'}});
 const hd=[...new Set(valid.map(r=>r.target_end))].sort(),z=active.map(m=>hd.map(d=>{const g=groups.get(d+'|'+m);return g?mean(g.map(r=>r.direction_hit)):null}));
 draw('heat',[{type:'heatmap',x:hd,y:active,z,zmin:0,zmax:1,colorscale:[[0,'#9e4853'],[.5,'#243752'],[1,'#429b8c']],hovertemplate:'%{y}<br>%{x}<br>准确率 %{z:.1%}<extra></extra>',colorbar:{tickformat:'.0%'}}],{yaxis:{gridcolor:'#27364d'},showlegend:false});
 renderNav();renderTable();
}
function renderNav(){const traces=[],choice=el('strategy').value;const buckets=new Map();D.strategies.filter(r=>r.strategy===choice&&r.segment_id>0&&r.cumulative_value!=null&&(!el('model').value||r.model_name===el('model').value)&&r.decision_date>=el('from').value&&r.decision_date<=el('to').value).forEach(r=>{const k=r.model_name+'|'+r.segment_id;if(!buckets.has(k))buckets.set(k,[]);buckets.get(k).push(r)});buckets.forEach((g,k)=>{g.sort((a,b)=>a.target_end.localeCompare(b.target_end));traces.push({x:g.map(r=>r.target_end),y:g.map(r=>r.cumulative_value),name:g[0].model_name+' · 连续段 '+g[0].segment_id,mode:'lines',line:{color:colors[g[0].model_name]||'#a5b1c7'},connectgaps:false})});draw('nav',traces,{yaxis:{...base.yaxis,tickformat:'.2f',title:'实际毛收益净值'}})}
function renderTable(){const rows=[...filtered].sort((a,b)=>b.decision_date.localeCompare(a.decision_date)||a.model_name.localeCompare(b.model_name)||a.predicted_rank-b.predicted_rank);const max=Math.max(1,Math.ceil(rows.length/100));page=Math.max(0,Math.min(page,max-1));el('table').replaceChildren();rows.slice(page*100,(page+1)*100).forEach(r=>{const tr=document.createElement('tr'),values=[r.decision_date,r.target_end,r.model_name,r.symbol,pct(r.predicted_return),pct(r.actual_return_raw),r.prediction_error_raw==null?'—':(r.prediction_error_raw*100).toFixed(2),r.predicted_rank,labels[r.evaluation_status]||r.evaluation_status];values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===5&&r.actual_return_raw!=null)td.className=r.actual_return_raw>=0?'pos':'neg';tr.appendChild(td)});el('table').appendChild(tr)});el('page').textContent=(page+1)+' / '+max;el('tablecount').textContent=rows.length.toLocaleString()+' 条 · 每页 100 条';el('prev').disabled=page===0;el('next').disabled=page>=max-1}
function save(text,name,type){const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
el('export').onclick=()=>{const fields=['decision_date','target_start','target_end','model_name','symbol','predicted_return','actual_return_raw','prediction_error_raw','direction_hit','predicted_rank','evaluation_status'];const cell=v=>'"'+String(v??'').replaceAll('"','""')+'"';save('\ufeff'+fields.join(',')+'\n'+filtered.map(r=>fields.map(k=>cell(r[k])).join(',')).join('\n'),'forecast_actual_filtered.csv','text/csv;charset=utf-8')};
el('manifest').onclick=()=>save(JSON.stringify(M,null,2),'run_manifest.json','application/json');
['coin','model','from','to','state'].forEach(id=>el(id).addEventListener('change',update));
el('strategy').onchange=renderNav;el('prev').onclick=()=>{page--;renderTable()};el('next').onclick=()=>{page++;renderTable()};
el('reset').onclick=()=>{el('coin').value='BTC';el('model').value='';el('from').value=M.start;el('to').value=M.end;el('state').value='';update()};
el('sources').textContent='运行 '+M.run_id+'；起点币池 '+prep.universe.length+' 个，有可用行情 '+prep.price_symbols+' 个，其中完整覆盖至截止日 '+(prep.complete_price_symbols??'—')+' 个、公开下载 '+(prep.downloaded_price_symbols??prep.price_symbols)+' 个；'+prep.news_rows.toLocaleString()+' 条新闻。未完整覆盖币种及其缓存区间如下；缺失实际价格不计为零。';
D.coverage.filter(r=>r.status!=='complete').forEach(r=>{const tr=document.createElement('tr');[r.symbol,r.status,r.first_date||'—',r.last_date||'—',r.error||'覆盖截止早于研究截止'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});el('unavailable').appendChild(tr)});
update();
</script></body></html>"""
