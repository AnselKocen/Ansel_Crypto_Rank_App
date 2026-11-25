#!/usr/bin/env python3
"""Evaluate saved forecasts as ranked portfolios, without retraining models."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from replay_core import DEFAULT_OUTPUT, digest, records, write_csv, write_json

MODELS = ["enet_all", "enet_market", "extra_all", "extra_market"]
COLORS = dict(zip(MODELS + ["fusion_all", "fusion_market", "universe"],
                  ["#649bea", "#39b7a4", "#b17fe0", "#e7a052", "#cb627f", "#82949f", "#263548"]))


def ranked_weights(frame, strategy):
    ordered = frame.sort_values("predicted_rank")
    if len(ordered) < 40:
        raise ValueError("Original Top/Bottom 20 requires at least 40 eligible assets")
    selected = [(s, "long", .05) for s in ordered.head(20).symbol]
    if strategy == "LS":
        selected += [(s, "short", -.05) for s in ordered.tail(20).symbol]
    return selected


def fusion_weights(a, b):
    positions = []
    for role, sign, field in [("long", 1, "is_top20"), ("short", -1, "is_bottom20")]:
        sa, sb = set(a.loc[a[field], "symbol"]), set(b.loc[b[field], "symbol"])
        raw = {s: .65 if s in sa & sb else .35 for s in sorted(sa | sb)}
        total = sum(raw.values())
        positions += [(s, role, sign*w/total) for s, w in raw.items()]
    return positions


def longest_common_segment(weekly):
    complete = weekly.groupby("decision_date").status.apply(lambda s: s.eq("evaluated").all())
    dates = complete[complete].index.sort_values()
    if not len(dates):
        raise ValueError("No jointly evaluable portfolio weeks")
    segments, current = [], []
    for d in dates:
        if current and d-current[-1] != pd.Timedelta(days=7):
            segments.append(current); current = []
        current.append(d)
    segments.append(current)
    # Availability only; never choose a period by portfolio returns.
    longest = max(segments, key=len)
    return set(dates), longest, segments


def cumulative_statistics(returns):
    r = np.asarray(returns, dtype=float)
    if not len(r) or not np.isfinite(r).all():
        raise ValueError("Continuous NAV requires every return to be known")
    nav = np.r_[1., np.cumprod(1+r)]
    dd = nav/np.maximum.accumulate(nav)-1
    std = np.std(r, ddof=1) if len(r) > 1 else np.nan
    return {"total_return": float(nav[-1]-1), "max_drawdown": float(dd.min()),
            "mean_weekly_return": float(r.mean()), "positive_week_ratio": float((r>0).mean()),
            "annualized_sharpe_rf_zero": float(r.mean()/std*np.sqrt(52)) if std > 0 else None}, nav, dd


def build():
    source_root = DEFAULT_OUTPUT / "target_experiment"
    source = source_root / json.loads((source_root / "latest_run.json").read_text())["path"]
    original_manifest = json.loads((source / "run_manifest.json").read_text())
    out = DEFAULT_OUTPUT / "portfolio_evaluation"
    out.mkdir(exist_ok=True)
    comparison = pd.read_csv(source / "forecast_actual_comparison.csv", float_precision="round_trip",
                             parse_dates=["decision_date", "target_end"])
    end = pd.Timestamp(original_manifest["end"])
    holding_rows, week_rows = [], []

    def add(variant, decision, model, strategy, selected, quotes):
        target = decision + pd.Timedelta(days=7)
        complete = target <= end and all(quotes.loc[s, "evaluation_status"] == "evaluated" for s, _, _ in selected)
        contributions = []
        for symbol, side, weight in selected:
            record = quotes.loc[symbol]
            contribution = weight * record.actual_return_raw
            contributions.append(contribution)
            holding_rows.append({"target_variant": variant, "decision_date": decision, "target_end": target,
                                 "model_name": model, "strategy": strategy, "symbol": symbol, "side": side,
                                 "weight": weight, "actual_return_raw": record.actual_return_raw,
                                 "weighted_contribution": contribution, "evaluation_status": record.evaluation_status})
        if target <= end:
            week_rows.append({"target_variant": variant, "decision_date": decision, "target_end": target,
                              "model_name": model, "strategy": strategy,
                              "weekly_return": float(sum(contributions)) if complete else np.nan,
                              "status": "evaluated" if complete else "missing_constituents",
                              "holdings_count": len(selected), "gross_leg_exposure": sum(abs(w) for _, _, w in selected)})

    for (variant, decision), group in comparison.groupby(["target_variant", "decision_date"], sort=True):
        frames = {m: group[group.model_name.eq(m)].copy() for m in MODELS}
        universe = set(frames[MODELS[0]].symbol)
        if any(set(f.symbol) != universe for f in frames.values()):
            raise ValueError("Models have different eligible universes; benchmark must be redefined explicitly")
        quotes = frames[MODELS[0]].set_index("symbol")
        for model, frame in frames.items():
            for strategy in ["EW", "LS"]:
                add(variant, decision, model, strategy, ranked_weights(frame, strategy), quotes)
        for tag in ["all", "market"]:
            add(variant, decision, "fusion_"+tag, "LS", fusion_weights(frames["enet_"+tag], frames["extra_"+tag]), quotes)
        add(variant, decision, "universe", "EW", [(s, "long", 1/len(universe)) for s in sorted(universe)], quotes)

    weekly = pd.DataFrame(week_rows)
    holdings = pd.DataFrame(holding_rows)
    # Independently rebuild original ranking portfolios from individual holdings.
    saved = pd.read_csv(source / "strategy_returns.csv", float_precision="round_trip", parse_dates=["decision_date"])
    keys = ["target_variant", "decision_date", "model_name", "strategy"]
    joined = weekly[weekly.model_name.ne("universe")].merge(
        saved[saved.strategy.ne("baseline_EW")], on=keys, suffixes=("_rebuilt", "_saved"), validate="one_to_one")
    delta = (joined.weekly_return_rebuilt-joined.weekly_return_saved).abs().max()
    if not (joined.status_rebuilt.eq(joined.status_saved).all() and delta < 1e-12):
        raise ValueError("Holdings reconstruction differs from saved portfolio returns")
    common, longest, segments = longest_common_segment(weekly)
    weekly["common_week"] = weekly.decision_date.isin(common)
    weekly["longest_continuous_window"] = weekly.decision_date.isin(longest)
    segment_map = {d: i+1 for i, segment in enumerate(segments) for d in segment}
    weekly["common_segment"] = weekly.decision_date.map(segment_map).fillna(0).astype(int)
    summaries, curves = [], []
    for (variant, model, strategy), group in weekly.groupby(["target_variant", "model_name", "strategy"]):
        group = group.sort_values("decision_date")
        baseline = weekly[weekly.target_variant.eq(variant) & weekly.model_name.eq("universe")].set_index("decision_date").weekly_return
        matched = group[group.common_week].set_index("decision_date")
        reference = baseline.reindex(matched.index) if strategy == "EW" else pd.Series(0., index=matched.index)
        summaries.append({"scope": "all_common_observed_weeks", "target_variant": variant, "model_name": model,
                          "strategy": strategy, "weeks": len(matched), "own_evaluated_weeks": int(group.status.eq("evaluated").sum()),
                          "mean_weekly_return": float(matched.weekly_return.mean()),
                          "mean_weekly_excess": float((matched.weekly_return-reference).mean()),
                          "beat_reference_week_ratio": float((matched.weekly_return>reference).mean()),
                          "reference": "same_universe_equal_weight" if strategy == "EW" else "zero_return_cash",
                          "total_return": None, "max_drawdown": None})
        window = group[group.longest_continuous_window]
        stats, _, _ = cumulative_statistics(window.weekly_return)
        summaries.append({"scope": "longest_continuous_common_window", "target_variant": variant,
                          "model_name": model, "strategy": strategy, "weeks": len(window), **stats,
                          "reference": "same_universe_equal_weight" if strategy == "EW" else "zero_return_cash"})
        for segment, part in group[group.common_week].groupby("common_segment"):
            part = part.sort_values("decision_date")
            _, nav, dd = cumulative_statistics(part.weekly_return)
            for date, value, drawdown in zip([part.decision_date.iloc[0], *part.target_end], nav, dd):
                curves.append({"target_variant": variant, "model_name": model, "strategy": strategy,
                               "segment": int(segment), "date": date, "nav": float(value), "drawdown": float(drawdown)})
    summary = pd.DataFrame(summaries)
    source_files = {f: digest(source / f) for f in ["forecast_history.csv", "forecast_actual_comparison.csv", "strategy_returns.csv"]}
    manifest = {"status": "complete_restricted_portfolio_evaluation", "source_run_id": source.name,
                "source_hashes": source_files, "decision_start": original_manifest["start"], "data_end": original_manifest["end"],
                "finished_weeks": weekly.decision_date.nunique(), "common_evaluated_weeks": len(common),
                "missing_common_weeks": weekly.decision_date.nunique()-len(common),
                "continuous_segments": len(segments), "longest_window_start": str(longest[0].date()),
                "longest_window_end": str((longest[-1]+pd.Timedelta(days=7)).date()), "longest_window_weeks": len(longest),
                "longest_window_segment": segment_map[longest[0]],
                "rules": "Top 20 equally weighted long; Top 20 long / Bottom 20 short; original 0.65/0.35 consensus fusion. No threshold on the absolute predicted return.",
                "costs": "Gross returns. No trading fees, slippage, funding or borrow costs. Long-short legs are 100% long plus 100% short; no leverage equivalence to long-only is implied.",
                "missing_policy": "Never fill missing returns with zero or renormalize incomplete holdings. Compare all models/variants on identical observed weeks. Do not chain across gaps.",
                "window_selection": "Longest window chosen only by common data availability, never by return.",
                "rebuilt_vs_saved_max_difference": float(delta), "model_retraining": False,
                "quality_rules": original_manifest["preparation"]["quality_rules"],
                "limitations": original_manifest["preparation"]["limitations"]}
    write_csv(out / "portfolio_holdings.csv", holdings)
    write_csv(out / "portfolio_weekly_returns.csv", weekly)
    write_csv(out / "portfolio_summary.csv", summary)
    write_csv(out / "portfolio_nav_segments.csv", pd.DataFrame(curves))
    write_json(out / "portfolio_manifest.json", manifest)
    render(out, manifest, summary, weekly, curves)
    print(json.dumps({k:v for k,v in manifest.items() if k in ["source_run_id","finished_weeks","common_evaluated_weeks","longest_window_start","longest_window_end","longest_window_weeks","rebuilt_vs_saved_max_difference"]}, indent=2))
    print(summary[summary.scope.eq("longest_continuous_common_window") & summary.target_variant.eq("legacy_clipped")][["model_name","strategy","weeks","total_return","max_drawdown"]].to_string(index=False))


def render(out, manifest, summary, weekly, curves):
    report = out / "report"
    report.mkdir(exist_ok=True)
    payload = {"manifest": manifest, "summary": records(summary), "weekly": records(weekly), "curves": curves}
    payload["curves"] = records(pd.DataFrame(curves))
    page = HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    (report / "index.html").write_text(page)
    for f in out.glob("*.csv"):
        shutil.copy2(f, report / f.name)
    data = pd.DataFrame(curves)
    data = data[data.segment.eq(manifest["longest_window_segment"]) & data.target_variant.eq("legacy_clipped")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, strategy in zip(axes, ["EW", "LS"]):
        for model, group in data[data.strategy.eq(strategy)].groupby("model_name"):
            ax.plot(pd.to_datetime(group.date), group.nav, label=model, color=COLORS[model],
                    lw=2.3 if model=="universe" else 1.5, ls="--" if model=="universe" else "-")
        ax.axhline(1, color="gray", lw=.7)
        ax.set_title("Top-20 equal-weight long" if strategy=="EW" else "Top/Bottom-20 long-short (200% gross legs)")
        ax.set_ylabel("Gross NAV, initial capital = 1")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f"Original clipped-target ranks: common continuous window {manifest['longest_window_start']} to {manifest['longest_window_end']}")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(report / "ranked_portfolio_nav.png", dpi=160)
    plt.close(fig)


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>排名选币组合收益 · Ansel Crypto Rank</title><script>__PLOTLY__</script><style>
:root{color-scheme:dark;font:14px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#e9effa;background:#0b101b}*{box-sizing:border-box}body{margin:0}main{max-width:1460px;margin:auto;padding:30px}h1{font-size:30px;margin:8px 0}h2{font-size:18px}a{color:#89baff}.muted{color:#aab9d0}.notice{background:#302719;color:#eed8ae;border:1px solid #665335;padding:16px;border-radius:10px}.panel{margin-top:18px;background:#141e30;border:1px solid #2b3a53;border-radius:12px;padding:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart{height:360px}select,a.button{background:#0d1728;color:#e9effa;border:1px solid #42536d;border-radius:6px;padding:8px;text-decoration:none}a.button{display:inline-block;margin:5px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:10px;border-bottom:1px solid #2b3a53;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}details{margin-top:12px}@media(max-width:850px){main{padding:15px}.grid{grid-template-columns:1fr}.chart{height:320px}h1{font-size:24px}}
</style></head><body><main><small class="muted">ANSEL · RANKED PORTFOLIOS</small><h1>每周按排名选币，组合实际表现如何？</h1><p>主评价：组合收益、回撤及基准超额。模型输出用于排序，不要求准确报出每个币的涨跌幅。</p><p><a href="/target-experiment">收益率误差与训练目标对照（辅助）</a> · <a href="/">项目首页</a></p>
<p class="notice" id="coverage"></p><section class="panel"><label>训练目标版本 <select id="variant"><option value="legacy_clipped">原截断目标（默认）</option><option value="raw">取消截断试验</option></select></label><p class="muted">Top 20：每周按分数最高的 20 个币各持有 5%。多空：最高 20 个各 +5%，最低 20 个各 −5%。融合沿用原 0.65/0.35 权重。绝对预测值为负也不改变原选币规则。</p></section>
<div class="grid"><section class="panel"><h2>连续共同区间：Top 20 等权做多</h2><p class="muted">与当周同一可预测币池的等权组合比较</p><div id="long" class="chart"></div></section><section class="panel"><h2>连续共同区间：Top / Bottom 20 多空</h2><p class="muted">两侧各 100%，总腿敞口 200%；零收益线作参考</p><div id="ls" class="chart"></div></section></div>
<section class="panel"><h2 id="windowtitle"></h2><div class="scroll"><table><thead><tr><th>模型 / 组合</th><th>周数</th><th>累计毛收益</th><th>最大回撤</th><th>平均周收益</th><th>盈利周占比</th></tr></thead><tbody id="window"></tbody></table></div></section>
<section class="panel"><h2>全部共同有效周：不只看上面的连续区间</h2><p class="muted">跨越资料缺口的周仅做逐周统计，不连成完整期间累计收益。做多的超额对照同币池等权；多空对照零收益，不将不同敞口混为一谈。</p><div class="scroll"><table><thead><tr><th>模型 / 组合</th><th>共同周数</th><th>自身可评价周数</th><th>平均周收益</th><th>平均周超额（百分点）</th><th>战胜参考的周占比</th></tr></thead><tbody id="allweeks"></tbody></table></div></section>
<section class="panel"><h2>逐周收益与缺失状态</h2><div class="scroll" style="max-height:500px"><table><thead><tr><th>决策日</th><th>目标结束</th><th>模型 / 组合</th><th>实际周收益</th><th>状态</th></tr></thead><tbody id="weekly"></tbody></table></div><p><a class="button" data-file="portfolio_holdings.csv">下载逐周持仓与收益贡献</a><a class="button" data-file="portfolio_weekly_returns.csv">下载逐周组合收益</a><a class="button" data-file="portfolio_summary.csv">下载组合指标</a></p></section>
<section class="panel"><h2>评价边界</h2><p>所有收益均为毛收益，未扣手续费、滑点、资金费和借币成本。多空不代表每个币都能实际借入。缺失持仓不补零、不按剩余资产重新分配权重。最长区间只按资料连续性选取，未按收益挑选。</p><p>沿用本次行情修正及质量隔离；这不是对全部第三方历史报价的独立认证，也不是当年保存的实时预测。旧回测的训练和日期问题仍需另行复核，不能仅凭旧曲线比较优劣。</p><details><summary>原始回放资料与辅助诊断</summary><p><a href="/historical-replay">原始归档</a> · <a href="/target-experiment">训练目标及误差诊断</a></p></details></section></main>
<script type="application/json" id="payload">__DATA__</script><script id="portfolio-script">
const D=JSON.parse(document.getElementById('payload').textContent),M=D.manifest,$=id=>document.getElementById(id),pct=x=>x==null?'—':(x*100).toFixed(2)+'%',pp=x=>x==null?'—':(x*100).toFixed(3),label=r=>r.model_name==='universe'?'同币池等权基准':r.model_name+' / '+(r.strategy==='EW'?'Top 20 做多':'多空');
const colors={enet_all:'#72aaff',enet_market:'#4ed7c1',extra_all:'#c49af7',extra_market:'#ffc16e',fusion_all:'#ef8ca5',fusion_market:'#9dafbb',universe:'#f0f4ff'};
$('coverage').textContent='完整模拟已有 '+M.finished_weeks+' 个结束周；各模型、两版及基准共同可评价 '+M.common_evaluated_weeks+' 周，另有 '+M.missing_common_weeks+' 个共同缺口。曲线展示最长连续共同区间：'+M.longest_window_start+' → '+M.longest_window_end+'，共 '+M.longest_window_weeks+' 周。完整有效周统计另列如下。';
$('windowtitle').textContent='连续 '+M.longest_window_weeks+' 周的组合结果（'+M.longest_window_start+' → '+M.longest_window_end+'）';
function row(parent,values){const tr=document.createElement('tr');values.forEach(x=>{const td=document.createElement('td');td.textContent=x;tr.appendChild(td)});parent.appendChild(tr)}
function render(){const v=$('variant').value;$('window').replaceChildren();$('allweeks').replaceChildren();$('weekly').replaceChildren();D.summary.filter(r=>r.target_variant===v).forEach(r=>{if(r.scope==='longest_continuous_common_window')row($('window'),[label(r),r.weeks,pct(r.total_return),pct(r.max_drawdown),pct(r.mean_weekly_return),pct(r.positive_week_ratio)]);else row($('allweeks'),[label(r),r.weeks,r.own_evaluated_weeks,pct(r.mean_weekly_return),pp(r.mean_weekly_excess),pct(r.beat_reference_week_ratio)])});
 for(const [id,s] of [['long','EW'],['ls','LS']]){const groups=new Map();D.curves.filter(r=>r.target_variant===v&&r.strategy===s&&r.segment===M.longest_window_segment).forEach(r=>{if(!groups.has(r.model_name))groups.set(r.model_name,[]);groups.get(r.model_name).push(r)});const traces=[...groups].map(([m,g])=>({x:g.map(r=>r.date),y:g.map(r=>r.nav),mode:'lines',name:m==='universe'?'同币池等权':m,line:{color:colors[m],width:m==='universe'?2.5:1.6,dash:m==='universe'?'dash':'solid'}}));Plotly.react(id,traces,{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#bccbdf',size:11},margin:{l:55,r:15,t:35,b:55},legend:{orientation:'h',y:1.2},xaxis:{gridcolor:'#293a52'},yaxis:{title:'净值（起点=1）',gridcolor:'#293a52'},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:'#78869b',width:.6}}]},{responsive:true,displaylogo:false,modeBarButtonsToRemove:['share','sendDataToCloud']})}
 D.weekly.filter(r=>r.target_variant===v).sort((a,b)=>b.decision_date.localeCompare(a.decision_date)||label(a).localeCompare(label(b))).forEach(r=>row($('weekly'),[r.decision_date,r.target_end,label(r),pct(r.weekly_return),r.status==='evaluated'?'已评价':'持仓价格缺失／隔离']));}
document.querySelectorAll('a[data-file]').forEach(a=>{a.href=location.protocol==='file:'?a.dataset.file:'/api/replay/portfolio/download/'+a.dataset.file;a.download=a.dataset.file});$('variant').onchange=render;render();
</script></body></html>'''


if __name__ == "__main__":
    build()
