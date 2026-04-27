"""
synthedge.report - Professional light dashboard HTML gap report
"""
import json
from datetime import datetime


def generate_report(se, output_path="synthedge_report.html",
                    dataset_name="Dataset", comparison_results=None):

    if se._top_voxels is None:
        raise RuntimeError("Call .analyze() before generating a report.")

    severity   = se._severity or {}
    top_voxels = se._top_voxels or []
    aug_meta   = se._aug_meta or {}
    gap_map_df = se.gap_map

    sev_level  = severity.get("severity", "UNKNOWN")
    sev_score  = severity.get("score", 0)
    will_help  = severity.get("will_help", False)
    rec_text   = severity.get("recommendation", "")
    signals    = severity.get("signals", {})
    total_added = aug_meta.get("total_added", 0)
    ctgan_used  = aug_meta.get("ctgan_used", False)
    vox_meta    = aug_meta.get("voxels", [])
    n_rows     = len(se.df)
    pos_rate   = round(float(se._y.mean()) * 100, 1)
    n_features = len(se.feature_cols)
    timestamp  = datetime.now().strftime("%d %b %Y, %H:%M")

    sev_map = {
        "NONE":     ("#22c55e", "#f0fdf4", "No structural gaps",       "✓"),
        "MILD":     ("#f59e0b", "#fffbeb", "Minor structural gaps",    "~"),
        "MODERATE": ("#f97316", "#fff7ed", "Moderate structural gaps", "!"),
        "SEVERE":   ("#ef4444", "#fef2f2", "Severe structural gaps",   "!!"),
    }
    sev_color, sev_bg, sev_sub, sev_icon = sev_map.get(
        sev_level, ("#6b7280", "#f9fafb", "Unknown", "?"))
    help_color = "#22c55e" if will_help else "#94a3b8"
    help_text  = "Recommended" if will_help else "Not required"

    # ── Voxel rows ────────────────────────────────────────────────────
    voxel_rows = ""
    for idx, (_, row) in enumerate(gap_map_df.iterrows()):
        score = float(row["gap_score"])
        bar   = int(score * 100)
        sc    = "#ef4444" if score >= 0.7 else "#f97316" if score >= 0.5 else "#f59e0b"
        rbg   = "#1e3a5f" if idx == 0 else "#f1f5f9"
        rcl   = "#ffffff"  if idx == 0 else "#475569"
        voxel_rows += (
            f'<tr>'
            f'<td><span style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:24px;height:24px;border-radius:50%;font-size:11px;font-weight:700;'
            f'background:{rbg};color:{rcl};">#{idx+1}</span></td>'
            f'<td><code style="background:#f1f5f9;padding:2px 8px;border-radius:4px;'
            f'font-size:12px;color:#1e3a5f;">{row["voxel"]}</code></td>'
            f'<td style="text-align:center;font-weight:600;">{row["observed"]}</td>'
            f'<td style="text-align:center;color:#94a3b8;">{row["expected"]}</td>'
            f'<td style="text-align:center;font-weight:600;">{row["n_pos"]}</td>'
            f'<td style="text-align:center;">{row["sparsity"]}</td>'
            f'<td style="text-align:center;">{row["entropy"]}</td>'
            f'<td><div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="flex:1;background:#e2e8f0;border-radius:4px;height:6px;">'
            f'<div style="width:{bar}%;background:{sc};border-radius:4px;height:6px;"></div>'
            f'</div><span style="font-weight:700;color:{sc};font-size:13px;min-width:36px;">'
            f'{row["gap_score"]}</span></div></td>'
            f'</tr>'
        )

    # ── Synthesis rows ────────────────────────────────────────────────
    synth_rows = ""
    for v in vox_meta:
        m = v.get("method","none")
        a = v.get("added", 0)
        if m == "ctgan":
            badge = '<span style="background:#dbeafe;color:#1d4ed8;font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;">CTGAN</span>'
        elif m == "gaussian":
            badge = '<span style="background:#ede9fe;color:#6d28d9;font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;">Gaussian</span>'
        else:
            badge = '<span style="background:#f1f5f9;color:#94a3b8;font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;">None</span>'
        ac = "#22c55e" if a > 0 else "#94a3b8"
        synth_rows += (
            f'<tr><td><code style="background:#f1f5f9;padding:2px 8px;border-radius:4px;'
            f'font-size:12px;color:#1e3a5f;">{v.get("voxel","—")}</code></td>'
            f'<td>{badge}</td>'
            f'<td style="font-weight:700;color:{ac};">+{a}</td></tr>'
        )
    if not synth_rows:
        synth_rows = '<tr><td colspan="3" style="text-align:center;color:#94a3b8;padding:1.5rem;">No synthesis performed</td></tr>'

    # ── Signal cards ──────────────────────────────────────────────────
    sig_items = [
        ("Positive rate",    f"{round(signals.get('positive_rate',0)*100,1)}%"),
        ("Max gap score",    f"{signals.get('max_gap_score',0):.4f}"),
        ("Mean gap score",   f"{signals.get('mean_gap_score',0):.4f}"),
        ("High-gap voxels",  f"{round(signals.get('high_gap_fraction',0)*100,1)}%"),
        ("Imbalance signal", f"{signals.get('imbalance_signal',0):.4f}"),
        ("Dataset size",     f"{signals.get('dataset_size',0):,}"),
    ]
    signal_cards = "".join(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
        f'padding:12px 16px;display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:12px;color:#64748b;">{lbl}</span>'
        f'<span style="font-size:14px;font-weight:700;color:#1e293b;">{val}</span></div>'
        for lbl, val in sig_items
    )

    # ── Metric cards ──────────────────────────────────────────────────
    mc_data = [
        ("Dataset size",   f"{n_rows:,}",    "rows",              "#3b82f6"),
        ("Positive rate",  f"{pos_rate}%",   "minority class",    "#8b5cf6"),
        ("Gap voxels",     str(len(top_voxels)), "detected",      "#f59e0b"),
        ("Samples added",  str(total_added), "targeted",          "#22c55e"),
        ("Max gap score",  f"{signals.get('max_gap_score',0):.3f}", "0-1","#ef4444"),
        ("Features",       str(n_features),  "dimensions",        "#06b6d4"),
    ]
    metric_html = "".join(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div><p style="font-size:12px;color:#64748b;margin-bottom:6px;">{t}</p>'
        f'<p style="font-size:26px;font-weight:700;color:#1e293b;letter-spacing:-0.5px;line-height:1;">{v}</p>'
        f'<p style="font-size:11px;color:#94a3b8;margin-top:4px;">{s}</p></div>'
        f'<div style="width:36px;height:36px;border-radius:8px;background:{c}18;'
        f'display:flex;align-items:center;justify-content:center;">'
        f'<div style="width:10px;height:10px;border-radius:50%;background:{c};"></div>'
        f'</div></div></div>'
        for t, v, s, c in mc_data
    )

    # ── Charts ────────────────────────────────────────────────────────
    chart_js = ""
    chart_section = ""
    if comparison_results:
        lbls   = list(comparison_results.keys())
        recalls  = [round(comparison_results[k].get("recall",  0)*100,1) for k in lbls]
        f1s      = [round(comparison_results[k].get("f1",      0)*100,1) for k in lbls]
        aucs     = [round(comparison_results[k].get("roc_auc", 0)*100,1) for k in lbls]
        pr_aucs  = [round(comparison_results[k].get("pr_auc",  0)*100,1) for k in lbls]
        colors   = ["#3b82f6" if "SynthEdge" in l else "#f97316" if "SMOTE" in l
                    else "#a855f7" if "ADASYN" in l else "#94a3b8" for l in lbls]

        chart_js = (
            f"const L={json.dumps(lbls)},R={json.dumps(recalls)},"
            f"F={json.dumps(f1s)},A={json.dumps(aucs)},P={json.dumps(pr_aucs)},"
            f"C={json.dumps(colors)};"
            "const opts=()=>({responsive:true,maintainAspectRatio:false,"
            "plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',padding:10,"
            "callbacks:{label:c=>' '+c.parsed.y.toFixed(1)+'%'}}},"
            "scales:{y:{beginAtZero:true,max:100,grid:{color:'#f1f5f9',drawBorder:false},"
            "ticks:{font:{size:11,family:'Inter'},color:'#94a3b8',callback:v=>v+'%'},"
            "border:{display:false}},x:{grid:{display:false},"
            "ticks:{font:{size:11,family:'Inter'},color:'#64748b'},"
            "border:{display:false}}}});"
            "const mk=(id,data)=>new Chart(document.getElementById(id),"
            "{type:'bar',data:{labels:L,datasets:[{label:'',data,backgroundColor:C,"
            "borderRadius:6,borderSkipped:false}]},options:opts()});"
            "mk('cR',R);mk('cF',F);mk('cA',A);mk('cP',P);"
        )

        legend_html = "".join(
            f'<span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#64748b;">'
            f'<span style="width:10px;height:10px;border-radius:2px;background:{c};"></span>{l}</span>'
            for l, c in zip(lbls, colors)
        )

        chart_section = (
            '<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;'
            'padding:20px 24px;margin-bottom:24px;">'
            '<div style="display:flex;justify-content:space-between;align-items:flex-start;'
            'margin-bottom:20px;flex-wrap:wrap;gap:12px;">'
            '<div><p style="font-size:15px;font-weight:600;color:#0f172a;">Model performance comparison</p>'
            '<p style="font-size:12px;color:#94a3b8;margin-top:3px;">Held-out test set — higher is better</p></div>'
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;">{legend_html}</div></div>'
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;">'
            + "".join(
                f'<div><p style="font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;'
                f'letter-spacing:0.06em;margin-bottom:10px;">{name}</p>'
                f'<div style="position:relative;height:180px;">'
                f'<canvas id="{cid}" role="img" aria-label="{name} comparison"></canvas></div></div>'
                for name, cid in [("Recall","cR"),("F1 Score","cF"),("ROC-AUC","cA"),("PR-AUC","cP")]
            )
            + '</div></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SynthEdge Report — {dataset_name}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Inter',system-ui,sans-serif;background:#f8fafc;color:#1e293b;font-size:14px;line-height:1.5;}}
a{{color:#3b82f6;text-decoration:none;}}
.topbar{{background:#fff;border-bottom:1px solid #e2e8f0;padding:0 32px;height:56px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 1px 3px rgba(0,0,0,0.04);}}
.logo{{display:flex;align-items:center;gap:10px;}}
.logo-mark{{width:32px;height:32px;background:#1e3a5f;border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;color:#fff;}}
.logo-name{{font-size:15px;font-weight:700;color:#1e293b;}}
.pill{{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:999px;padding:4px 12px;font-size:11px;color:#64748b;}}
.btn-dark{{background:#1e3a5f;color:#fff;border:none;border-radius:8px;padding:7px 16px;font-size:12px;font-weight:600;cursor:pointer;}}
.page-header{{background:#fff;border-bottom:1px solid #e2e8f0;padding:24px 32px;}}
.page-header-inner{{max-width:1200px;margin:0 auto;display:flex;justify-content:space-between;align-items:flex-start;}}
.page-title{{font-size:22px;font-weight:700;color:#0f172a;letter-spacing:-0.5px;}}
.page-sub{{font-size:13px;color:#64748b;margin-top:4px;}}
.bc{{font-size:11px;color:#94a3b8;margin-bottom:8px;}}
.main{{max-width:1200px;margin:0 auto;padding:28px 32px;}}
.sev-strip{{border-radius:12px;padding:20px 24px;margin-bottom:24px;display:flex;align-items:center;gap:24px;border:1px solid;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
thead tr{{background:#f8fafc;}}
th{{padding:10px 14px;text-align:left;font-size:11px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid #e2e8f0;}}
td{{padding:12px 14px;border-bottom:1px solid #f1f5f9;color:#334155;vertical-align:middle;}}
tr:last-child td{{border-bottom:none;}}
tbody tr:hover td{{background:#f8fafc;}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;margin-bottom:24px;}}
.card-hdr{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;flex-wrap:wrap;gap:12px;}}
.card-title{{font-size:15px;font-weight:600;color:#0f172a;}}
.card-sub{{font-size:12px;color:#94a3b8;margin-top:3px;}}
.footer{{background:#fff;border-top:1px solid #e2e8f0;padding:16px 32px;display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#94a3b8;margin-top:8px;}}
</style>
</head>
<body>

<nav class="topbar">
  <div class="logo">
    <div class="logo-mark">SE</div>
    <span class="logo-name">SynthEdge</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <span class="pill">v0.1.0</span>
    <span class="pill">{timestamp}</span>
    <a href="https://github.com/Juzt-nik/SynthEdge" target="_blank">
      <button class="btn-dark">GitHub ↗</button>
    </a>
  </div>
</nav>

<div class="page-header">
  <div class="page-header-inner">
    <div>
      <p class="bc">Reports &rsaquo; Gap Analysis</p>
      <h1 class="page-title">{dataset_name}</h1>
      <p class="page-sub">{n_rows:,} rows &middot; {n_features} features &middot; {pos_rate}% positive class</p>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.06em;">Severity</div>
      <div style="font-size:24px;font-weight:800;color:{sev_color};">{sev_level}</div>
    </div>
  </div>
</div>

<main class="main">

  <div class="sev-strip" style="background:{sev_bg};border-color:{sev_color}33;color:{sev_color};">
    <div style="text-align:center;min-width:80px;">
      <div style="font-size:36px;font-weight:800;letter-spacing:-1.5px;line-height:1;">{sev_score:.2f}</div>
      <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;opacity:0.7;">severity score</div>
    </div>
    <div style="width:1px;height:56px;background:currentColor;opacity:0.15;"></div>
    <div style="flex:1;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;letter-spacing:0.06em;padding:4px 10px;border-radius:999px;border:1px solid {sev_color}55;color:{sev_color};">{sev_icon} {sev_level}</span>
        <span style="font-size:12px;opacity:0.7;">{sev_sub}</span>
      </div>
      <div style="font-size:15px;font-weight:600;margin-bottom:4px;">{rec_text.split(".")[0]}.</div>
      <div style="font-size:13px;opacity:0.8;max-width:640px;">{"".join(rec_text.split(".")[1:]).strip()}</div>
      <div style="display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;margin-top:8px;color:{help_color};">
        <span style="font-size:8px;">&#9679;</span> SynthEdge {help_text}
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:24px;">
    {metric_html}
  </div>

  {chart_section}

  <div class="card">
    <div class="card-hdr">
      <div>
        <p class="card-title">Gap voxel map</p>
        <p class="card-sub">Top {len(top_voxels)} sparse regions ranked by gap score</p>
      </div>
      <span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:999px;padding:4px 12px;font-size:11px;font-weight:600;color:#64748b;">3D local density scan</span>
    </div>
    <table>
      <thead><tr><th>#</th><th>Voxel</th><th style="text-align:center;">Observed</th><th style="text-align:center;">Expected</th><th style="text-align:center;">Positives</th><th style="text-align:center;">Sparsity</th><th style="text-align:center;">Entropy</th><th>Gap score</th></tr></thead>
      <tbody>{voxel_rows}</tbody>
    </table>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px;">
    <div class="card" style="margin-bottom:0;">
      <div class="card-hdr">
        <div>
          <p class="card-title">Synthesis summary</p>
          <p class="card-sub">{"CTGAN + Gaussian fallback" if ctgan_used else "Gaussian sampling"}</p>
        </div>
        <span style="background:{"#dbeafe" if ctgan_used else "#ede9fe"};color:{"#1d4ed8" if ctgan_used else "#6d28d9"};border-radius:999px;padding:4px 12px;font-size:11px;font-weight:600;">{"CTGAN" if ctgan_used else "Gaussian"}</span>
      </div>
      <table>
        <thead><tr><th>Voxel</th><th>Method</th><th>Added</th></tr></thead>
        <tbody>{synth_rows}</tbody>
      </table>
      <div style="margin-top:16px;padding-top:16px;border-top:1px solid #f1f5f9;display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:12px;color:#64748b;">Total samples injected</span>
        <span style="font-size:18px;font-weight:800;color:#22c55e;">+{total_added}</span>
      </div>
    </div>

    <div class="card" style="margin-bottom:0;">
      <div class="card-hdr">
        <div>
          <p class="card-title">Severity signals</p>
          <p class="card-sub">Raw inputs to the classifier</p>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">{signal_cards}</div>
      <div style="margin-top:16px;padding-top:16px;border-top:1px solid #f1f5f9;display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:12px;color:#64748b;">Final severity score</span>
        <span style="font-size:18px;font-weight:800;color:{sev_color};">{sev_score:.4f}</span>
      </div>
    </div>
  </div>

</main>

<footer class="footer">
  <span>SynthEdge v0.1.0 &mdash; Diagnosis-first synthetic data augmentation</span>
  <div style="display:flex;gap:16px;">
    <a href="https://github.com/Juzt-nik/SynthEdge">GitHub</a>
    <a href="https://pypi.org/project/synthedge">PyPI</a>
  </div>
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>{chart_js}</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("[SynthEdge] Report saved: " + output_path)
    return output_path
