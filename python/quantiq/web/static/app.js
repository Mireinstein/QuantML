// QuantIQ dashboard frontend. No CDN dependencies -- charts are hand-rolled
// inline SVG (a couple of small helpers below) rather than pulling in a
// charting library, matching the rest of this project's "runs offline, no
// unnecessary dependencies" standard.

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toFixed(digits);
}

function statTile(label, value, cls) {
  const div = document.createElement("div");
  div.className = "stat-tile" + (cls ? " " + cls : "");
  div.innerHTML = `<span class="label">${label}</span><span class="value">${value}</span>`;
  return div;
}

function renderSummaryTiles(container, summary, groupClass, prefix) {
  container.innerHTML = "";
  const label = document.createElement("div");
  label.style.cssText = "font-size:0.75rem;color:var(--muted);width:100%;margin-top:0.4rem;";
  label.textContent = prefix;
  container.appendChild(label);
  const row = document.createElement("div");
  row.className = "stat-row";
  row.appendChild(statTile("Sharpe", fmt(summary.sharpe), groupClass));
  row.appendChild(statTile("CAGR", (summary.cagr * 100).toFixed(2) + "%", groupClass));
  row.appendChild(statTile("Max DD", (summary.max_drawdown * 100).toFixed(2) + "%", groupClass));
  row.appendChild(statTile("Win rate", (summary.win_rate * 100).toFixed(1) + "%", groupClass));
  row.appendChild(statTile("Final equity", fmt(summary.final_equity), groupClass));
  container.appendChild(row);
}

// --- Small reusable SVG line chart: builds a <polyline> per series from
// normalized (x,y) points, plus min/max y-axis labels and gridlines. -------
function lineChart(container, series, opts = {}) {
  const W = opts.width || 900;
  const H = opts.height || 220;
  const pad = { l: 50, r: 15, t: 10, b: 20 };
  container.innerHTML = "";

  const allYs = series.flatMap((s) => s.values).filter((v) => v !== null && !Number.isNaN(v));
  if (!allYs.length) {
    container.textContent = "No data.";
    return;
  }
  const yMin = Math.min(...allYs);
  const yMax = Math.max(...allYs);
  const yRange = yMax - yMin || 1;
  const n = series[0].values.length;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });

  const xAt = (i) => pad.l + (i / Math.max(n - 1, 1)) * (W - pad.l - pad.r);
  const yAt = (v) => H - pad.b - ((v - yMin) / yRange) * (H - pad.t - pad.b);

  // gridlines + y labels (min/mid/max)
  [0, 0.5, 1].forEach((f) => {
    const val = yMin + f * yRange;
    const y = yAt(val);
    svg.appendChild(el("line", { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: "var(--grid)", "stroke-width": 1 }));
    const t = el("text", { x: 4, y: y + 4, fill: "var(--muted)", "font-size": 10 });
    t.textContent = opts.yfmt ? opts.yfmt(val) : val.toFixed(2);
    svg.appendChild(t);
  });

  const colors = opts.colors || ["var(--accent)", "var(--accent2)", "var(--green)"];
  series.forEach((s, si) => {
    const pts = s.values
      .map((v, i) => (v === null || Number.isNaN(v) ? null : `${xAt(i)},${yAt(v)}`))
      .filter(Boolean)
      .join(" ");
    svg.appendChild(el("polyline", { points: pts, fill: "none", stroke: colors[si % colors.length], "stroke-width": 1.75 }));
  });

  // x-axis labels: first / last category
  if (opts.xlabels && opts.xlabels.length) {
    const first = el("text", { x: pad.l, y: H - 4, fill: "var(--muted)", "font-size": 10 });
    first.textContent = opts.xlabels[0];
    svg.appendChild(first);
    const last = el("text", { x: W - pad.r, y: H - 4, fill: "var(--muted)", "font-size": 10, "text-anchor": "end" });
    last.textContent = opts.xlabels[opts.xlabels.length - 1];
    svg.appendChild(last);
  }

  container.appendChild(svg);
}

// --- Small reusable SVG bar chart. -----------------------------------------
function barChart(container, values, opts = {}) {
  const W = opts.width || 900;
  const H = opts.height || 200;
  const pad = { l: 50, r: 15, t: 10, b: 24 };
  container.innerHTML = "";

  if (!values.length) {
    container.textContent = "No data.";
    return;
  }
  const yMax = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });

  const zeroY = opts.zeroCentered ? pad.t + (H - pad.t - pad.b) / 2 : H - pad.b;
  const usableH = opts.zeroCentered ? (H - pad.t - pad.b) / 2 : H - pad.t - pad.b;

  svg.appendChild(el("line", { x1: pad.l, x2: W - pad.r, y1: zeroY, y2: zeroY, stroke: "var(--grid)", "stroke-width": 1 }));

  const bw = (W - pad.l - pad.r) / values.length;
  values.forEach((v, i) => {
    const h = (Math.abs(v) / yMax) * usableH;
    const x = pad.l + i * bw + bw * 0.15;
    const y = v >= 0 ? zeroY - h : zeroY;
    svg.appendChild(
      el("rect", {
        x,
        y,
        width: bw * 0.7,
        height: Math.max(h, 1),
        fill: v >= 0 ? "var(--accent)" : "var(--red)",
        rx: 1,
      })
    );
  });

  const maxLabel = el("text", { x: 4, y: pad.t + 8, fill: "var(--muted)", "font-size": 10 });
  maxLabel.textContent = (opts.yfmt ? opts.yfmt(yMax) : yMax.toFixed(2));
  svg.appendChild(maxLabel);

  container.appendChild(svg);
}

// --- Fetch helpers -----------------------------------------------------
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `${url} -> ${r.status}`);
  }
  return r.json();
}

// --- Section loaders -----------------------------------------------------
async function loadBacktest() {
  const d = await getJSON("/api/dashboard");
  lineChart(
    document.getElementById("backtest-chart"),
    [
      { values: d.baseline_equity },
      { values: d.overlay_equity },
    ],
    { xlabels: [d.dates[0], d.dates[d.dates.length - 1]] }
  );
  renderSummaryTiles(document.getElementById("backtest-stats-baseline"), d.baseline_summary, "group", "Baseline (MA crossover)");
  renderSummaryTiles(document.getElementById("backtest-stats-overlay"), d.overlay_summary, "overlay", "RAG overlay (MA + sentiment)");
}

async function loadWalkForward() {
  const d = await getJSON("/api/walk-forward?n_folds=5");
  barChart(document.getElementById("wf-chart"), d.fold_sharpe, { zeroCentered: true });
  const row = document.getElementById("wf-stats");
  row.innerHTML = "";
  row.appendChild(statTile("Mean Sharpe", fmt(d.summary.mean_sharpe)));
  row.appendChild(statTile("Std Sharpe", fmt(d.summary.std_sharpe)));
  row.appendChild(statTile("Worst fold Sharpe", fmt(d.summary.worst_fold_sharpe)));
  row.appendChild(statTile("Worst fold DD", (d.summary.worst_fold_drawdown * 100).toFixed(2) + "%"));
}

async function loadVar() {
  const d = await getJSON("/api/var?horizon_days=10&confidence=0.95");
  const row = document.getElementById("var-stats");
  row.innerHTML = "";
  row.appendChild(statTile("VaR", (d.var * 100).toFixed(2) + "%"));
  row.appendChild(statTile("CVaR", (d.cvar * 100).toFixed(2) + "%"));
  row.appendChild(statTile("Horizon", `${d.horizon_days}d`));
  row.appendChild(statTile("Confidence", `${(d.confidence * 100).toFixed(0)}%`));
  row.appendChild(statTile("Sims", d.n_sims));
}

async function loadVolatility() {
  const d = await getJSON("/api/volatility");
  lineChart(
    document.getElementById("vol-chart"),
    [{ values: d.conditional_vol }],
    { xlabels: [d.dates[0], d.dates[d.dates.length - 1]], colors: ["var(--accent)"], yfmt: (v) => (v * 100).toFixed(1) + "%" }
  );
  const row = document.getElementById("vol-stats");
  row.innerHTML = "";
  row.appendChild(statTile("GARCH forecast vol", (d.forecast_vol * 100).toFixed(2) + "%"));
  row.appendChild(statTile("Naive 20d rolling vol", (d.naive_vol * 100).toFixed(2) + "%"));
}

async function loadRiskLimits() {
  const d = await getJSON("/api/risk-limits?max_drawdown=0.10");
  lineChart(
    document.getElementById("risk-chart"),
    [{ values: d.limited_equity }, { values: d.unlimited_equity }],
    { xlabels: [d.dates[0], d.dates[d.dates.length - 1]] }
  );
  const note = document.getElementById("risk-note");
  if (d.breach_type) {
    note.textContent = `Kill switch tripped: ${d.breach_type} breached on ${d.breach_date}. Trading halted for the rest of the run.`;
    note.className = "note";
  } else {
    note.textContent = "Not breached over this run.";
    note.className = "note ok";
  }
}

async function loadMlSignal() {
  const metaRow = document.getElementById("ml-meta");
  const statsRow = document.getElementById("ml-stats");
  metaRow.innerHTML = "";
  statsRow.innerHTML = "";
  try {
    const d = await getJSON("/api/ml-signal");
    metaRow.appendChild(statTile("Model", d.model_type));
    metaRow.appendChild(statTile("Version", d.version));
    metaRow.appendChild(statTile("Held-out AUC", fmt(d.held_out_auc)));
    metaRow.appendChild(statTile("Trained on", d.data_source));

    const bt = d.held_out_backtest;
    statsRow.appendChild(statTile("Held-out Sharpe", fmt(bt.sharpe)));
    statsRow.appendChild(statTile("Held-out CAGR", (bt.cagr * 100).toFixed(2) + "%"));
    statsRow.appendChild(statTile("Held-out Max DD", (bt.max_drawdown * 100).toFixed(2) + "%"));
    statsRow.appendChild(statTile("Held-out win rate", (bt.win_rate * 100).toFixed(1) + "%"));
  } catch (e) {
    metaRow.innerHTML = `<div class="note">No trained model yet -- run \`python -m quantiq.ml.train\` from python/, then reload.</div>`;
  }
}

async function predictMlSignal(ticker) {
  const resultRow = document.getElementById("ml-predict-result");
  resultRow.innerHTML = `<div class="note">Predicting...</div>`;
  try {
    const d = await getJSON(`/api/ml-signal/predict?ticker=${encodeURIComponent(ticker)}`);
    resultRow.innerHTML = "";
    resultRow.appendChild(statTile("Ticker", d.ticker));
    resultRow.appendChild(statTile("As of", d.as_of_date));
    resultRow.appendChild(statTile("Last close", "$" + fmt(d.last_close, 2)));
    resultRow.appendChild(statTile("P(up)", (d.predicted_proba_up * 100).toFixed(1) + "%"));
    resultRow.appendChild(statTile("Suggested position", fmt(d.suggested_position, 2)));
  } catch (e) {
    resultRow.innerHTML = `<div class="note">${e.message}</div>`;
  }
}

// --- Boot ------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  loadBacktest().catch(console.error);
  loadWalkForward().catch(console.error);
  loadVar().catch(console.error);
  loadVolatility().catch(console.error);
  loadRiskLimits().catch(console.error);
  loadMlSignal().catch(console.error);

  document.getElementById("ml-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const ticker = document.getElementById("ml-ticker").value.trim().toUpperCase();
    if (ticker) predictMlSignal(ticker).catch(console.error);
  });
});
