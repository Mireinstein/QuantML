// QuantML dashboard frontend. No CDN dependencies -- charts are hand-rolled
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

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// Company-name -> ticker autocomplete (e.g. typing "apple" offers AAPL),
// backed by /api/tickers/search. Selecting a result just replaces the
// input's value with the ticker symbol -- every consumer of these fields
// (predict, explain, trade) already expects a plain ticker string, so no
// other wiring needed.
function attachTickerAutocomplete(inputEl, resultsEl) {
  const runSearch = debounce(async (query) => {
    if (!query.trim()) {
      resultsEl.classList.remove("open");
      resultsEl.innerHTML = "";
      return;
    }
    try {
      const d = await getJSON(`/api/tickers/search?q=${encodeURIComponent(query)}`);
      if (!d.results.length) {
        resultsEl.innerHTML = `<div class="ac-empty">No matches</div>`;
      } else {
        resultsEl.innerHTML = d.results
          .map(
            (r) =>
              `<div class="ac-result" data-symbol="${escapeHtml(r.symbol)}"><span class="symbol">${escapeHtml(r.symbol)}</span><span class="name">${escapeHtml(r.name)}</span></div>`
          )
          .join("");
      }
      resultsEl.classList.add("open");
    } catch (e) {
      resultsEl.classList.remove("open");
    }
  }, 250);

  inputEl.addEventListener("input", () => runSearch(inputEl.value));
  inputEl.addEventListener("focus", () => {
    if (resultsEl.innerHTML) resultsEl.classList.add("open");
  });
  // mousedown, not click: fires before the input's blur event, so
  // selecting a result isn't swallowed by blur closing the dropdown first.
  resultsEl.addEventListener("mousedown", (e) => {
    const item = e.target.closest(".ac-result");
    if (!item || !item.dataset.symbol) return;
    inputEl.value = item.dataset.symbol;
    resultsEl.classList.remove("open");
    resultsEl.innerHTML = "";
  });
  document.addEventListener("click", (e) => {
    if (e.target !== inputEl && !resultsEl.contains(e.target)) {
      resultsEl.classList.remove("open");
    }
  });
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

async function postJSON(url) {
  const r = await fetch(url, { method: "POST" });
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
    metaRow.innerHTML = `<div class="note">No trained model yet -- run \`python -m quantml.ml.train\` from python/, then reload.</div>`;
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

async function loadExplain(ticker) {
  const container = document.getElementById("explain-chart");
  container.innerHTML = `<div class="note">Computing permutation importance...</div>`;
  try {
    const d = await getJSON(`/api/ml-signal/explain?ticker=${encodeURIComponent(ticker)}`);
    barChart(
      container,
      d.importances.map((fi) => fi.importance_mean),
      { yfmt: (v) => v.toFixed(3) }
    );
    const labelsRow = document.createElement("div");
    labelsRow.className = "stat-row";
    d.importances.forEach((fi) => labelsRow.appendChild(statTile(fi.feature, fi.importance_mean.toFixed(3))));
    container.appendChild(labelsRow);
  } catch (e) {
    container.innerHTML = `<div class="note">${e.message}</div>`;
  }
}

async function runTrade(ticker, qtyPerUnit) {
  const resultRow = document.getElementById("trade-result");
  const confirmed = window.confirm(
    `Submit a REAL order to your Alpaca PAPER account for ${ticker}, sized to the model's current prediction (up to ${qtyPerUnit} shares)? This is fake money, but it's a real order, not a dry run.`
  );
  if (!confirmed) return;

  resultRow.innerHTML = `<div class="note">Submitting...</div>`;
  try {
    const d = await postJSON(
      `/api/trade/run?ticker=${encodeURIComponent(ticker)}&qty_per_unit=${encodeURIComponent(qtyPerUnit)}`
    );
    resultRow.innerHTML = "";
    resultRow.appendChild(statTile("Ticker", d.ticker));
    resultRow.appendChild(statTile("Last close", "$" + fmt(d.last_close, 2)));
    resultRow.appendChild(statTile("Current shares", d.current_shares));
    resultRow.appendChild(statTile("Target shares", d.target_shares));
    resultRow.appendChild(
      statTile("Order", d.delta === 0 ? "already at target" : (typeof d.order === "object" ? `${d.order.side} ${d.order.qty} (${d.order.status})` : d.order))
    );
  } catch (e) {
    resultRow.innerHTML = `<div class="note">${e.message}</div>`;
  }
}

function tableFrom(rows, columns) {
  // columns: [[header, key, formatter?]]
  if (!rows.length) return `<div class="note">No data yet.</div>`;
  const head = columns.map(([label]) => `<th>${label}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${columns.map(([, key, fmtFn]) => `<td>${fmtFn ? fmtFn(row[key], row) : (row[key] ?? "-")}</td>`).join("")}</tr>`
    )
    .join("");
  return `<table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

async function loadBotStatus() {
  const tile = document.getElementById("bot-status-tile");
  try {
    const d = await getJSON("/api/autonomous/status");
    if (!d.configured) {
      tile.innerHTML = `<span class="label">Bot status</span><span class="value">not deployed here</span>`;
      document.getElementById("bot-start-btn").disabled = true;
      document.getElementById("bot-stop-btn").disabled = true;
      return;
    }
    tile.innerHTML = `<span class="label">Bot status</span><span class="value">${d.running ? "running" : "stopped"}</span>`;
    tile.className = "stat-tile" + (d.running ? " group" : "");
  } catch (e) {
    tile.innerHTML = `<span class="label">Bot status</span><span class="value">unknown</span>`;
  }
}

async function setBotRunning(shouldRun) {
  const note = document.getElementById("bot-control-note");
  note.textContent = shouldRun ? "Starting..." : "Stopping...";
  try {
    const r = await fetch(shouldRun ? "/api/autonomous/start" : "/api/autonomous/stop", { method: "POST" });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `${r.status}`);
    }
    note.textContent = "";
    loadBotStatus().catch(console.error);
  } catch (e) {
    note.textContent = e.message;
  }
}

async function loadEquity() {
  const chartContainer = document.getElementById("equity-chart");
  const statsRow = document.getElementById("equity-stats");
  try {
    const d = await getJSON("/api/autonomous/equity?period=1M&timeframe=1D");
    lineChart(
      chartContainer,
      [{ values: d.equity }],
      { xlabels: [d.timestamps[0]?.slice(0, 10), d.timestamps[d.timestamps.length - 1]?.slice(0, 10)], colors: ["var(--accent)"], yfmt: (v) => "$" + v.toFixed(0) }
    );
    statsRow.innerHTML = "";
    const latestPL = d.profit_loss[d.profit_loss.length - 1] ?? 0;
    const latestPLPct = d.profit_loss_pct[d.profit_loss_pct.length - 1] ?? 0;
    statsRow.appendChild(statTile("Base value", "$" + fmt(d.base_value, 2)));
    statsRow.appendChild(statTile("Current equity", "$" + fmt(d.equity[d.equity.length - 1], 2)));
    statsRow.appendChild(statTile("P/L", "$" + fmt(latestPL, 2), latestPL >= 0 ? "group" : "overlay"));
    statsRow.appendChild(statTile("P/L %", (latestPLPct * 100).toFixed(3) + "%", latestPL >= 0 ? "group" : "overlay"));
  } catch (e) {
    chartContainer.innerHTML = `<div class="note">${e.message === "Failed to fetch" ? "Risk model / paper trading not configured on this server." : e.message}</div>`;
    statsRow.innerHTML = "";
  }
}

async function loadTrades() {
  const container = document.getElementById("trades-table");
  try {
    const d = await getJSON("/api/autonomous/trades?limit=50");
    container.innerHTML = tableFrom(d.trades, [
      ["Submitted", "submitted_at", (v) => v.slice(0, 19).replace("T", " ")],
      ["Side", "side"],
      ["Qty", "qty"],
      ["Status", "status"],
      ["Filled qty", "filled_qty"],
      ["Filled @", "filled_avg_price", (v) => (v ? "$" + Number(v).toFixed(2) : "pending")],
    ]);
  } catch (e) {
    container.innerHTML = `<div class="note">${e.message}</div>`;
  }
}

async function loadGenerations() {
  const container = document.getElementById("generations-table");
  try {
    const d = await getJSON("/api/autonomous/generations");
    const rows = d.generations
      .slice()
      .reverse()
      .map((g) => ({
        timestamp: (g.timestamp || "").slice(0, 19).replace("T", " "),
        outcome: g.event === "model_promoted" ? "promoted" : "rejected",
        model_type: g.model_type ?? g.candidate_model_type ?? "-",
        auc: g.auc ?? g.candidate_auc,
        sharpe: g.sharpe ?? g.candidate_sharpe,
        reasons: (g.reasons || []).join("; "),
      }));
    container.innerHTML = tableFrom(rows, [
      ["When", "timestamp"],
      ["Outcome", "outcome"],
      ["Model", "model_type"],
      ["AUC", "auc", (v) => fmt(v, 3)],
      ["Sharpe", "sharpe", (v) => fmt(v, 3)],
      ["Why rejected", "reasons"],
    ]);
  } catch (e) {
    container.innerHTML = `<div class="note">${e.message}</div>`;
  }
}

function orderText(order) {
  // `order` from a cycle log entry is one of: null (no order needed that
  // cycle), a string ("DRY RUN -- ..."), {id, side, qty, status} (Alpaca
  // accepted it), or {error} (Alpaca rejected it -- e.g. the wash-trade
  // guard firing because a previous order is still open/unfilled).
  if (!order) return "no order";
  if (typeof order === "string") return order;
  if (order.error) return `rejected: ${order.error}`;
  return `${order.side} ${order.qty} (${order.status})`;
}

async function loadAutonomousActivity() {
  const container = document.getElementById("autonomous-activity");
  try {
    const d = await getJSON("/api/autonomous/activity?n=20");
    if (!d.activity.length) {
      container.innerHTML = `<div class="note">Not running on this machine right now -- start it with \`python -m quantml.autonomous --ticker AAPL\` from python/.</div>`;
      return;
    }
    const rows = d.activity
      .slice()
      .reverse()
      .map((e) => {
        if (e.event === "cycle") {
          return `<div class="stat-tile"><span class="label">${e.replayed_day} (gen ${e.generation})</span><span class="value">P(up)=${fmt(e.predicted_proba_up, 2)} pos=${fmt(e.suggested_position, 2)} -- ${orderText(e.order)}</span></div>`;
        }
        if (e.event === "model_promoted") {
          return `<div class="stat-tile group"><span class="label">Model promoted (gen ${e.generation})</span><span class="value">${e.model_type}, AUC ${fmt(e.auc, 3)}, Sharpe ${fmt(e.sharpe, 3)}</span></div>`;
        }
        if (e.event === "retrain_rejected") {
          return `<div class="stat-tile"><span class="label">Retrain rejected</span><span class="value">${e.reasons.join("; ")}</span></div>`;
        }
        if (e.event === "cycle_error") {
          return `<div class="stat-tile"><span class="label">Cycle error</span><span class="value">${e.error}</span></div>`;
        }
        return `<div class="stat-tile"><span class="label">${e.event}</span><span class="value"></span></div>`;
      })
      .join("");
    container.innerHTML = `<div class="stat-row">${rows}</div>`;
  } catch (e) {
    container.innerHTML = `<div class="note">${e.message}</div>`;
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
  loadAutonomousActivity().catch(console.error);
  setInterval(() => loadAutonomousActivity().catch(console.error), 15000);

  loadEquity().catch(console.error);
  loadTrades().catch(console.error);
  loadGenerations().catch(console.error);
  loadBotStatus().catch(console.error);
  setInterval(() => {
    loadEquity().catch(console.error);
    loadTrades().catch(console.error);
    loadGenerations().catch(console.error);
    loadBotStatus().catch(console.error);
  }, 30000);

  document.getElementById("bot-start-btn").addEventListener("click", () => setBotRunning(true));
  document.getElementById("bot-stop-btn").addEventListener("click", () => setBotRunning(false));

  loadExplain("AAPL").catch(console.error);

  document.getElementById("ml-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const ticker = document.getElementById("ml-ticker").value.trim().toUpperCase();
    if (ticker) {
      predictMlSignal(ticker).catch(console.error);
      loadExplain(ticker).catch(console.error);
    }
  });

  document.getElementById("trade-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const ticker = document.getElementById("trade-ticker").value.trim().toUpperCase();
    const qty = parseInt(document.getElementById("trade-qty").value, 10) || 10;
    if (ticker) runTrade(ticker, qty).catch(console.error);
  });

  attachTickerAutocomplete(document.getElementById("ml-ticker"), document.getElementById("ml-ticker-results"));
  attachTickerAutocomplete(document.getElementById("trade-ticker"), document.getElementById("trade-ticker-results"));
});
