# QuantML

An applied ML platform for systematic trading: a feature engineering +
model training pipeline (scikit-learn and a PyTorch GRU), evaluated with
no-lookahead walk-forward validation and a standalone quality gate,
driving a backtester, a research dashboard, and paper trading execution.

No real capital is ever at risk. Everything runs on synthetic price data
and a sample news corpus by default. Two pieces optionally connect to
real external services: `--real-data` pulls real historical prices from
Yahoo Finance (read-only), and `paper_trading.py`/`paper_runner.py`
submit orders to Alpaca's paper-trading sandbox — real market prices,
simulated cash, hard-coded to the paper API endpoint. Not financial
advice.

## Components

### 1. `python/quantml/ml/` — the ML pipeline

**Features** (`features.py`): technical indicators from OHLCV — 1/5/10-day
returns, 20-day rolling volatility, a 10/50-day moving-average ratio,
14-day RSI, 5-day volume change, daily high-low range. Every feature at
row *t* only uses data through day *t*'s close. Label is next-day
direction; the last row is dropped (no next day to label).

**Models** (`model.py`), both behind one `predict_proba(features)`
interface:

- `SklearnSignalModel` — logistic regression or `HistGradientBoostingClassifier`.
- `TorchSignalModel` — a PyTorch GRU reading a 20-day rolling sequence of
  features. Rows before the first full window return a neutral 0.5.

**Training** (`train.py`): a single chronological train/test split (price
history is sequential, so a random split would leak future information
through overlapping rolling-window features). Trains all three models,
evaluates each by held-out ROC-AUC and by running the model's signal
through the backtester on the same held-out period. Selection is by
held-out backtest Sharpe, not AUC — a model can have a mediocre AUC and
still be a good trading signal. Logged to MLflow (local SQLite backend).

**Evaluation gate** (`eval_harness.py`): separate from the pytest unit
tests, checks the model *artifact on disk* is good enough to ship.
Re-evaluates against data disjoint from training — a different synthetic
seed for a synthetic-trained model, a freshly re-fetched real series for
a real-data model. Fails (non-zero exit) if AUC or held-out Sharpe drop
below fixed floors or regress past a tolerance versus the recorded
baseline.

**Registry** (`registry.py`): loads whichever model `train.py` most
recently selected.

**Explainability** (`explain.py`, `GET /api/ml-signal/explain`):
permutation importance on real recent data — for each feature, shuffle
its values and measure the drop in the live model's held-out AUC.
Model-agnostic (only calls `predict_proba`), no new dependency.

### 2. `python/quantml/strategies.py::MLSignalStrategy`

Same `Strategy` interface as the rule-based strategies —
`positions(prices) -> pd.Series` in `[-1, 1]` — so the trained model
plugs into the same backtester, walk-forward evaluator, risk-limit kill
switch, and paper-trading runner. `P(next day up)` maps to a position
sized by confidence rather than a hard threshold. Positions are shifted
by one day, so the position held on day *t* only used information known
before day *t* opened.

### 3. `python/quantml/` — backtesting engine

Vectorized backtester: strategies emit a target position per bar, the
engine nets out transaction costs into an equity curve. Metrics: Sharpe,
CAGR, max drawdown, win rate, Monte Carlo VaR/CVaR (historical bootstrap).

**Walk-forward evaluation** (`walk_forward.py`) splits the backtest into
sequential out-of-sample folds instead of one full-period Sharpe.

**Risk management** (`risk.py`): `RiskLimits` + `apply_risk_limits`
enforce a max position size and a max-drawdown/max-daily-loss kill
switch that forces the strategy flat once breached.

**Volatility modeling** (`volatility.py`): ARIMA (statsmodels) for the
conditional mean, GARCH(1,1) (`arch`) for volatility clustering.

Also includes two baseline technical strategies (moving-average crossover,
mean reversion) and a `SignalOverlayStrategy` that blends a base
strategy's position with an external signal series.

### 4. `python/quantml/rag/` — retrieval-augmented signal layer

TF-IDF retrieval over a small bundled corpus of sample financial
documents, turned into a per-day sentiment signal fed into
`SignalOverlayStrategy`. Three scoring backends (`--sentiment-backend`):
**lexicon** (deterministic keyword scoring), **llm** (a local Ollama
model, structured output validated with pydantic), and **finetuned** (the
LoRA-fine-tuned classifier below) — `llm` and `finetuned` fall back to
the lexicon scorer if unreachable.

### 4b. `python/quantml/finetune/` — LLM fine-tuning

LoRA fine-tunes `distilbert-base-uncased` for 3-class financial sentiment
(Bearish/Bullish/Neutral) on
[`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
(~9.5k labeled tweets). LoRA (`peft`) trains low-rank matrices injected
into DistilBERT's attention projections (`q_lin`, `v_lin`) instead of the
full ~66M parameters — 1.09% of parameters trainable (740,355 of
67,696,134). Same train/eval/gate discipline as the trading-signal model:
`train.py` fits on a 3,000-tweet subsample (2 epochs, CPU, ~2 minutes) and
logs to MLflow; `eval_harness.py` gates on a fresh held-out slice.

Result: 70.9% accuracy / 0.43 macro-F1 on held-out data, against a ~65%
majority-class baseline.

### 5. `python/quantml/web/` — research dashboard

A FastAPI app (`app.py`): equity curves, walk-forward folds, VaR/CVaR,
GARCH volatility, the risk-limit kill switch, the ML signal's held-out
performance, live prediction, feature importance, and an activity feed
for the autonomous loop. Endpoints reuse `cli.py`'s recipe and recompute
on every call rather than caching. No CDN dependencies — charts in
`static/app.js` are a small hand-rolled inline-SVG helper.

`/api/ml-signal` reports the model's recorded chronological held-out
performance rather than re-scoring against the dashboard's own demo data
(which shares an RNG seed with training data, so re-scoring against it
would show in-sample performance). `/api/ml-signal/predict` does live
inference on today's real market data instead.

### 6. Real market data + paper trading

`data.py::load_real_ohlcv(ticker, period, interval)` pulls real daily
OHLCV from Yahoo Finance via `yfinance`, normalized to the same column
contract as the synthetic generator.

`paper_trading.py` is a REST client for Alpaca's paper trading API. The
base URL is hard-coded to the paper endpoint, not configurable via env
var or argument. Credentials load from a gitignored `python/.env`; every
call raises `PaperTradingError` rather than silently no-opping.

`paper_runner.py` wires it together: `python -m quantml.paper_runner
--ticker AAPL --strategy ml_signal` pulls real market data, runs the
model, compares its target position against the account's actual paper
position, and submits whatever order closes the gap (`--dry-run` to just
print it).

The dashboard has an on-demand version: a "Run trade now" button
(`POST /api/trade/run`) that does the same rebalance from the browser.
It isn't exposed on the live Azure deployment — that container has no
Alpaca credentials configured, so calls there fail closed with a 502.
Don't add Alpaca credentials to the Azure deployment without adding auth
first.

### Autonomous continuous-learning loop (`autonomous.py`, local only)

    python -m quantml.autonomous --ticker AAPL

Trades every cycle and periodically retrains the model on what it's
learned since, gated the same way as a manual retrain.

A real daily bar only updates once per trading day, so a loop waiting on
live data would sit idle most of the time, especially overnight. Instead
this loop replays real historical data the model hasn't been evaluated
against yet, one day at a time, at an accelerated cadence
(`--cycle-seconds`, default 90s per day).

Each cycle: get the model's prediction for the next replayed day, size
and submit a real order to the Alpaca paper account, then — since this is
a replay of known history — learn that day's actual realized outcome.
Every `--retrain-every` cycles (default 15), it retrains on an expanding
window of the same historical series, evaluates the candidate through the
same gate as `ml/eval_harness.py`, and only promotes it if it passes. A
candidate that fails is discarded; the previously-promoted model keeps
serving.

Each cycle also runs a feature drift check (same z-score approach as
TenantIQ's `ml/serve.py` `/monitoring` endpoint) against the training
distribution.

Every cycle is appended to `ml/autonomous_log.jsonl` (gitignored,
machine-local) and shown on the dashboard's "Live autonomous trading"
panel if the dashboard is running on the same machine. A single cycle's
failure is logged and the loop moves on rather than dying.

**Bot trading performance panel**: reports on the loop, it doesn't
require operating it. Three endpoints, all reading straight from Alpaca
rather than reconstructing state locally:

- `GET /api/autonomous/equity` — the paper account's real equity over
  time (`paper_trading.py::get_portfolio_history`), stays flat until
  orders actually fill.
- `GET /api/autonomous/trades` — real order history
  (`paper_trading.py::list_orders`): side, qty, status, and whether it's
  actually filled (`filled_qty`/`filled_avg_price`) or still pending.
- `GET /api/autonomous/generations` — every retrain the loop attempted,
  promoted or rejected, with the AUC/Sharpe that decided it (derived from
  the local activity log, the only record of promotion decisions) — shows
  whether the model has actually been improving.

## Deployment

**Docker** (`python/Dockerfile`): trains the model at image build time
(`RUN python3 -m quantml.ml.train`), not at container startup. Training
loads PyTorch + transformers + scikit-learn together, which needs more
memory than the serving container carries at runtime — an earlier
runtime-training version OOM-crashed on Azure Container Apps' 0.5Gi. Every
image now starts from an immutable model artifact baked in at build time.

```bash
docker build -t quantml-dashboard python/
docker run -p 8080:8080 quantml-dashboard
```

**Azure** (`terraform/`): deploys to Azure Container Apps (consumption
plan, scale-to-zero) behind an Azure Container Registry, provisioned with
Terraform. The image is built and pushed with `az acr build`.

```bash
cd terraform
terraform init
terraform apply   # needs ARM_CLIENT_ID / ARM_CLIENT_SECRET / ARM_TENANT_ID /
                   # ARM_SUBSCRIPTION_ID in the environment (see
                   # python/scripts/set_azure_env.sh)

# First apply creates the registry but fails on the Container App -- no
# image in the registry yet. Build and push one:
az acr build --registry "$(terraform output -raw acr_login_server | cut -d. -f1)" \
  --image quantml-dashboard:latest ../python/

terraform apply   # succeeds now that the image exists
```

Live: **https://quantml-dashboard.salmonmeadow-1842758f.eastus.azurecontainerapps.io**

Cost: Container Registry (Basic SKU) is a flat ~$5/month; Container Apps'
consumption plan scales to zero when idle, within the free monthly
allowance for a low-traffic demo.

## Results

Synthetic data (geometric Brownian motion, no exploitable pattern by
construction): all three models land at AUC ~0.50-0.53, held-out Sharpe
near zero — the expected outcome for random-walk data.

5 years of real AAPL data: logistic regression AUC 0.544, gradient
boosting AUC 0.580, GRU AUC 0.509 but held-out backtest Sharpe 1.7-1.8 —
the best of the three by the trading-relevant metric despite the weakest
classification score, the AUC/Sharpe divergence `train.py`'s
selection-by-backtest-Sharpe logic is designed to catch. Re-running the
selected model through `eval_harness.py` on freshly re-pulled data came
back at AUC 0.518, narrowly below the 0.52 floor.

Fine-tuned sentiment classifier: 70.9% accuracy / 0.43 macro-F1 on
held-out financial tweets, against a ~65% majority-class baseline.

## Limitations

- The eval harness's fresh-data check is fully rigorous for a
  synthetic-trained model (a disjoint RNG seed) but only a partial gate
  for a real-data-trained model, since re-fetching the same ticker/period
  on the same day returns nearly the same history.
- The RAG sentiment corpus covers one sample ticker (`ACME`), so
  `--real-data`'s sentiment overlay is neutral on real tickers.
- Feature set is 8 standard technical indicators; no fundamental,
  cross-asset, or order-flow features.
- `paper_runner.py` sizes positions with a fixed `qty_per_unit`, not a
  portfolio-sizing/risk-budgeting system.
- The Azure deployment is a single instance, no custom domain, no
  authentication, and the Container Registry uses admin credentials
  rather than a managed identity (the deploy service principal's
  Contributor role excludes assigning RBAC roles to other principals —
  see `terraform/registry.tf`).
- `/api/trade/run` has no authentication of its own — it's only safe on
  the public Azure deployment because that container has no Alpaca
  credentials configured. Real auth is a prerequisite for putting trading
  credentials in that environment.
- `autonomous.py`'s continuous learning replays a fixed historical
  download from when the loop started, not genuinely new market data —
  an expanding-window retrain on data the model has already technically
  seen once (as held-out test data), not the same as ingesting real new
  bars every trading day.

## Technologies

- **Applied ML**: scikit-learn (logistic regression, gradient boosting),
  PyTorch (GRU sequence model), feature engineering, no-lookahead
  labeling, chronological train/test splitting
- **MLOps**: MLflow experiment tracking, a standalone model-quality eval
  harness with disjoint-data re-evaluation and baseline-regression
  gating, a model registry
- **Quant research**: Python, pandas, numpy — vectorized backtesting,
  walk-forward evaluation, Monte Carlo VaR/CVaR, ARIMA/GARCH volatility
  modeling, position limits + drawdown kill switches, transaction-cost
  modeling, lookahead-safe signal shifting
- **Applied LLM/RAG**: TF-IDF + cosine similarity retrieval,
  pydantic-validated structured LLM output, OpenAI-compatible LLM client
  with graceful fallback
- **LLM fine-tuning**: `transformers` + `peft` (LoRA) on a pretrained
  transformer (DistilBERT), MLflow-tracked, gated by a standalone eval
  harness
- **Web**: FastAPI, hand-rolled inline-SVG charting (no CDN dependencies)
- **Deployment**: Docker (train-at-build-time image), Terraform-provisioned
  Azure Container Apps + Container Registry
- **Live market integration**: real historical data via `yfinance`, a REST
  client for Alpaca's paper-trading API, `.env`-based credential loading
- **Continuous learning**: an accelerated-replay autonomous loop that
  trades every cycle and periodically retrains on an expanding window,
  gated by the same promotion check as a manual retrain, with per-cycle
  feature drift monitoring
- **Testing**: pytest (107 tests) covering feature engineering, both
  model families including PyTorch save/load round-trips, no-lookahead
  shift behavior, LoRA fine-tuning and its eval-harness gate, the
  backtester, walk-forward, risk limits, volatility modeling, the RAG
  pipeline, the FastAPI dashboard/ML-signal API, real data loading, and
  paper trading (mocked where it touches a network/account)

## Layout

```
python/
  quantml/
    data.py                # synthetic OHLCV generator + real (yfinance) loader
    strategies.py           # MA crossover, mean reversion, signal overlay, MLSignalStrategy
    engine.py                 # vectorized backtest engine
    walk_forward.py             # sequential out-of-sample fold evaluation
    risk.py                       # position limits + drawdown/daily-loss kill switch
    volatility.py                   # ARIMA/GARCH volatility modeling
    metrics.py                        # Sharpe, CAGR, drawdown, win rate, Monte Carlo VaR
    paper_trading.py                    # Alpaca paper-trading REST client
    paper_runner.py                       # rebalances a paper account to a strategy's target position
    autonomous.py                           # continuous-learning + paper-trading loop
    ml/
      features.py     # technical features + next-day-direction labels
      model.py          # SklearnSignalModel, TorchSignalModel (GRU)
      train.py            # chronological train/test split, MLflow logging, model selection
      eval_harness.py       # standalone model-quality gate
      registry.py             # loads whichever model train.py selected
      explain.py                # permutation importance
    rag/
      retriever.py    # TF-IDF retrieval
      signal.py         # lexicon/LLM/finetuned scoring -> daily signal
      llm.py              # OpenAI-compatible client
    finetune/
      data.py         # financial-tweet sentiment dataset (HF Hub)
      model.py          # LoRA adapter inference wrapper
      train.py            # LoRA fine-tuning of DistilBERT, MLflow logging
      eval_harness.py       # standalone model-quality gate
    web/
      app.py          # FastAPI dashboard + ML-signal API
      static/         # index.html, app.js (hand-rolled SVG charts), style.css
    cli.py             # end-to-end demo
  tests/
  Dockerfile           # containerized dashboard (train-at-build-time)
data/
  sample_docs/        # sample corpus (ticker ACME)
```

## Run it

```bash
cd python
pip install -r requirements.txt
python -m pytest tests/ -v

python -m quantml.ml.train                              # train on synthetic data
python -m quantml.ml.train --real-data --ticker AAPL --period 5y   # or real data
python -m quantml.ml.eval_harness --update-baseline      # quality gate + record baseline

python -m quantml.cli                                    # full demo: backtest, walk-forward, VaR,
                                                          # GARCH, risk limits, ML signal
python -m quantml.cli --real-data --ticker AAPL          # same demo, real Yahoo Finance history
python -m quantml.cli --sentiment-backend llm            # score RAG docs with a local LLM (needs Ollama)
python -m quantml.cli --sentiment-backend finetuned       # score RAG docs with the fine-tuned model
```

### LLM fine-tuning

```bash
cd python
python -m quantml.finetune.train                         # LoRA fine-tune DistilBERT (~2 min, CPU)
python -m quantml.finetune.eval_harness --update-baseline # quality gate + record baseline
```

### Docker

```bash
docker build -t quantml-dashboard python/
docker run -p 8080:8080 quantml-dashboard
```

### Web dashboard

```bash
cd python
uvicorn quantml.web.app:app --reload --port 8080
# then open http://localhost:8080
```

### Paper trading (real market data, no real capital)

```bash
cd python
# once: sign up free at https://app.alpaca.markets/paper/dashboard/overview
# and generate an API key pair
cp .env.example .env      # fill in ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY
                           # (.env is gitignored)

python -m quantml.paper_runner --ticker AAPL --strategy ml_signal --dry-run
python -m quantml.paper_runner --ticker AAPL --strategy ml_signal   # submits the paper order
```

### Autonomous loop

```bash
cd python
python -m quantml.autonomous --ticker AAPL
```

## Roadmap

- Larger, real financial-document corpus for the RAG layer.
- Cross-ticker generalization testing — train on one ticker's history,
  evaluate on a different one, as a stronger disjointness guarantee than
  re-fetching the same ticker.
- Hyperparameter tuning (currently fixed architecture/learning rate for
  the GRU, fixed depth/iterations for gradient boosting).
- Richer features: cross-asset signals, order-flow-style features, longer
  lookback windows for the sequence model.
- Managed identity (instead of ACR admin credentials) for the Container
  App's registry pull.
- Real auth on the dashboard, as the prerequisite for running
  `/api/trade/run` or `autonomous.py` against anything other than a local
  `.env`.
- A/B testing / canary rollout for newly-promoted models instead of
  `autonomous.py`'s current all-or-nothing promotion gate.
- Replace `autonomous.py`'s accelerated historical replay with genuine
  incremental ingestion of new trading days as they close.
