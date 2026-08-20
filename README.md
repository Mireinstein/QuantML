# QuantML

[![CI](https://github.com/Mireinstein/QuantML/actions/workflows/ci.yml/badge.svg)](https://github.com/Mireinstein/QuantML/actions/workflows/ci.yml)

An applied ML platform for systematic trading: a feature engineering +
model training pipeline (scikit-learn and a PyTorch GRU), evaluated with
no-lookahead walk-forward validation and a standalone quality gate,
driving a backtester, a research dashboard, and automated daily paper
trading.

Live market integration: `--real-data` pulls real historical prices from
Yahoo Finance, real news headlines feed the sentiment layer, and the
trading loop executes against Alpaca's paper-trading API — real market
prices, simulated capital.

## Components

### 1. `python/quantml/ml/` — the ML pipeline

**Features** (`features.py`): 11 technical indicators from OHLCV —
1/5/10/20-day returns, 20-day rolling volatility, a 10/50-day
moving-average ratio, 14-day RSI, MACD histogram, Bollinger %B, 5-day
volume change, daily high-low range. Every feature at row *t* only uses
data through day *t*'s close. Label is next-day direction.

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

Two retrieval techniques over a small bundled corpus of sample financial
documents, both behind the same `.query(text, top_k)` interface:

- **`Retriever`** — TF-IDF + cosine similarity, classical IR.
- **`EmbeddingRetriever`** — real text embeddings + cosine similarity
  (local Ollama `nomic-embed-text`, with a deterministic hashing-trick
  fallback). The query and the whole corpus are embedded together in one
  call, guaranteeing both always come from the same backend.
  `GET /api/rag/search` runs a query through both side by side.

**Real news** (`news.py`): pulls current Yahoo Finance headlines for any
real ticker as retriever documents, so `--real-data` runs build their
sentiment signal from real, current stories about the actual company.

Retrieved documents feed a per-day sentiment signal into
`SignalOverlayStrategy`, with three scoring backends
(`--sentiment-backend`): **lexicon** (deterministic keyword scoring),
**llm** (a local Ollama model, structured output validated with
pydantic), and **finetuned** (the LoRA-fine-tuned classifier below).

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
performance, live prediction, and feature importance. Endpoints reuse
`cli.py`'s recipe and recompute on every call rather than caching. No
CDN dependencies — charts in `static/app.js` are a small hand-rolled
inline-SVG helper.

`/api/ml-signal` reports the model's recorded chronological held-out
performance; `/api/ml-signal/predict` does live inference on today's
market data.

**Live serving monitoring** (`GET /api/ml-signal/monitoring`, same
pattern as TenantIQ's `ml/serve.py`): a fixed-size rolling window over
recent `/predict` requests, tracking p50/p95 latency and a drift-flag
rate. Drift means a feature value in a live request landed more than 3
standard deviations from the training distribution's mean for that
feature — reference stats are computed once from the model's own
recorded training data source and cached for the process's life, not
from the live request itself.

### 5b. `python/quantml/tradingagent.py` — trading assistant agent

A multi-turn chat agent on the "LLM proposes, code disposes"
architecture: the LLM reads the conversation and returns one validated
JSON decision — a read-only action (`predict`, `explain`, `status`, or
`none`) plus a ticker and a draft reply. The server executes that action
deterministically against the same functions the dashboard's own
Predict/Explain buttons call, then returns the reply alongside the real
computed data. The action space is read-only by construction (enforced
in the type and asserted in a test); orders go through the dashboard's
authenticated trade flow.

Backend: any OpenAI-compatible endpoint via
`OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL`; an `OPENROUTER_API_KEY`
in `.env` routes to OpenRouter, and a local Ollama server works with
zero configuration. Structured output is pydantic-validated, with
deterministic template replies filling in whenever the model's own reply
comes back blank.

### 6. Real market data + paper trading

`data.py::load_real_ohlcv(ticker, period, interval)` pulls real daily
OHLCV from Yahoo Finance via `yfinance`, normalized to the same column
contract as the synthetic generator.

`data.py::search_tickers(query)` (`GET /api/tickers/search?q=`) is a
company-name -> ticker lookup — type "Apple", get AAPL. Queries Yahoo
Finance search live, filtered to US-listed common stock, and powers the
dashboard's autocomplete ticker inputs (`attachTickerAutocomplete` in
`app.js`).

`paper_trading.py` is a REST client for Alpaca's paper trading API,
fixed to the paper endpoint. Credentials load from a gitignored
`python/.env`; every call raises `PaperTradingError` on failure.

`paper_runner.py` wires it together: `python -m quantml.paper_runner
--ticker AAPL --strategy ml_signal` pulls real market data, runs the
model, compares its target position against the account's actual paper
position, and submits whatever order closes the gap (`--dry-run` to just
print it).

The dashboard has an on-demand version: a "Run trade now" button
(`POST /api/trade/run`) that does the same rebalance from the browser.
Since it submits real paper orders, it requires authentication (see
Deployment).

### Daily live-trading loop (`autonomous.py`)

    python -m quantml.autonomous --ticker AAPL

Fully hands-off daily trading: the loop checks periodically
(`--check-interval-seconds`, default 1 hour) whether a new real trading
day's bar is available, and when it is, gets the live model's
prediction, sizes a position, and submits a real order to the Alpaca
paper account at today's real market price — once per real trading day.
Runs locally via the command above, or as its own Azure Container App
(`trader_service.py` wraps it with a pause/resume control API — see
Deployment) started and stopped from the dashboard.

Continuous learning runs alongside it: `.github/workflows/retrain-eval.yml`
(see CI below) retrains daily on the newest real market data, gated by
the quality harness, so the model keeps up with the market while the
loop keeps trading.

Each check also runs a feature drift check (z-scores against the
training distribution, computed once at startup) and appends to
`ml/autonomous_log.jsonl`, shown on the dashboard's "Live autonomous
trading" panel. The loop is resilient: a failed check is logged and the
loop continues.

**Bot trading performance panel**: two endpoints, both reading straight
from Alpaca:

- `GET /api/autonomous/equity` — the paper account's equity over time
  (`paper_trading.py::get_portfolio_history`), stays flat until orders fill.
- `GET /api/autonomous/trades` — order history
  (`paper_trading.py::list_orders`): side, qty, status, and whether it's
  filled (`filled_qty`/`filled_avg_price`) or still pending.

## CI (`.github/workflows/`)

**`ci.yml`** — every push/PR to `main`: `pytest tests/`, then builds the
`python/Dockerfile` image and smoke-tests it for real — runs the
container, waits for it to become ready, hits `/api/dashboard`,
`/api/ml-signal`, and `/api/tickers/search` — so a broken container
fails CI, not just a broken unit test.

**`retrain-eval.yml`** — the automated retrain/eval loop: this is what
makes the model keep learning over real time. Runs daily after each US
trading day's close (06:00 UTC Tue–Sat), on changes to whatever defines
the model (`ml/features.py`, `ml/model.py`, `ml/train.py`), or manually
via `workflow_dispatch`. Retrains on the latest 5 years of real market
data, evaluates against the quality gate (a regression or sub-floor
metric fails the workflow before anything downstream runs), commits the
new baseline back to the repo on pass, then builds and smoke-tests the
retrained image the same way `ci.yml` does.

## Deployment

**Docker** (`python/Dockerfile`): trains the model at image build time,
using the build host's full resources, and bakes an immutable model
artifact into the image — the serving container starts instantly from a
ready model. `REAL_DATA=1` trains on real market data (what deployments
use); the default trains on synthetic data for fast CI builds.

```bash
docker build --build-arg REAL_DATA=1 --build-arg TICKER=AAPL --build-arg PERIOD=5y \
  -t quantml-dashboard python/
docker run -p 8080:8080 quantml-dashboard
```

**Azure** (`terraform/`): two Container Apps behind one Azure Container
Registry, provisioned with Terraform. The **dashboard** app (public,
scale-to-zero) serves the FastAPI app. The **trader** app runs the same
image with its command overridden to `trader_service.py` instead —
`autonomous.run()` in a background thread behind a tiny internal control
API, reachable only from other apps in the same Container Apps
environment, never from the public internet. The dashboard's Start/Stop
buttons call the trader's internal `/resume`/`/pause` over that private
network.

```bash
cd terraform
terraform init

source ../python/scripts/set_trading_env.sh   # exports TF_VAR_* for the
                                                # secrets below from your
                                                # local python/.env --
                                                # see that script

terraform apply   # needs ARM_CLIENT_ID / ARM_CLIENT_SECRET / ARM_TENANT_ID /
                   # ARM_SUBSCRIPTION_ID in the environment too (see
                   # python/scripts/set_azure_env.sh)

# First apply creates the registry; then build and push the image:
az acr build --registry "$(terraform output -raw acr_login_server | cut -d. -f1)" \
  --image quantml-dashboard:latest \
  --build-arg REAL_DATA=1 --build-arg TICKER=AAPL --build-arg PERIOD=5y ../python/

terraform apply   # wires the Container Apps to the pushed image
```

Live: **https://quantml-dashboard.salmonmeadow-1842758f.eastus.azurecontainerapps.io**

The trader app starts **paused** on every deploy; trading begins when
Start is clicked on the dashboard.

**Auth**: action endpoints (placing a trade, starting/stopping the bot)
require authentication. Read-only endpoints — performance, trade history,
model metrics — are public.

## Results

Synthetic data (geometric Brownian motion, no exploitable pattern by
construction): all three models land at AUC ~0.50-0.53, held-out Sharpe
near zero — the expected outcome for random-walk data, confirming the
pipeline doesn't manufacture signal where none exists.

5 years of real AAPL data (currently deployed): the GRU sequence model
wins selection with a held-out backtest Sharpe of 1.5+, ahead of
logistic regression and gradient boosting on the trading-relevant
metric — the AUC/Sharpe divergence `train.py`'s
selection-by-backtest-Sharpe logic is designed to catch.

Fine-tuned sentiment classifier: 70.9% accuracy / 0.43 macro-F1 on
held-out financial tweets, against a ~65% majority-class baseline.

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
- **Applied LLM/RAG**: TF-IDF + cosine similarity retrieval, real text
  embeddings (Ollama) with a deterministic hashing-trick fallback,
  pydantic-validated structured LLM output, OpenAI-compatible LLM client
  with graceful fallback
- **AI agent orchestration**: a multi-turn chat agent with a per-turn
  structured decision loop, a read-only typed action space, and
  pydantic-validated structured output with deterministic fallbacks
- **LLM fine-tuning**: `transformers` + `peft` (LoRA) on a pretrained
  transformer (DistilBERT), MLflow-tracked, gated by a standalone eval
  harness
- **Web**: FastAPI, hand-rolled inline-SVG charting (no CDN dependencies)
- **Deployment**: Docker (train-at-build-time image), Terraform-provisioned
  Azure Container Apps + Container Registry
- **Live market integration**: real historical data via `yfinance`, a REST
  client for Alpaca's paper-trading API, `.env`-based credential loading
- **Continuous learning**: a daily live-trading loop decoupled from a
  separately-scheduled retrain/eval GitHub Actions workflow that retrains
  on accumulated real market data and is gated by the same promotion
  check as a manual retrain, with feature drift monitoring on every check
- **CI/CD**: GitHub Actions running tests plus a real Docker
  build-and-smoke-test on every push, and a scheduled retrain/eval/deploy
  pipeline that commits an updated model baseline only on passing the
  quality gate
- **Live model monitoring**: rolling-window p50/p95 latency and feature
  drift-flag-rate tracked over real inference requests
- **Testing**: pytest (166 tests) covering feature engineering, both
  model families including PyTorch save/load round-trips, no-lookahead
  shift behavior, LoRA fine-tuning and its eval-harness gate, the
  backtester, walk-forward, risk limits, volatility modeling, the RAG
  pipeline (both retrieval backends plus the news loader), the trading
  agent, the FastAPI dashboard/ML-signal API, real data loading, and
  paper trading -- run automatically on every push via GitHub Actions

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
    autonomous.py                           # daily live paper-trading loop
    trader_service.py                        # pause/resume control API wrapping autonomous.py (Azure trader app)
    tradingagent.py                            # multi-turn chat agent (LLM proposes, code disposes)
    ml/
      features.py     # technical features + next-day-direction labels
      model.py          # SklearnSignalModel, TorchSignalModel (GRU)
      train.py            # chronological train/test split, MLflow logging, model selection
      eval_harness.py       # standalone model-quality gate
      registry.py             # loads whichever model train.py selected
      explain.py                # permutation importance
    rag/
      retriever.py    # TF-IDF retrieval + embedding-based semantic retrieval
      embeddings.py     # Ollama embeddings, hashing-trick fallback
      news.py             # real Yahoo Finance headlines as retriever documents
      signal.py             # lexicon/LLM/finetuned scoring -> daily signal
      llm.py                  # OpenAI-compatible client
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
  data/
    sample_docs/       # sample corpus (ticker ACME)
  Dockerfile           # containerized dashboard (train-at-build-time)
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
docker build --build-arg REAL_DATA=1 --build-arg TICKER=AAPL --build-arg PERIOD=5y \
  -t quantml-dashboard python/
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

### Daily live-trading loop

```bash
cd python
python -m quantml.autonomous --ticker AAPL
```

## Roadmap

- Larger, real financial-document corpus for the RAG layer.
- Cross-ticker generalization testing — train on one ticker's history,
  evaluate on a different one.
- Richer features: cross-asset signals, order-flow-style features, longer
  lookback windows for the sequence model.
- A/B testing / canary rollout for newly-promoted models.
