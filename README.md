# QuantML

[![CI](https://github.com/Mireinstein/QuantML/actions/workflows/ci.yml/badge.svg)](https://github.com/Mireinstein/QuantML/actions/workflows/ci.yml)

An applied ML platform for systematic trading: feature engineering, model
training and selection, a quality-gated retraining pipeline, a
backtesting/risk toolkit, a research dashboard, and an autonomous daily
paper-trading loop — deployed on Azure.

**Live**: https://quantml-dashboard.salmonmeadow-1842758f.eastus.azurecontainerapps.io

## What it does

**ML signal** (`quantml/ml/`) — 11 technical indicators built from OHLCV
with strict no-lookahead discipline; three model families (logistic
regression, gradient boosting, a PyTorch GRU reading 20-day sequences)
trained on a chronological split and selected by held-out backtest
Sharpe; MLflow-tracked; permutation-importance explainability.

**Quality gate** (`ml/eval_harness.py`) — a standalone check on the model
artifact itself, scored on data disjoint from the training window (a
fresh post-training-cutoff slice for real-data models, a different RNG
seed for synthetic). Ships only on passing absolute floors and a
no-regression check against the recorded baseline.

**Continuous learning** (`.github/workflows/retrain-eval.yml`) — retrains
daily after each US trading day's close on the latest 5 years of real
market data, runs the quality gate, commits the new baseline, and
smoke-tests an image carrying the exact artifact that passed the gate.

**Quant research toolkit** — vectorized backtester with transaction
costs; walk-forward fold evaluation; Monte Carlo VaR/CVaR; ARIMA +
GARCH(1,1) volatility modeling; position limits with a
drawdown/daily-loss kill switch.

**RAG sentiment layer** (`quantml/rag/`) — TF-IDF and embedding-based
retrieval side by side over a document corpus; real Yahoo Finance
headlines for real tickers; three scoring backends (lexicon, local LLM
with validated structured output, and a LoRA-fine-tuned DistilBERT);
sentiment tilts a base strategy through `SignalOverlayStrategy`.

**LLM fine-tuning** (`quantml/finetune/`) — LoRA on DistilBERT for
financial sentiment (1.09% of parameters trained), gated by its own eval
harness: 70.9% accuracy / 0.43 macro-F1 held-out vs a ~65%
majority-class baseline.

**Trading assistant** (`tradingagent.py`) — a multi-turn chat agent on
the "LLM proposes, code disposes" pattern: the LLM returns one validated
JSON decision from a read-only action space; the server executes it
against the same functions behind the dashboard's Predict/Explain
buttons and returns real computed data. Works with any OpenAI-compatible
backend (OpenRouter, local Ollama).

**Research dashboard** (`quantml/web/`) — FastAPI + hand-rolled inline-SVG
charts: equity curves, walk-forward folds, VaR, GARCH, live prediction
with company-name ticker autocomplete, feature importance, live serving
monitoring (rolling p50/p95 latency + feature-drift rate), the chat
assistant, and bot controls.

**Autonomous daily trading** (`autonomous.py`) — checks hourly for a new
*completed* daily bar (today's in-progress bar is excluded until the
session closes), then predicts, sizes a position, and rebalances the
Alpaca paper account — once per real trading day, hands-off. Runs as its
own Azure Container App with a pause/resume API driven by the
dashboard's Start/Stop buttons; equity and order history stream straight
from Alpaca onto the dashboard.

## Results

- Synthetic GBM data (no exploitable pattern by construction): all
  models land at AUC ~0.5, Sharpe ~0 — the pipeline finds no signal
  where none exists.
- Real AAPL, 5 years: gradient boosting selected — held-out AUC 0.587,
  held-out backtest Sharpe 1.59 on post-training-cutoff data.
- Sentiment classifier: 70.9% accuracy / 0.43 macro-F1 held-out.

## Layout

```
python/
  quantml/
    data.py            # synthetic OHLCV generator + real (yfinance) loader + ticker search
    strategies.py      # MA crossover, mean reversion, signal overlay, MLSignalStrategy
    engine.py          # vectorized backtest engine
    walk_forward.py    # sequential out-of-sample fold evaluation
    risk.py            # position limits + drawdown/daily-loss kill switch
    volatility.py      # ARIMA/GARCH volatility modeling
    metrics.py         # Sharpe, CAGR, drawdown, win rate, Monte Carlo VaR
    paper_trading.py   # Alpaca paper-trading REST client
    paper_runner.py    # single-shot paper-account rebalance
    autonomous.py      # autonomous daily paper-trading loop
    trader_service.py  # pause/resume control API (Azure trader app)
    tradingagent.py    # multi-turn chat agent
    ml/                # features, models, training, eval gate, registry, explainability
    rag/               # retrieval (TF-IDF + embeddings), news, sentiment scoring
    finetune/          # LoRA fine-tuning + eval gate
    web/               # FastAPI dashboard + static frontend
    cli.py             # end-to-end demo
  tests/               # 169 tests, run on every push
  data/sample_docs/    # sample corpus (ticker ACME)
  Dockerfile           # trains at build time, or ships a provided artifact
terraform/             # Azure Container Apps + Container Registry
```

## Run it

```bash
cd python
pip install -r requirements.txt
python -m pytest tests/ -v

python -m quantml.ml.train --real-data --ticker AAPL --period 5y
python -m quantml.ml.eval_harness --update-baseline

python -m quantml.cli --real-data --ticker AAPL    # backtest, walk-forward, VaR, GARCH, ML signal
uvicorn quantml.web.app:app --reload --port 8080   # dashboard at localhost:8080

python -m quantml.paper_runner --ticker AAPL --strategy ml_signal --dry-run
python -m quantml.autonomous --ticker AAPL         # autonomous daily loop
```

Paper trading uses a free Alpaca paper account (keys in a gitignored
`.env` — see `.env.example`).

### Deploy

```bash
cd terraform
terraform init
source ../python/scripts/set_trading_env.sh
terraform apply
az acr build --registry "$(terraform output -raw acr_login_server | cut -d. -f1)" \
  --image quantml-dashboard:latest \
  --build-arg REAL_DATA=1 --build-arg TICKER=AAPL --build-arg PERIOD=5y ../python/
terraform apply
```

Two Container Apps share one image: the public dashboard, and an
internal-only trader running the autonomous loop. The trader starts
paused; trading begins from the dashboard's Start button. Action
endpoints require authentication.

## Roadmap

- Larger, real financial-document corpus for the RAG layer.
- Cross-ticker generalization testing — train on one ticker's history,
  evaluate on a different one.
- Richer features: cross-asset signals, order-flow-style features, longer
  lookback windows for the sequence model.
- A/B testing / canary rollout for newly-promoted models.
