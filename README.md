# QuantIQ

An applied ML platform for systematic trading: a feature-engineering +
model-training pipeline (scikit-learn and a PyTorch GRU sequence model),
evaluated with no-lookahead walk-forward validation and a standalone
MLOps quality gate, driving a real backtester, a research dashboard, and
real (paper) order execution. Built to demonstrate the core skills ML
engineering roles actually test for: turning a model into a full pipeline
(features → training → evaluation → a quality gate → serving), not just a
notebook that fits once and stops — plus the judgment to report a model
that *doesn't* beat the market as honestly as one that does.

**No real capital is ever at risk.** By default everything runs on
synthetic price data and a synthetic news/filing corpus (see below). Two
pieces optionally connect to real external services, both opt-in and both
capital-safe: `--real-data` pulls real historical prices from Yahoo Finance
(read-only, no account needed), and `paper_trading.py`/`paper_runner.py`
submit orders to a real broker's **paper-trading sandbox** (Alpaca) — real
market prices, simulated fake cash, hard-coded to the paper API endpoint so
it can never reach a live-money account. No financial advice, no claims
about real-market performance.

## Components

### 1. `python/quantiq/ml/` — the ML pipeline

**Features** (`features.py`): technical indicators computed from OHLCV --
1/5/10-day returns, 20-day rolling volatility, a 10/50-day moving-average
ratio, 14-day RSI, 5-day volume change, and the daily high-low range --
with the same no-lookahead discipline as the strategies below: every
feature at row *t* only uses data available through day *t*'s close. The
label is next-day direction (`build_features_and_labels`), and the very
last row is dropped because it has no "next day" to label yet, not filled
with a fake value.

**Models** (`model.py`), both exposed through one `predict_proba(features)`
interface so nothing downstream needs to know which kind it's holding:

- `SklearnSignalModel` -- logistic regression baseline or a
  `HistGradientBoostingClassifier`, same wrapper shape as TenantIQ's
  `RiskModel`.
- `TorchSignalModel` -- a small PyTorch **GRU** that reads a 20-day rolling
  *sequence* of features rather than a single flat row, a real (if modest)
  recurrent model over the temporal structure. Rows before the first full
  window get a neutral 0.5 rather than crashing or being dropped (see
  Honest bugs below for why that specific behavior exists).

**Training** (`train.py`, `python -m quantiq.ml.train [--real-data --ticker
AAPL --period 5y]`): a single **chronological** train/test split (not
random -- price history is sequential, so a random split would leak future
information into training through overlapping rolling-window features,
unlike TenantIQ's applicant data where rows really are independent).
Trains all three models on the same split and evaluates each two ways:
ROC-AUC on the held-out labels, *and* the model's signal run through the
real backtester (`engine.run_backtest` + `MLSignalStrategy`) on the same
held-out period. **Selection is by held-out backtest Sharpe, not AUC** --
a model can have a mediocre AUC and still be a perfectly fine trading
signal (or a great AUC on the wrong moves and a bad one), and this project
found a real case of exactly that divergence (see Results). Everything is
logged to **MLflow** (local SQLite backend, `ml/mlflow.db`, no server
needed) -- params, both metric families, per-model runs.

**Evaluation gate** (`eval_harness.py`, `python -m quantiq.ml.eval_harness
[--update-baseline]`): separate from the pytest unit tests (which check the
*code*), this checks the *model artifact on disk* is good enough to ship.
Re-evaluates the saved model against data disjoint from training -- a
different synthetic seed for a synthetic-trained model (fully rigorous, no
different from TenantIQ's risk-model gate), or a freshly re-fetched real
series for a real-data-trained model (an honest partial gate -- see Honest
limitations). Fails (non-zero exit, the kind of thing a CI pipeline would
gate on) if AUC or held-out Sharpe drop below fixed floors or regress past
a tolerance versus the last recorded baseline (`eval_baseline.json`).

**Registry** (`registry.py`): loads whichever model `train.py` most
recently selected, so `cli.py`, the dashboard, and `paper_runner.py` don't
each need to know both model classes and pick the right one.

### 2. `python/quantiq/strategies.py::MLSignalStrategy`

Same `Strategy` interface as the rule-based strategies below --
`positions(prices) -> pd.Series` in `[-1, 1]` -- so the trained model
plugs into the *exact same* backtester, walk-forward evaluator, risk-limit
kill switch, and paper-trading runner as everything else, with zero special
casing anywhere else in the codebase. The model's `P(next day up)` maps to
a position sized by confidence (0.5 → flat, 1.0 → full long, 0.0 → full
short) rather than a hard threshold, so a barely-confident prediction
produces a small position, not the same full-size bet as a highly
confident one. Same shift-by-one-day convention as every other strategy
here, so the position actually held on day *t* was decided using only
information known before day *t* opened.

### 3. `python/quantiq/` — backtesting engine (rule-based strategies + risk)

Vectorized backtester: strategies emit a target position per bar, the
engine nets out transaction costs and produces an equity curve. Metrics:
Sharpe ratio, CAGR, max drawdown, win rate, and **Monte Carlo VaR/CVaR**
(historical bootstrap resampling of the strategy's own realized returns --
no assumed distribution).

**Walk-forward evaluation** (`walk_forward.py`) splits the backtest into
sequential out-of-sample folds instead of reporting one full-period Sharpe
that a single lucky/unlucky stretch can dominate. Real finding from this
project's own baseline strategy: full-period Sharpe looks fine (0.84), but
walk-forward reveals high variance across folds (std of fold Sharpe: 1.77,
worst fold: -1.999).

**Risk management** (`risk.py`): `RiskLimits` + `apply_risk_limits` enforce
a max position size and a max-drawdown (or max-daily-loss) kill switch --
this actually forces the strategy flat once breached and keeps it flat.
Demonstrated live in `cli.py`: a 10% max-drawdown limit trips on a real
synthetic drawdown day, capping both further losses *and* the subsequent
recovery (final equity 1.12 with the limit vs. 1.62 unlimited).

**Volatility modeling** (`volatility.py`): ARIMA (statsmodels) for the
conditional mean, GARCH(1,1) (`arch`) for volatility clustering, tested
against a simulated ground-truth GARCH process rather than just "it runs."

Includes two baseline technical strategies (moving-average crossover, mean
reversion) and a `SignalOverlayStrategy` that blends a base strategy's
position with an external signal series (used by the RAG layer below).

### 4. `python/quantiq/rag/` — retrieval-augmented signal layer

TF-IDF retrieval over a small corpus of sample financial documents
(`data/sample_docs/`), turned into a per-day sentiment signal that feeds
into `SignalOverlayStrategy`. Three scoring backends, chosen with
`--sentiment-backend`: **lexicon** (default, deterministic keyword
scoring, zero dependencies), **llm** (sends each document to a local
Ollama model for a structured `{score, rationale}` response, validated
with pydantic), and **finetuned** (the actually-fine-tuned classifier from
`quantiq/finetune/`, below) -- both `llm` and `finetuned` fall back to the
lexicon scorer per-document if the endpoint is unreachable / no adapter
has been trained yet. The 10 documents in `data/sample_docs/` are
synthetic text for a fictional ticker (`ACME`) -- not real news.

### 4b. `python/quantiq/finetune/` — LLM fine-tuning

The `llm` backend above calls a general-purpose model through a prompt --
useful, but not the same skill as actually fine-tuning one. This module
**LoRA fine-tunes a real pretrained transformer** (`distilbert-base-uncased`)
for 3-class financial sentiment (Bearish/Bullish/Neutral) on a genuine
public labeled dataset -- [`zeroshot/twitter-financial-news-sentiment`]
(https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
on the Hugging Face Hub, ~9.5k labeled financial tweets, not the 10-document
synthetic RAG corpus (which is intentionally tiny, just enough to make
retrieval runnable offline -- a real fine-tuning target needs real scale).

LoRA (Low-Rank Adaptation, via `peft`) trains a small pair of low-rank
matrices injected into DistilBERT's attention projections (`q_lin`,
`v_lin`) instead of updating the base model's ~66M parameters directly --
**1.09% of parameters are actually trainable** (740,355 of 67,696,134),
which is closer to how fine-tuning is done in practice than updating a
full model. Same
train/eval/gate discipline as the trading-signal model: `train.py` fits on
a 3,000-tweet subsample (2 epochs, CPU-only, ~2 minutes -- capped for
speed, not because more data wouldn't help) and logs to MLflow;
`eval_harness.py` is a standalone quality gate re-evaluating on a fresh
held-out slice, with the same fail-below-floor/fail-on-regression logic as
the other two eval harnesses in this project (and TenantIQ's).

**Real, measured result**: 70.9% accuracy / 0.43 macro-F1 on the held-out
validation set (majority-class baseline on this dataset is ~65% Neutral,
so this is genuine per-class signal, not just calling everything
Neutral) -- not a cherry-picked number, this is what a 2-epoch, 3k-example
LoRA run on a CPU actually produces, reported as-is.

### 5. `python/quantiq/web/` — research dashboard

A FastAPI app (`app.py`) turning the CLI demo into a browser UI: equity
curves, walk-forward folds, VaR/CVaR, GARCH volatility, the risk-limit
kill switch, and the ML signal's real held-out performance -- plus a
**live prediction** endpoint that asks the currently-trained model what it
thinks about a real ticker right now. Read-only endpoints reuse `cli.py`'s
exact recipe (same seeds, same strategy construction) so numbers match
this README, and nothing is cached except the static sample-doc corpus --
every backtest is recomputed on every call. Frontend has no CDN
dependencies: charts in `static/app.js` are a small hand-rolled inline-SVG
line/bar-chart helper, not a pulled-in library.

**Why `/api/ml-signal` doesn't just re-run the model on the dashboard's own
data**: this project's default synthetic demo series shares its RNG seed
with `train.py`'s default training data, so re-scoring the model against
it would silently show in-sample performance dressed up as a real result
(this actually happened during development -- see Honest bugs). Instead,
`/api/ml-signal` reports the model's *recorded* chronological held-out
performance from training time, and `/api/ml-signal/predict` does a
genuinely different, leakage-free thing: live inference on today's real
market data, which isn't a backtest metric at all.

### 6. Real market data + paper trading

`data.py::load_real_ohlcv(ticker, period, interval)` pulls real historical
daily OHLCV from Yahoo Finance via `yfinance` -- free, no account or API
key -- normalized to the exact same column contract as the synthetic
generator, so it's a drop-in replacement everywhere a price DataFrame is
expected. `cli.py --real-data --ticker AAPL` and `ml/train.py --real-data
--ticker AAPL` both use it.

`paper_trading.py` is a thin REST client for **Alpaca's paper trading
API** -- simulated order execution against real market prices with fake
starting cash. The paper API's base URL is hard-coded
(`https://paper-api.alpaca.markets`), deliberately not configurable via env
var or argument, so this client can never be one misconfigured setting
away from a live-money account. Credentials load from a gitignored
`python/.env` file (`ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY`) via
`python-dotenv` rather than being typed into a shell command; every call
raises a clear `PaperTradingError` instead of silently no-opping if
they're missing or a request fails.

`paper_runner.py` wires it together: `python -m quantiq.paper_runner
--ticker AAPL --strategy ml_signal` pulls real market data, runs the
trained model, compares its target position against the account's actual
paper position, and submits whatever market order closes the gap (or just
prints what it would do, with `--dry-run`). Verified end-to-end against a
real Alpaca paper account, not just mocked in tests.

## Deployment

**Docker** (`python/Dockerfile`): containerizes the FastAPI dashboard.
Trains a model on first boot if none is baked into the image (kept out via
`.dockerignore`, so every image starts from a known-fresh artifact rather
than whatever happened to be on the host when it was built) -- same
train-if-missing pattern as TenantIQ's `ml/Dockerfile`. Verified by
simulating the exact container entrypoint locally (Docker itself wasn't
available in the environment this was built in): copying the source tree
minus the `.dockerignore`'d artifacts into a clean directory, running the
same `test -f ... || python3 -m quantiq.ml.train; exec uvicorn ...` command
the `CMD` executes, and confirming it trains fresh then serves correctly.

```bash
docker build -t quantiq-dashboard python/
docker run -p 8080:8080 quantiq-dashboard
```

**Azure** (`terraform/`): deploys the dashboard to **Azure Container Apps**
(consumption plan, scale-to-zero when idle) behind an **Azure Container
Registry**, provisioned with Terraform (`azurerm` provider, credentials
via a service principal read from the environment -- never hard-coded).
The image is built and pushed with `az acr build` (a cloud-side build, so
this doesn't depend on Docker being installed locally either).

**A real bug, found by actually deploying this**: the Dockerfile
originally trained the model at container *startup* (same train-if-missing
pattern as the local Docker section above, and as TenantIQ's
`ml/Dockerfile`). That works fine for TenantIQ's lightweight sklearn
model, but this project's training step loads PyTorch + transformers +
scikit-learn together, and on Container Apps' 0.5Gi memory it OOM-crashed
on every cold start -- confirmed via `az containerapp replica list`
showing `restartCount` climbing while the container never reached a ready
state. Fixed by moving training into the image *build* step instead (`RUN
python3 -m quantiq.ml.train` in the Dockerfile, using the build host's
full resources, not the constrained serving container's) -- which also
happens to be the more correct pattern anyway: an immutable, reproducible
model artifact baked into the image, not retrained unpredictably on every
cold start.

```bash
cd terraform
terraform init
terraform apply   # needs ARM_CLIENT_ID / ARM_CLIENT_SECRET / ARM_TENANT_ID /
                   # ARM_SUBSCRIPTION_ID in the environment (see
                   # python/scripts/set_azure_env.sh)

# First apply creates the registry but fails on the Container App -- there's
# no image in the registry yet for it to pull. Build and push one (this
# `terraform output` gives you the registry name this apply just created):
az acr build --registry "$(terraform output -raw acr_login_server | cut -d. -f1)" \
  --image quantiq-dashboard:latest ../python/

terraform apply   # succeeds now that the image exists
```

Live: **https://quantiq-dashboard.redcliff-1024218d.eastus.azurecontainerapps.io**
-- verified end-to-end, not just "resources exist": real HTTP 200s, the
real trained model's metadata from `/api/ml-signal`, and a real live
prediction against real AAPL data fetched from *inside* the running
container via `/api/ml-signal/predict?ticker=AAPL`.

Cost: the Container Registry (Basic SKU) is a flat ~$5/month; Container
Apps' consumption plan scales to zero compute cost when idle, and its free
monthly allowance (180K vCPU-seconds / 360K GiB-seconds) comfortably
covers a low-traffic demo dashboard.

## Results (honest, not cherry-picked)

Trained on **synthetic data** (geometric Brownian motion, genuinely no
exploitable pattern by construction): all three models land at AUC
~0.50-0.53, essentially random, and held-out Sharpe near zero or negative.
This is the theoretically *correct* outcome for random-walk data -- a
model finding real signal there would be the actual red flag.

Trained on **5 years of real AAPL data**: logistic regression AUC 0.544,
gradient boosting AUC 0.580 (genuinely above random), GRU AUC 0.509 (near
random) but held-out backtest Sharpe **1.7-1.8** -- the best of the three
by the trading-relevant metric despite the weakest classification score,
exactly the AUC/Sharpe divergence `train.py`'s selection-by-backtest-Sharpe
logic exists to catch. Re-running the *same* selected model through
`eval_harness.py` on freshly re-pulled data came back at AUC 0.518,
narrowly failing the 0.52 floor -- a real, close, honestly-reported
result showing exactly why an independent evaluation gate (not just
"trust the training run's own numbers") earns its place in the pipeline.

**Fine-tuned sentiment classifier** (`finetune/`): 70.9% accuracy / 0.43
macro-F1 on held-out financial tweets, against a ~65% majority-class
("Neutral") baseline -- genuine per-class signal from a 2-epoch, 3,000-tweet
LoRA run on a CPU in about 2 minutes, not a cherry-picked number.

## Honest bugs found and fixed during development

- **PyTorch + scikit-learn segfault**: on this machine (Anaconda's numpy/
  scikit-learn stack), importing `torch` before scikit-learn's native
  `HistGradientBoostingClassifier` code ever runs hard-crashes the
  interpreter (SIGSEGV) -- a known class of conflict between PyTorch's
  bundled OpenMP runtime and the one scikit-learn/numpy initialize, where
  whichever library's native runtime registers first "wins." Fixed by
  reordering imports in `model.py` (sklearn before torch) and documented
  in-line so it can't be silently reordered back.
- **GRU crash on short input**: the live `/api/ml-signal/predict` endpoint
  originally passed only the single latest feature row to the model, but
  the GRU needs a full 20-day window to form one prediction --
  `np.stack([])` on zero sequences crashed with a 500. Fixed at the root
  (`TorchSignalModel.predict_proba` now returns neutral 0.5 for input
  shorter than its window instead of crashing) plus a regression test.
- **Data leakage in the CLI/dashboard demo**: an early version re-ran the
  trained model against `cli.py`'s own demo data to show "live" backtest
  numbers -- and because that demo data shares an RNG seed with
  `train.py`'s default training data, the first ~750 days are bit-identical
  to what the model trained on, producing a Sharpe of 7.4 that was really
  just in-sample memorization dressed up as a result. Fixed by having
  `cli.py` and `/api/ml-signal` report the model's *recorded* true
  held-out performance instead of ever re-scoring it against data that
  might overlap what it trained on.

## Honest limitations

- The eval harness's "fresh data" check is fully rigorous for a
  synthetic-trained model (a disjoint RNG seed) but only a partial gate
  for a real-data-trained model, since re-fetching the same ticker/period
  on the same day returns nearly the same history -- it becomes a true
  walk-forward check once enough real time has passed for new trading days
  to exist beyond the training cutoff.
- The RAG sentiment corpus only covers a fictional ticker (`ACME`), so
  `--real-data`'s sentiment overlay is neutral (no real tilt) on real
  tickers -- reported plainly by `cli.py` rather than hidden.
- Feature set is a modest, standard technical-indicator set (8 features);
  no fundamental, cross-asset, or order-flow features.
- `paper_runner.py` sizes positions with a fixed `qty_per_unit`, not a
  real portfolio-sizing/risk-budgeting system.
- The Azure deployment is a single instance behind Azure's default
  auto-generated domain, no custom domain, no authentication in front of
  it, and the Container Registry uses admin credentials rather than a
  managed identity (the deploy service principal only has the Contributor
  role, which deliberately excludes assigning RBAC roles to other
  principals -- see `terraform/registry.tf`). Fine for a demo, not how
  you'd run this for real users.

## Technologies

- **Applied ML**: scikit-learn (logistic regression, gradient boosting),
  **PyTorch** (a GRU sequence model, real `nn.Module`/training loop/
  `DataLoader`-free batch training), feature engineering, no-lookahead
  labeling, chronological (not random) train/test splitting for time
  series
- **MLOps**: MLflow experiment tracking (local SQLite backend), a
  standalone model-quality eval harness with disjoint-data re-evaluation
  and baseline-regression gating, a model registry that abstracts which
  model type is currently deployed
- **Quant research**: Python, pandas, numpy -- vectorized backtesting,
  walk-forward evaluation, Monte Carlo VaR/CVaR, ARIMA/GARCH volatility
  modeling, position limits + drawdown kill switches, transaction-cost
  modeling, lookahead-safe signal shifting
- **Applied LLM/RAG**: scikit-learn (TF-IDF + cosine similarity
  retrieval), pydantic-validated structured LLM output, OpenAI-compatible
  LLM client with graceful fallback
- **LLM fine-tuning**: `transformers` + `peft` (LoRA) fine-tuning of a real
  pretrained transformer (DistilBERT) on a real public labeled dataset,
  MLflow-tracked, gated by its own standalone eval harness
- **Web**: FastAPI, vanilla-JS hand-rolled inline-SVG charting (no CDN
  dependencies)
- **Deployment**: Docker (train-if-missing container entrypoint)
- **Live market integration**: real historical data via `yfinance`, a REST
  client for a real broker's paper-trading API (Alpaca) with a hard-coded
  sandbox endpoint, `.env`-based credential loading, and loud-not-silent
  error handling
- **Testing**: pytest (93 tests covering feature engineering, both trading
  model families including PyTorch save/load round-trips, the strategy's
  no-lookahead shift behavior, LoRA fine-tuning and its eval-harness gate,
  the backtester, walk-forward, risk limits, volatility modeling, the RAG
  pipeline, the FastAPI dashboard/ML-signal API, real data loading, and
  paper trading -- all mocked where it touches a network/account, no live
  credentials needed to run the suite)

## Layout

```
python/
  quantiq/
    data.py                # synthetic OHLCV generator + real (yfinance) loader
    strategies.py           # MA crossover, mean reversion, signal overlay, MLSignalStrategy
    engine.py                 # vectorized backtest engine
    walk_forward.py             # sequential out-of-sample fold evaluation
    risk.py                       # position limits + drawdown/daily-loss kill switch
    volatility.py                   # ARIMA/GARCH volatility modeling
    metrics.py                        # Sharpe, CAGR, drawdown, win rate, Monte Carlo VaR
    paper_trading.py                    # Alpaca paper-trading REST client
    paper_runner.py                       # rebalances a paper account to a strategy's target position
    ml/
      features.py     # technical features + next-day-direction labels
      model.py          # SklearnSignalModel, TorchSignalModel (GRU)
      train.py            # chronological train/test split, MLflow logging, model selection
      eval_harness.py       # standalone model-quality gate
      registry.py             # loads whichever model train.py selected
    rag/
      retriever.py    # TF-IDF retrieval
      signal.py         # lexicon/LLM/finetuned scoring -> daily signal
      llm.py              # OpenAI-compatible client
    finetune/
      data.py         # real financial-tweet sentiment dataset (HF Hub)
      model.py          # LoRA adapter inference wrapper
      train.py            # LoRA fine-tuning of DistilBERT, MLflow logging
      eval_harness.py       # standalone model-quality gate
    web/
      app.py          # FastAPI dashboard + ML-signal API
      static/         # index.html, app.js (hand-rolled SVG charts), style.css
    cli.py             # end-to-end demo
  tests/
  Dockerfile           # containerized dashboard (train-if-missing entrypoint)
data/
  sample_docs/        # synthetic sample corpus (fictional ticker ACME)
```

## Run it

```bash
cd python
pip install -r requirements.txt
python -m pytest tests/ -v

python -m quantiq.ml.train                              # train on synthetic data
python -m quantiq.ml.train --real-data --ticker AAPL --period 5y   # or real data
python -m quantiq.ml.eval_harness --update-baseline      # quality gate + record baseline

python -m quantiq.cli                                    # full demo: backtest, walk-forward, VaR,
                                                          # GARCH, risk limits, ML signal
python -m quantiq.cli --real-data --ticker AAPL          # same demo, real Yahoo Finance history
python -m quantiq.cli --sentiment-backend llm            # score RAG docs with a local LLM (needs Ollama)
python -m quantiq.cli --sentiment-backend finetuned       # score RAG docs with the fine-tuned model
```

### LLM fine-tuning

```bash
cd python
python -m quantiq.finetune.train                         # LoRA fine-tune DistilBERT (~2 min, CPU)
python -m quantiq.finetune.eval_harness --update-baseline # quality gate + record baseline
```

### Docker

```bash
docker build -t quantiq-dashboard python/
docker run -p 8080:8080 quantiq-dashboard
```

### Web dashboard

```bash
cd python
uvicorn quantiq.web.app:app --reload --port 8080
# then open http://localhost:8080
```

### Paper trading (real market data, no real capital)

```bash
cd python
# once: sign up free (paper trading needs no funding/ID verification for
# that alone), generate an API key pair at
# https://app.alpaca.markets/paper/dashboard/overview
cp .env.example .env      # then fill in ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY
                           # (.env is gitignored -- never committed, loaded automatically)

python -m quantiq.paper_runner --ticker AAPL --strategy ml_signal --dry-run
python -m quantiq.paper_runner --ticker AAPL --strategy ml_signal   # actually submits the paper order
```

## Roadmap

- Larger, real financial-document corpus for the RAG layer (the sample
  corpus only covers a fictional ticker).
- Cross-ticker generalization testing -- train on one ticker's history,
  evaluate on a different one entirely, as a stronger disjointness
  guarantee for real-data models than re-fetching the same ticker.
- Hyperparameter tuning (currently fixed architecture/learning rate for
  the GRU, fixed depth/iterations for gradient boosting).
- Richer features: cross-asset signals, order-flow-style features, longer
  lookback windows for the sequence model.
- Managed identity (instead of ACR admin credentials) for the Container
  App's registry pull, once the deploy principal has a role scoped to grant
  that without needing subscription-level elevated permissions.
