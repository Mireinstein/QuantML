# QuantIQ

A quant research platform combining a low-latency C++ limit order book with a
Python backtesting engine and a retrieval-augmented (RAG) signal pipeline.
Built to demonstrate the core skill areas quant development roles
(market-making/HFT firms, systematic trading desks) actually test for:
low-latency systems programming, market microstructure, and rigorous
strategy backtesting — plus applied LLM/RAG engineering on top.

**This is a research/backtesting project, not a live trading system.** It
does not connect to a brokerage or execute real trades, and the price data
and news/filing corpus used in the demo are synthetic (see below) — no
financial advice, no claims about real-market performance.

## Components

### 1. `cpp/` — limit order book engine

A price-time priority matching engine (`OrderBook`): submit, match, cancel,
best bid/ask. Includes a latency benchmark harness that measures real
per-order `submit()` latency (p50/p90/p99/p99.9/max) under random order flow.

Sample run (1M orders, this machine):

```
Orders processed: 1000000
Latency per submit() call (ns):
  p50:   208
  p90:   458
  p99:   875
  p99.9: 3500
```

### 2. `python/quantiq/` — backtesting engine

Vectorized backtester: strategies emit a target position per bar (no
lookahead — positions are shifted forward one bar before being applied to
returns), the engine nets out transaction costs and produces an equity
curve. Metrics: Sharpe ratio, CAGR, max drawdown, win rate.

Includes two baseline technical strategies (moving-average crossover, mean
reversion) and a `SignalOverlayStrategy` that blends a base strategy's
position with an external signal series.

### 3. `python/quantiq/rag/` — retrieval-augmented signal layer

TF-IDF retrieval (`Retriever`) over a small corpus of sample financial
documents (`data/sample_docs/`), turned into a per-day sentiment signal
(`build_signal`) that feeds into `SignalOverlayStrategy`. Two scoring
backends:

- **lexicon** (default): deterministic keyword-based scoring, zero
  dependencies, fully reproducible.
- **llm**: sends each retrieved document to an OpenAI-compatible chat
  endpoint (defaults to a local, free Ollama model, same pattern as
  [TenantIQ](https://github.com/Mireinstein/TenantIQ)) for a structured
  `{score, rationale}` response, validated with pydantic. Falls back to the
  lexicon scorer per-document if the endpoint isn't reachable.

**On the sample corpus**: the 10 documents in `data/sample_docs/` are
synthetic text I wrote for a fictional ticker (`ACME`) so the retrieval →
signal → backtest pipeline is fully runnable without a licensed news/filings
feed. They are not real news.

## Technologies

- **Low-latency systems**: C++17, price-time priority matching, `std::map`
  price ladders, latency percentile benchmarking
- **Quant research**: Python, pandas, numpy — vectorized backtesting,
  Sharpe/CAGR/drawdown, transaction-cost modeling, lookahead-safe signal
  shifting
- **Applied ML/RAG**: scikit-learn (TF-IDF + cosine similarity retrieval),
  pydantic-validated structured LLM output, OpenAI-compatible LLM client
  with graceful fallback
- **Testing**: C++ unit tests (order book correctness, price-time priority),
  pytest (14 tests covering backtester, metrics, and RAG pipeline)

## Layout

```
cpp/
  include/          # OrderBook, types
  src/               # OrderBook impl + latency benchmark (main.cpp)
  tests/             # order book correctness tests
python/
  quantiq/
    data.py          # synthetic OHLCV generator
    strategies.py    # MA crossover, mean reversion, signal overlay
    engine.py         # backtest engine
    metrics.py         # Sharpe, CAGR, drawdown, win rate
    rag/
      retriever.py    # TF-IDF retrieval
      signal.py        # lexicon/LLM scoring -> daily signal
      llm.py            # OpenAI-compatible client
    cli.py             # end-to-end demo
  tests/
data/
  sample_docs/        # synthetic sample corpus (fictional ticker ACME)
```

## Run it

### C++ order book + latency benchmark

```bash
cd cpp
make test      # builds + runs correctness tests
make bench     # builds the benchmark
./bin/bench 1000000
```

### Python backtester + RAG demo

```bash
cd python
pip install -r requirements.txt
python -m pytest tests/ -v
python -m quantiq.cli                 # lexicon-scored sentiment overlay
python -m quantiq.cli --use-llm       # score docs with a local LLM (requires Ollama running)
```

## Roadmap

- Paper-trading integration (e.g. Alpaca sandbox API) to run strategies
  against live simulated market data — still no real capital.
- Real historical market data source, swapped in behind the same
  `generate_synthetic_ohlcv` interface.
- Larger, real financial-document corpus for the RAG layer.

## Skills roadmap (planned, not yet implemented)

The order book and backtester above are real and tested; this is the next
phase, aimed squarely at what market-making/prop-trading firms (Jane
Street, Citadel, HRT, etc.) actually screen for:

- **Concurrency & lock-free structures** — multithreaded order matching,
  a lock-free ring buffer for market data ingestion (replacing the current
  single-threaded benchmark loop).
- **Performance engineering** — cache-line-aware data layout, SIMD where it
  applies, `perf`-driven profiling with before/after latency numbers.
- **Market data plumbing** — a tick-by-tick feed simulator with
  UDP/multicast-style delivery, feeding the order book under realistic
  load.
- **Quantitative statistics** — time-series volatility modeling
  (ARIMA/GARCH), Monte Carlo simulation for risk (VaR), proper walk-forward
  backtesting instead of a single in-sample run.
- **Risk management module** — position limits, real-time VaR, kill
  switches on the strategy layer.
- **C++/Python boundary** — pybind11 bindings so the Python backtester can
  drive the real C++ order book instead of a separate simulated fill model.
