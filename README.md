# QuantIQ

A quant research platform combining a low-latency C++ limit order book with a
Python backtesting engine and a retrieval-augmented (RAG) signal pipeline.
Built to demonstrate the core skill areas quant development roles
(market-making/HFT firms, systematic trading desks) actually test for:
low-latency systems programming, concurrency, market microstructure, and
rigorous strategy backtesting — plus applied LLM/RAG engineering on top.

**This is a research/backtesting project, not a live trading system.** It
does not connect to a brokerage or execute real trades, and the price data
and news/filing corpus used in the demo are synthetic (see below) — no
financial advice, no claims about real-market performance.

## Components

### 1. `cpp/` — limit order book engine + concurrency

A price-time priority matching engine (`OrderBook`): submit, match, cancel,
best bid/ask. Includes a latency benchmark harness that measures real
per-order `submit()` latency (p50/p90/p99/p99.9/max) under random order flow.

Sample run (1M orders, this machine):

```
Orders processed: 1000000
Latency per submit() call (ns):
  p50:   125
  p90:   292
  p99:   584
  p99.9: 1917
```

**Concurrency**: a lock-free single-producer single-consumer ring buffer
(`ring_buffer.hpp`) for market-data/order ingestion — cache-line-aligned
head/tail atomics, acquire/release ordering, no locks. Correctness is
verified under real concurrent load: `make ring-test` runs a 2M-item
producer/consumer stress test asserting zero data loss and strict FIFO
ordering across threads, not just single-threaded logic.

`make mt-bench` runs a producer thread pushing orders through the ring
buffer into a consumer thread that submits them to the OrderBook, measuring
end-to-end handoff latency. Honest caveat: in this sandboxed dev environment
that cross-thread latency is dominated by OS/hypervisor scheduling jitter
(double-digit milliseconds), not the data structure itself — the ring
buffer's own overhead is proven correct and lock-free by the stress test,
but a clean latency number for the cross-thread handoff requires pinning
threads to isolated cores on dedicated hardware, which isn't available here.
I'm reporting this rather than hiding it.

### 2. `python/quantiq/` — backtesting engine

Vectorized backtester: strategies emit a target position per bar (no
lookahead — positions are shifted forward one bar before being applied to
returns), the engine nets out transaction costs and produces an equity
curve. Metrics: Sharpe ratio, CAGR, max drawdown, win rate, and
**Monte Carlo VaR/CVaR** (historical bootstrap resampling of the strategy's
own realized returns — no assumed distribution).

**Walk-forward evaluation** (`walk_forward.py`) splits the backtest into
sequential out-of-sample folds instead of reporting one full-period Sharpe
that a single lucky/unlucky stretch can dominate. Real finding from this
project's own baseline strategy: full-period Sharpe looks fine (0.84), but
walk-forward reveals high variance across folds (std of fold Sharpe: 1.77,
worst fold: -1.999) — exactly the kind of fragility a single backtest number
hides and walk-forward is supposed to catch.

Includes two baseline technical strategies (moving-average crossover, mean
reversion) and a `SignalOverlayStrategy` that blends a base strategy's
position with an external signal series.

### 3. `python/quantiq/execution.py` — real order-book execution simulation

The backtester above is a vectorized approximation (position × return).
`execution.py` instead drives the **real C++ OrderBook** via pybind11
bindings (`cpp/src/bindings.cpp`, module `quantiq_cpp`): synthetic
tick-level background liquidity is continuously refreshed in the book, and
a strategy's target-position changes are submitted as real marketable
orders that fill against it — so the reported slippage (basis points paid
crossing the spread) is measured from actual matching-engine fills, not
assumed.

### 4. `python/quantiq/rag/` — retrieval-augmented signal layer

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
- **Concurrency**: lock-free SPSC ring buffer (atomics, acquire/release
  memory ordering, cache-line alignment), multithreaded producer/consumer
- **C++/Python interop**: pybind11 bindings exposing the real matching
  engine to Python
- **Quant research**: Python, pandas, numpy — vectorized backtesting,
  walk-forward evaluation, Monte Carlo VaR/CVaR, transaction-cost modeling,
  lookahead-safe signal shifting
- **Applied ML/RAG**: scikit-learn (TF-IDF + cosine similarity retrieval),
  pydantic-validated structured LLM output, OpenAI-compatible LLM client
  with graceful fallback
- **Testing**: C++ unit tests (order book correctness, price-time priority,
  concurrent ring-buffer stress test), pytest (27 tests covering backtester,
  metrics, execution simulation, walk-forward, and the RAG pipeline)

## Layout

```
cpp/
  include/            # OrderBook, types, lock-free ring buffer
  src/                 # OrderBook impl, latency benchmark, mt benchmark, pybind11 bindings
  tests/               # order book + ring buffer correctness tests
python/
  quantiq_cpp*.so      # compiled extension module (build with setup.py)
  setup.py             # builds the pybind11 extension
  quantiq/
    data.py            # synthetic OHLCV generator
    tick_data.py        # synthetic tick-level price path for execution sim
    strategies.py       # MA crossover, mean reversion, signal overlay
    engine.py            # vectorized backtest engine
    execution.py          # real order-book execution simulation (pybind11)
    walk_forward.py        # sequential out-of-sample fold evaluation
    metrics.py              # Sharpe, CAGR, drawdown, win rate, Monte Carlo VaR
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

### C++ order book, ring buffer, and benchmarks

```bash
cd cpp
make test        # order book correctness tests
make ring-test   # lock-free ring buffer correctness + 2M-item concurrent stress test
make bench       # single-threaded submit() latency benchmark
make mt-bench    # multithreaded producer/consumer benchmark
```

### Python backtester, execution sim, and RAG demo

```bash
cd python
pip install -r requirements.txt
python3 setup.py build_ext --inplace   # builds quantiq_cpp (pybind11 extension)
python -m pytest tests/ -v
python -m quantiq.cli                 # lexicon-scored sentiment overlay + execution sim + walk-forward + VaR
python -m quantiq.cli --use-llm       # score docs with a local LLM (requires Ollama running)
```

## Roadmap

- Paper-trading integration (e.g. Alpaca sandbox API) to run strategies
  against live simulated market data — still no real capital.
- Real historical market data source, swapped in behind the same
  `generate_synthetic_ohlcv` interface.
- Larger, real financial-document corpus for the RAG layer.

## Skills roadmap (planned, not yet implemented)

- **Performance engineering** — cache-line-aware data layout beyond the
  ring buffer, SIMD where it applies, `perf`-driven profiling.
- **Market data plumbing** — a tick-by-tick feed simulator with
  UDP/multicast-style delivery (the execution-simulation tick generator is
  in-process/synthetic, not a network feed).
- **Time-series volatility modeling** — ARIMA/GARCH.
- **Risk management module** — position limits and kill switches on the
  strategy layer (Monte Carlo VaR/CVaR is done; enforcement isn't).
