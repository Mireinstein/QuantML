# QuantIQ

A quant research platform combining a low-latency C++ limit order book with a
Python backtesting engine and a retrieval-augmented (RAG) signal pipeline.
Built to demonstrate the core skill areas quant development roles
(market-making/HFT firms, systematic trading desks) actually test for:
low-latency systems programming, concurrency, networking, market
microstructure, quantitative statistics, and rigorous strategy backtesting —
plus applied LLM/RAG engineering on top.

**This is a research/backtesting project, not a live trading system.** It
does not connect to a brokerage or execute real trades, and the price data
and news/filing corpus used in the demo are synthetic (see below) — no
financial advice, no claims about real-market performance.

## Components

### 1. `cpp/` — limit order book engine + concurrency + networking

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

**SIMD**: `volatility.cpp` computes rolling volatility with an ARM NEON-
vectorized kernel (portable scalar fallback on non-ARM), verified numerically
identical to the scalar path across many window sizes including remainder
cases (`make vol-test`). Measured speedup on this machine (`make vol-bench`,
2M points, 20-period window): **3.5x**.

**Networking**: `feed_publisher`/`feed_subscriber` are a real UDP market-data
feed — a publisher sends serialized tick messages over a socket, a subscriber
receives them, submits each as an order to the real `OrderBook`, and reports
actual measured network+processing latency and any packet loss (UDP has no
delivery guarantee; the subscriber detects gaps via sequence numbers, though
it doesn't implement gap-recovery/resync). Loopback unicast rather than
multicast — multicast group membership needs network configuration that
isn't portable in a generic dev sandbox; swapping in a multicast join is a
socket-option change, not a different architecture. Measured on this machine
at 1M ticks: p50 ~14.6μs, zero loss.

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

**Risk management** (`risk.py`): `RiskLimits` + `apply_risk_limits` enforce
a max position size and a max-drawdown (or max-daily-loss) kill switch —
this actually forces the strategy flat once breached and keeps it flat, it
doesn't just report the breach. Demonstrated live in `cli.py`: a 10%
max-drawdown limit on the baseline strategy trips on a real synthetic
drawdown day, capping both further losses *and* the subsequent recovery
(final equity 1.12 with the limit vs. 1.62 unlimited) — the real tradeoff
risk limits impose, not a cherry-picked win.

**Volatility modeling** (`volatility.py`): ARIMA (statsmodels) for the
conditional mean, GARCH(1,1) (`arch`) for volatility clustering. Tested
against a simulated ground-truth GARCH process, not just "it runs": the
fitted model's in-sample conditional volatility correlates >0.5 with the
*true* simulated variance path, and correctly forecasts higher volatility
for a deliberately turbulent regime than a calm one.

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
- **SIMD**: ARM NEON intrinsics for vectorized rolling volatility (3.5x
  measured speedup), portable scalar fallback
- **Networking**: BSD sockets (UDP), wire serialization, sequence-based
  packet-loss detection
- **C++/Python interop**: pybind11 bindings exposing the real matching
  engine to Python
- **Quant research**: Python, pandas, numpy — vectorized backtesting,
  walk-forward evaluation, Monte Carlo VaR/CVaR, ARIMA/GARCH volatility
  modeling, position limits + drawdown kill switches, transaction-cost
  modeling, lookahead-safe signal shifting
- **Applied ML/RAG**: scikit-learn (TF-IDF + cosine similarity retrieval),
  pydantic-validated structured LLM output, OpenAI-compatible LLM client
  with graceful fallback
- **Testing**: C++ unit tests (order book correctness, price-time priority,
  concurrent ring-buffer stress test, SIMD-vs-scalar equivalence), pytest
  (37 tests covering backtester, metrics, execution simulation,
  walk-forward, risk limits, volatility modeling, and the RAG pipeline)

## Layout

```
cpp/
  include/              # OrderBook, types, ring buffer, volatility, feed protocol
  src/                   # OrderBook, benchmarks, mt bench, SIMD volatility, UDP feed, pybind11 bindings
  tests/                 # order book, ring buffer, and volatility correctness tests
python/
  quantiq_cpp*.so        # compiled extension module (build with setup.py)
  setup.py               # builds the pybind11 extension
  quantiq/
    data.py              # synthetic OHLCV generator
    tick_data.py          # synthetic tick-level price path for execution sim
    strategies.py         # MA crossover, mean reversion, signal overlay
    engine.py              # vectorized backtest engine
    execution.py            # real order-book execution simulation (pybind11)
    walk_forward.py          # sequential out-of-sample fold evaluation
    risk.py                   # position limits + drawdown/daily-loss kill switch
    volatility.py              # ARIMA/GARCH volatility modeling
    metrics.py                  # Sharpe, CAGR, drawdown, win rate, Monte Carlo VaR
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

### C++ order book, ring buffer, SIMD, and networking

```bash
cd cpp
make test        # order book correctness tests
make ring-test   # lock-free ring buffer correctness + 2M-item concurrent stress test
make vol-test    # SIMD-vs-scalar volatility correctness tests
make vol-bench   # SIMD vs scalar speedup benchmark
make bench       # single-threaded submit() latency benchmark
make mt-bench    # multithreaded producer/consumer benchmark
make feed        # builds feed_publisher and feed_subscriber

# run the UDP feed demo (two processes):
./bin/feed_subscriber 9001 100000 &
./bin/feed_publisher 9001 100000
```

### Python backtester, execution sim, and RAG demo

```bash
cd python
pip install -r requirements.txt
python3 setup.py build_ext --inplace   # builds quantiq_cpp (pybind11 extension)
python -m pytest tests/ -v
python -m quantiq.cli                 # full demo: backtest, execution sim, walk-forward,
                                       # VaR, GARCH, risk limits
python -m quantiq.cli --use-llm       # score docs with a local LLM (requires Ollama running)
```

## Roadmap

- Paper-trading integration (e.g. Alpaca sandbox API) to run strategies
  against live simulated market data — still no real capital.
- Real historical market data source, swapped in behind the same
  `generate_synthetic_ohlcv` interface.
- Larger, real financial-document corpus for the RAG layer.
- UDP feed gap-recovery/snapshot resync (loss is detected, not yet healed).
