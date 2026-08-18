# Build context is python/:  docker build -t quantml-dashboard python/
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Trains the model at BUILD time, not at container startup. An earlier
# version trained on first boot (mirroring TenantIQ's ml/Dockerfile, whose
# lightweight sklearn model is cheap enough for that to work fine) -- but
# this project's training step loads PyTorch + transformers + scikit-learn
# together and genuinely needs more memory than a small serving container
# should carry at runtime. Deployed to Azure Container Apps at 0.5Gi, that
# runtime-training approach OOM-crash-looped on every cold start
# (confirmed via `az containerapp replica list` showing restartCount
# climbing with the container never reaching a ready state). Training
# inside the image build instead uses the build host's full resources, and
# bakes an immutable, reproducible model artifact into the image -- a
# better practice regardless of the memory constraint that surfaced it.
#
# REAL_DATA (empty by default -- synthetic, fast, no network call) is
# what ci.yml's docker-smoke-test builds with on every push, since its
# job is "does the container serve," not "is the model good." Real
# deployments (az acr build, retrain-eval.yml) pass
# --build-arg REAL_DATA=1 so the artifact that actually ships was trained
# on real market data -- an earlier version of this Dockerfile always
# trained on synthetic data regardless of caller, silently disconnecting
# the real-data training/eval pipeline in retrain-eval.yml from what was
# actually deployed. Confirmed live: the deployed dashboard's
# /api/ml-signal reported "data_source": "synthetic" despite
# retrain-eval.yml training and gating a real-data model every week.
#
# Three separate single-token args, not one free-form TRAIN_ARGS string
# -- `az acr build --build-arg` reconstructs a command line on its remote
# build agent, and a value containing its own spaces/flags (e.g.
# "--real-data --ticker AAPL --period 5y") gets re-split there, so
# "--ticker" ends up parsed as a top-level docker-build flag instead of
# part of the arg's value ("unknown flag: --ticker"). Confirmed via a
# failed `az acr build` run.
ARG REAL_DATA=""
ARG TICKER="AAPL"
ARG PERIOD="5y"
RUN if [ -n "$REAL_DATA" ]; then \
      python3 -m quantml.ml.train --real-data --ticker "$TICKER" --period "$PERIOD"; \
    else \
      python3 -m quantml.ml.train; \
    fi

EXPOSE 8080
CMD ["uvicorn", "quantml.web.app:app", "--host", "0.0.0.0", "--port", "8080"]
