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
RUN python3 -m quantml.ml.train

EXPOSE 8080
CMD ["uvicorn", "quantml.web.app:app", "--host", "0.0.0.0", "--port", "8080"]
