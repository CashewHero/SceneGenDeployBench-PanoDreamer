FROM python:3.12-slim

WORKDIR /app/model_repo

# In model repos, build from the repository root:
# docker build -f runner_wrapper/Dockerfile -t my-runner .
#
# In this repo, the test runner can be built from runner_wrapper/:
# docker build -f runner_wrapper/Dockerfile -t scenegendeploybench-testrunner runner_wrapper
COPY . /app/model_repo

# If the build context is runner_wrapper/ itself, normalize it
RUN if [ -f server.py ] && [ -f adapter.py ] && [ ! -f runner_wrapper/server.py ]; then \
      mkdir /tmp/runner_wrapper; \
      mv ./* /tmp/runner_wrapper/; \
      mv /tmp/runner_wrapper runner_wrapper; \
    fi

ENV PYTHONPATH=/app/model_repo:/app

ARG INSTALL_REQUIREMENTS=1
RUN if [ "$INSTALL_REQUIREMENTS" = "1" ] && [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    fi

ARG INSTALL_EDITABLE=0
RUN if [ "$INSTALL_EDITABLE" = "1" ] && { [ -f pyproject.toml ] || [ -f setup.py ]; }; then \
      pip install --no-cache-dir -e .; \
    fi

EXPOSE 58090

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNER_PORT=58090
ENV RUNNER_NAME=test-runner
ENV RUNNER_TYPE=generator
ENV RUNNER_VERSION=0.1.0
ENV RUNNER_CONTRACT_VERSION=1
ENV RUNNER_IDLE_TIMEOUT_SECONDS=900
ENV RUNNER_ADAPTER=runner_wrapper.adapter:run_job
ENV TEST_RUNNER_MIN_SECONDS=360
ENV TEST_RUNNER_MAX_SECONDS=720

CMD ["python", "-m", "runner_wrapper.server"]
