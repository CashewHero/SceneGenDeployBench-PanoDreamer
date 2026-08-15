#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

safe_name() {
  tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_.-]/-/g'
}

repo_name="$(basename "${REPO_ROOT}" | safe_name)"

IMAGE="${RUNNER_IMAGE:-${repo_name}-runner:local}"
CONTAINER="${RUNNER_CONTAINER:-${repo_name}-runner-localtest}"
HOST_PORT="${RUNNER_HOST_PORT:-58090}"
DATA_DIR="${RUNNER_DATA_DIR:-${REPO_ROOT}/data}"
RUNNER_NAME="${RUNNER_NAME:-${repo_name}-runner}"
RUNNER_TYPE="${RUNNER_TYPE:-generator}"
RUNNER_VERSION="${RUNNER_VERSION:-0.1.0}"
RUNNER_ADAPTER="${RUNNER_ADAPTER:-runner_wrapper.adapter:run_job}"
REQUEST_FILE="${RUNNER_REQUEST_FILE:-${SCRIPT_DIR}/examples/${RUNNER_TYPE}_job_request.json}"

usage() {
  cat <<EOF
Usage:
  runner_wrapper/localtest.sh test
  runner_wrapper/localtest.sh build
  runner_wrapper/localtest.sh run
  runner_wrapper/localtest.sh smoke
  runner_wrapper/localtest.sh status
  runner_wrapper/localtest.sh logs
  runner_wrapper/localtest.sh down

Environment:
  RUNNER_IMAGE=${IMAGE}
  RUNNER_CONTAINER=${CONTAINER}
  RUNNER_HOST_PORT=${HOST_PORT}
  RUNNER_TYPE=${RUNNER_TYPE}
  RUNNER_NAME=${RUNNER_NAME}
  RUNNER_VERSION=${RUNNER_VERSION}
  RUNNER_ADAPTER=${RUNNER_ADAPTER}
  RUNNER_REQUEST_FILE=${REQUEST_FILE}
  RUNNER_DATA_DIR=${DATA_DIR}

For the bundled test adapter, set TEST_RUNNER_MIN_SECONDS=0 and
TEST_RUNNER_MAX_SECONDS=0 when you want a fast smoke run.
EOF
}

require_tools() {
  command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }
  command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
}

run_tests() {
  PYTHONPATH="${REPO_ROOT}" \
    python3 -m unittest discover -s "${SCRIPT_DIR}/tests" -v
}

build_image() {
  run_tests
  docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE}" \
    "${REPO_ROOT}"
}

prepare_data() {
  mkdir -p \
    "${DATA_DIR}/datasets/smoke" \
    "${DATA_DIR}/model_cache" \
    "${DATA_DIR}/pipelines" \
    "${DATA_DIR}/output/my-generator@0.1.0/smoke-dataset/sample-1"

  if [[ ! -f "${DATA_DIR}/datasets/smoke/image.png" ]]; then
    printf 'smoke input\n' > "${DATA_DIR}/datasets/smoke/image.png"
  fi

  if [[ ! -f "${DATA_DIR}/datasets/smoke/reference.png" ]]; then
    printf 'smoke reference\n' > "${DATA_DIR}/datasets/smoke/reference.png"
  fi

  if [[ ! -f "${DATA_DIR}/output/my-generator@0.1.0/smoke-dataset/sample-1/scene.glb" ]]; then
    printf 'smoke generated scene\n' > "${DATA_DIR}/output/my-generator@0.1.0/smoke-dataset/sample-1/scene.glb"
  fi
}

run_container() {
  prepare_data
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

  local env_args=(
    -e "RUNNER_PORT=58090"
    -e "RUNNER_NAME=${RUNNER_NAME}"
    -e "RUNNER_TYPE=${RUNNER_TYPE}"
    -e "RUNNER_VERSION=${RUNNER_VERSION}"
    -e "RUNNER_CONTRACT_VERSION=1"
    -e "RUNNER_ADAPTER=${RUNNER_ADAPTER}"
    -e "PATH_DATASETS=/data/datasets"
    -e "PATH_MODEL_CACHE=/data/model_cache"
    -e "PATH_OUTPUT=/data/output"
    -e "PATH_PIPELINES=/data/pipelines"
  )

  if [[ -n "${TEST_RUNNER_MIN_SECONDS:-}" ]]; then
    env_args+=(-e "TEST_RUNNER_MIN_SECONDS=${TEST_RUNNER_MIN_SECONDS}")
  fi
  if [[ -n "${TEST_RUNNER_MAX_SECONDS:-}" ]]; then
    env_args+=(-e "TEST_RUNNER_MAX_SECONDS=${TEST_RUNNER_MAX_SECONDS}")
  fi
  if [[ -n "${RUNNER_LOG_LEVEL:-}" ]]; then
    env_args+=(-e "RUNNER_LOG_LEVEL=${RUNNER_LOG_LEVEL}")
  fi

  docker run -d \
    --name "${CONTAINER}" \
    -p "${HOST_PORT}:58090" \
    "${env_args[@]}" \
    -v "${DATA_DIR}:/data" \
    "${IMAGE}" >/dev/null

  wait_ready
  echo "runner available at http://127.0.0.1:${HOST_PORT}"
}

wait_ready() {
  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/status" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "runner did not become ready" >&2
  docker logs "${CONTAINER}" >&2 || true
  exit 1
}

submit_request() {
  [[ -f "${REQUEST_FILE}" ]] || { echo "missing request file: ${REQUEST_FILE}" >&2; exit 1; }
  curl -fsS \
    -X POST "http://127.0.0.1:${HOST_PORT}/run-job" \
    -H 'Content-Type: application/json' \
    --data @"${REQUEST_FILE}"
  echo
}

status_json() {
  curl -fsS "http://127.0.0.1:${HOST_PORT}/status"
}

status_field() {
  python3 -c 'import json, sys; print(json.load(sys.stdin).get(sys.argv[1]) or "")' "$1"
}

poll_terminal() {
  local attempt state
  for attempt in $(seq 1 3600); do
    state="$(status_json | status_field state)"
    case "${state}" in
      finished)
        status_json
        echo
        return 0
        ;;
      failed)
        status_json
        echo
        return 1
        ;;
    esac
    sleep 1
  done

  echo "runner job did not finish before local poll timeout" >&2
  return 1
}

smoke() {
  build_image
  run_container
  submit_request
  poll_terminal
}

main() {
  command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
  if [[ "${1:-}" == "test" ]]; then
    run_tests
    return
  fi

  require_tools
  case "${1:-smoke}" in
    build)
      build_image
      ;;
    run)
      build_image
      run_container
      ;;
    smoke)
      smoke
      ;;
    status)
      status_json
      echo
      ;;
    logs)
      docker logs -f "${CONTAINER}"
      ;;
    down)
      docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "unknown command: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
