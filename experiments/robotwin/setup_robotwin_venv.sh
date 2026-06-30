#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

ROBOTWIN_VENV=${ROBOTWIN_VENV:-.venv-robotwin}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-third_party/RoboTwin}
PYTHON_VERSION=${PYTHON_VERSION:-$(cat "${PROJECT_ROOT}/.python-version")}
CUROBO_VERSION=${CUROBO_VERSION:-v0.7.5}

case "${ROBOTWIN_VENV}" in
  /*) VENV_DIR="${ROBOTWIN_VENV}" ;;
  *) VENV_DIR="${PROJECT_ROOT}/${ROBOTWIN_VENV}" ;;
esac

case "${ROBOTWIN_ROOT}" in
  /*) ROBOTWIN_ROOT_DIR="${ROBOTWIN_ROOT}" ;;
  *) ROBOTWIN_ROOT_DIR="${PROJECT_ROOT}/${ROBOTWIN_ROOT}" ;;
esac

PYTHON_BIN="${VENV_DIR}/bin/python"
CUROBO_DIR="${ROBOTWIN_ROOT_DIR}/envs/curobo"

cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" || ! -f "${PROJECT_ROOT}/uv.lock" ]]; then
  echo "Run this script from a checkout containing pyproject.toml and uv.lock." >&2
  exit 1
fi

if [[ ! -f "${ROBOTWIN_ROOT_DIR}/script/eval_policy.py" ]]; then
  echo "RoboTwin root is missing script/eval_policy.py: ${ROBOTWIN_ROOT_DIR}" >&2
  echo "Clone RoboTwin first or set ROBOTWIN_ROOT=/path/to/RoboTwin." >&2
  exit 1
fi

if [[ -x "${PYTHON_BIN}" ]]; then
  echo "Reusing RoboTwin virtualenv: ${VENV_DIR}"
else
  echo "Creating RoboTwin virtualenv: ${VENV_DIR}"
  uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
fi

echo "Syncing project dependencies into RoboTwin virtualenv."
UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --index-strategy unsafe-best-match

echo "Installing RoboTwin simulator dependencies."
uv pip install --python "${PYTHON_BIN}" "sapien==3.0.0b1" "warp-lang==0.11.0"

if [[ ! -d "${CUROBO_DIR}" ]]; then
  echo "Cloning cuRobo ${CUROBO_VERSION} into ${CUROBO_DIR}."
  git clone --branch "${CUROBO_VERSION}" https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
fi

if [[ ! -f "${CUROBO_DIR}/pyproject.toml" ]]; then
  echo "cuRobo source is missing pyproject.toml: ${CUROBO_DIR}" >&2
  exit 1
fi

echo "Installing cuRobo from source."
uv pip install --python "${PYTHON_BIN}" "pip==25.3" "vcs-versioning==2.2.0"
# cuRobo v0.7.5 builds a wheel whose filename version and metadata version can
# disagree under uv's stricter wheel validation, while pip installs the source
# package correctly with the same no-build-isolation constraints.
"${PYTHON_BIN}" -m pip install "${CUROBO_DIR}" --no-build-isolation

echo "Verifying RoboTwin virtualenv."
"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

expected = {
    "sapien": "3.0.0b1",
    "warp-lang": "0.11.0",
}
for package_name, expected_version in expected.items():
    actual_version = version(package_name)
    if actual_version != expected_version:
        raise SystemExit(f"{package_name}=={actual_version}, expected {expected_version}")

import curobo  # noqa: F401
import g05  # noqa: F401
import sapien  # noqa: F401
import sapien.physx  # noqa: F401
import torch
import warp  # noqa: F401

print(f"sapien=={version('sapien')}")
print(f"warp-lang=={version('warp-lang')}")
print(f"nvidia-curobo=={version('nvidia-curobo')}")
print(f"torch=={torch.__version__}")
print("RoboTwin virtualenv verification passed.")
PY

echo "RoboTwin virtualenv ready: ${PYTHON_BIN}"
