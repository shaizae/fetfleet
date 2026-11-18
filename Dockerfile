# syntax=docker/dockerfile:1
# Reproduces the ubuntu-latest jobs of .github/workflows/tests.yml: an Ubuntu image with every
# python-version of the matrix, each in its own venv, each with the same `pip install -e ".[dev]"`.
#
# Use scripts/docker_test.py (runs the whole os x python matrix of tests.yml) rather than by hand:
#   docker build --build-arg PYTHON_VERSIONS="3.11 3.13" -t fitfleet-tests .
#   docker run --rm --shm-size=2g fitfleet-tests                          # every version
#   docker run --rm --shm-size=2g -e PYTHON_VERSIONS=3.11 fitfleet-tests -k mixed -x
#
# Works on Docker Desktop for Windows, macOS (Intel and Apple Silicon) and Linux: the image is
# built for the host's CPU (amd64 or arm64), no emulation needed.
ARG UBUNTU_VERSION=24.04

FROM ghcr.io/astral-sh/uv:0.5.11 AS uv

FROM ubuntu:${UBUNTU_VERSION}
ARG PYTHON_VERSIONS="3.11 3.13"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PYTHON_VERSIONS=${PYTHON_VERSIONS}

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu ships only one Python; uv installs the exact versions of the matrix (like setup-python does)
COPY --from=uv /uv /usr/local/bin/uv
RUN uv python install ${PYTHON_VERSIONS} \
 && for v in ${PYTHON_VERSIONS}; do uv venv --seed --python "$v" "/venvs/py$v" || exit 1; done

WORKDIR /app
COPY . .

# the workflow's install step, once per Python version; the cache mount keeps rebuilds fast
RUN --mount=type=cache,target=/root/.cache/pip \
    for v in ${PYTHON_VERSIONS}; do "/venvs/py$v/bin/pip" install -e ".[dev]" || exit 1; done

# the workflow's test step for every version in $PYTHON_VERSIONS; extra `docker run` args go to pytest
ENTRYPOINT ["bash", "-c", "rc=0; for v in $PYTHON_VERSIONS; do echo \"========== ubuntu / Python $v ==========\"; \"/venvs/py$v/bin/python\" -m pytest -q \"$@\" || rc=1; done; exit $rc", "run-tests"]
