#!/usr/bin/env bash
# Convenience wrapper: ./run_all.sh [--quick]
set -e
cd "$(dirname "$0")"
python run_all.py "$@"
