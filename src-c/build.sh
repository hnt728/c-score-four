#!/usr/bin/env bash
set -euo pipefail

gcc -shared -fPIC -O3 -o libscorefour.so engine.c
echo "built: $(pwd)/libscorefour.so"
