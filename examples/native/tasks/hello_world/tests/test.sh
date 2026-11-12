#!/bin/sh
set -eu

test "$(cat /hello 2>/dev/null || true)" = "ok"
printf '{"reward":1}\n' >reward.json
