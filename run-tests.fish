#!/usr/bin/env fish

set -x MYPYPATH (pwd)/stubs:~/Applications/Binary\ Ninja.app/Contents/Resources/python/

function build_and_run
  ruff check .
  mypy sc62015/pysc62015
  pytest -vv
end

build_and_run

while fswatch -1 .
  reset
  build_and_run
  sleep 1
end

