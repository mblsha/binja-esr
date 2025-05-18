#!/usr/bin/env fish

set -x MYPYPATH ~/Applications/Binary\ Ninja.app/Contents/Resources/python/
cd sc62015

function build_and_run
  ruff check
  pytest -vv
  mypy pysc62015
end

build_and_run

while fswatch -1 .
  reset
  build_and_run
  sleep 1
end

