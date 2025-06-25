#!/usr/bin/env fish

function build_and_run
  ruff check .
  pyright sc62015/pysc62015
  # pytest -vv
  pytest --cov=sc62015/pysc62015 --cov-report=term-missing
end

build_and_run

while fswatch -1 .
  reset
  build_and_run
  sleep 1
end
