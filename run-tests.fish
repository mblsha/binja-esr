#!/usr/bin/env fish

function build_and_run
  # Set environment for using mock Binary Ninja API
  set -x FORCE_BINJA_MOCK 1
  
  ruff check .
  pyright sc62015/pysc62015
  # pytest -vv
  pytest --cov=sc62015/pysc62015 --cov-report=term-missing
end

# Check for --once flag
if test "$argv[1]" = "--once"
  build_and_run
else
  build_and_run
  
  while fswatch -1 .
    reset
    build_and_run
    sleep 1
  end
end
