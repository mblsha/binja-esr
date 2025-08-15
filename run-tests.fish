#!/usr/bin/env fish

function build_and_run
  # Set environment for using mock Binary Ninja API when real Binary Ninja is installed
  set -x FORCE_BINJA_MOCK 1
  
  uv run ruff check .
  uv run ruff format --check .
  uv run pyright sc62015/pysc62015
  # Run tests for sc62015, pce500, and web modules
  uv run pytest --cov=sc62015/pysc62015 --cov-report=term-missing
  uv run pytest pce500/tests/ --cov=pce500 --cov-report=term-missing
  uv run pytest web/tests/ --cov=web --cov-report=term-missing
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
