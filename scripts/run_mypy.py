import os
import sys
from mypy import api

# Ensure repository root is in sys.path so sc62015 can be imported when the
# script is run from the 'scripts' directory.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

bn_path = os.path.expanduser(
    "~/Applications/Binary Ninja.app/Contents/Resources/python/"
)
if os.path.isdir(bn_path) and bn_path not in sys.path:
    sys.path.append(bn_path)

try:
    import binaryninja  # noqa: F401
    has_binja = True
except ImportError:
    has_binja = False

if not has_binja:
    from sc62015.pysc62015 import binja_api  # noqa: F401
    stub_dir = os.path.join(os.path.dirname(__file__), "..", "stubs")
    os.environ["MYPYPATH"] = os.path.abspath(stub_dir)
    print(f"Using stubs from {os.environ['MYPYPATH']}")
else:
    os.environ["MYPYPATH"] = bn_path
    print(f"Using Binary Ninja from {bn_path}")

stdout, stderr, exit_status = api.run(["sc62015/pysc62015"])
print(stdout, end="")
print(stderr, end="", file=sys.stderr)
sys.exit(exit_status)
