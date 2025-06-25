import os
import sys
import subprocess

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
    from binja_helpers import binja_api  # noqa: F401
    stub_dir = os.path.join(os.path.dirname(__file__), "..", "stubs")
    print(f"Using stubs from {os.path.abspath(stub_dir)}")
else:
    print(f"Using Binary Ninja from {bn_path}")

# Run pyright on the target directory
result = subprocess.run(["pyright", "sc62015/pysc62015"], capture_output=True, text=True)
print(result.stdout, end="")
print(result.stderr, end="", file=sys.stderr)
sys.exit(result.returncode)

