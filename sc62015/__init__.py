import os
import sys

third_party = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third-party")
if os.path.isdir(third_party) and third_party not in sys.path:
    sys.path.insert(0, third_party)

