#!/usr/bin/env python3
"""Test script for git bisect to find PC-E500 emulator regressions.

This script tests two conditions:
1. The emulator should complete within 7 seconds
2. Both LCD chips should have "Display ON: True"

Exit codes:
- 0: Success (both conditions pass)
- 1: Failure (timeout or LCD display off)
"""

import subprocess
import sys
import re
from pathlib import Path

def main():
    # Path to the emulator script
    script_path = Path(__file__).parent / "pce500" / "run_pce500.py"
    
    # Command to run
    cmd = [sys.executable, str(script_path), "--dump-pc", "0xfffff"]
    
    print(f"Running: {' '.join(cmd)}")
    print("Timeout: 7 seconds")
    
    try:
        # Run with 7 second timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7.0,
            env={"FORCE_BINJA_MOCK": "1"}  # Ensure we use mock Binary Ninja
        )
        
        # Process completed within timeout
        print(f"Process completed in under 7 seconds (exit code: {result.returncode})")
        
        # Check for Display ON lines
        display_on_pattern = re.compile(r"Display ON: (True|False)")
        matches = display_on_pattern.findall(result.stdout)
        
        if not matches:
            print("ERROR: No 'Display ON:' lines found in output")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            return 1
        
        print(f"Found {len(matches)} 'Display ON:' lines")
        
        # Check if any are False
        false_count = matches.count("False")
        true_count = matches.count("True")
        
        print(f"Display ON counts: True={true_count}, False={false_count}")
        
        if false_count > 0:
            print(f"FAIL: Found {false_count} 'Display ON: False' lines")
            # Print the relevant lines for debugging
            for line in result.stdout.splitlines():
                if "Display ON:" in line:
                    print(f"  {line.strip()}")
            return 1
        
        if true_count != 2:
            print(f"FAIL: Expected 2 'Display ON: True' lines, found {true_count}")
            return 1
        
        print("SUCCESS: Both LCD chips show 'Display ON: True'")
        return 0
        
    except subprocess.TimeoutExpired:
        print("FAIL: Process timed out after 7 seconds")
        print("(This is expected in the current state - emulator takes 800s+)")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())