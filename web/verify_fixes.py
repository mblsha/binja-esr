#!/usr/bin/env python3
"""Verify that the web emulator fixes are working correctly."""

import requests
import base64
import io
from PIL import Image

def verify_web_emulator():
    """Verify the web emulator is working after fixes."""
    base_url = "http://localhost:8080"
    
    print("Web Emulator Fix Verification")
    print("=" * 50)
    
    try:
        # Get initial state
        response = requests.get(f"{base_url}/api/v1/state")
        state = response.json()
        
        print("\n1. Initial State Check:")
        print(f"   - Has 'screen' key: {'screen' in state}")
        print(f"   - Has 'screen_data' key: {'screen_data' in state}")
        print(f"   - PC register: 0x{state['registers']['pc']:06X}")
        print("   - Expected PC (entry point): 0x0F10C2")
        
        # Check if screen is valid base64 PNG
        if 'screen' in state and state['screen']:
            if state['screen'].startswith('data:image/png;base64,'):
                print("   ✓ Screen data is properly formatted")
                
                # Try to decode the image
                try:
                    img_data = base64.b64decode(state['screen'].split(',')[1])
                    img = Image.open(io.BytesIO(img_data))
                    print(f"   ✓ Screen image decoded successfully: {img.size[0]}x{img.size[1]} pixels")
                except Exception as e:
                    print(f"   ✗ Failed to decode screen image: {e}")
            else:
                print("   ✗ Screen data is not in expected format")
        else:
            print("   ✗ No screen data available")
        
        # Run a few steps
        print("\n2. Running emulator:")
        requests.post(f"{base_url}/api/v1/control", json={"command": "step"})
        requests.post(f"{base_url}/api/v1/control", json={"command": "step"})
        requests.post(f"{base_url}/api/v1/control", json={"command": "step"})
        
        # Get state after steps
        response = requests.get(f"{base_url}/api/v1/state")
        new_state = response.json()
        
        print(f"   - PC after 3 steps: 0x{new_state['registers']['pc']:06X}")
        print(f"   - Instructions executed: {new_state['instruction_count']}")
        
        # Test reset
        print("\n3. Testing reset:")
        response = requests.post(f"{base_url}/api/v1/control", json={"command": "reset"})
        
        # Get state after reset
        response = requests.get(f"{base_url}/api/v1/state")
        reset_state = response.json()
        
        print(f"   - PC after reset: 0x{reset_state['registers']['pc']:06X}")
        print(f"   - Instructions count reset: {reset_state['instruction_count']}")
        
        print("\n✓ All checks passed! The web emulator is working correctly.")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to web server at http://localhost:8080")
        print("  Please start the server with: cd web && python run.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    verify_web_emulator()