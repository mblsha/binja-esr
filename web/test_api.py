#!/usr/bin/env python3
"""Test script for PC-E500 web emulator API."""

import requests
import time

BASE_URL = "http://localhost:8080/api/v1"


def test_get_state():
    """Test getting emulator state."""
    print("Testing GET /api/v1/state...")
    response = requests.get(f"{BASE_URL}/state")
    if response.status_code == 200:
        state = response.json()
        print("✓ State retrieved successfully")
        print(f"  PC: 0x{state['registers']['pc']:06X}")
        print(f"  Instructions: {state['instruction_count']}")
        print(f"  Running: {state['is_running']}")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        return False


def test_control(command):
    """Test emulator control."""
    print(f"\nTesting POST /api/v1/control (command={command})...")
    response = requests.post(f"{BASE_URL}/control", json={"command": command})
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Command '{command}' executed: {result['status']}")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        return False


def test_key_press(key_code):
    """Test key press/release."""
    print(f"\nTesting POST /api/v1/key (key={key_code})...")

    # Press key
    response = requests.post(
        f"{BASE_URL}/key", json={"key_code": key_code, "action": "press"}
    )
    if response.status_code == 200:
        result = response.json()
        print("✓ Key press sent")
        if "debug" in result:
            print(f"  Debug: {result['debug']}")
    else:
        print(f"✗ Press failed: {response.status_code}")
        return False

    time.sleep(0.1)

    # Release key
    response = requests.post(
        f"{BASE_URL}/key", json={"key_code": key_code, "action": "release"}
    )
    if response.status_code == 200:
        print("✓ Key release sent")
        return True
    else:
        print(f"✗ Release failed: {response.status_code}")
        return False


def main():
    """Run API tests."""
    print("PC-E500 Web Emulator API Test\n")

    # Test getting initial state
    test_get_state()

    # Test reset
    test_control("reset")

    # Test step
    test_control("step")
    test_get_state()

    # Test key press
    test_key_press("KEY_A")

    # Test running
    test_control("run")
    time.sleep(0.5)
    test_get_state()

    # Test pause
    test_control("pause")
    test_get_state()

    print("\n✓ All tests completed")


if __name__ == "__main__":
    main()
