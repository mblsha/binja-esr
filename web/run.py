#!/usr/bin/env python3
"""Run the PC-E500 web emulator."""

from app import app, initialize_emulator

if __name__ == '__main__':
    # Initialize emulator on startup
    try:
        initialize_emulator()
        print("PC-E500 emulator initialized successfully")
    except Exception as e:
        print(f"Failed to initialize emulator: {e}")
    
    # Run Flask app
    print("Starting web server at http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)