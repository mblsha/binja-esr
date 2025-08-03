"""Tests for emulator state update logic."""

import unittest
import time
import threading
import sys
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set FORCE_BINJA_MOCK before importing app
os.environ['FORCE_BINJA_MOCK'] = '1'

import app
from app import (
    update_emulator_state, emulator_run_loop, 
    UPDATE_TIME_THRESHOLD, UPDATE_INSTRUCTION_THRESHOLD
)


class TestEmulatorStateUpdates(unittest.TestCase):
    """Test emulator state update mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset global state
        app.emulator_state = {
            "is_running": False,
            "last_update_time": 0,
            "last_update_instructions": 0,
            "screen_data": None,
            "registers": {},
            "flags": {},
            "instruction_count": 0
        }
        
        # Create mock emulator
        self.mock_emulator = Mock()
        self.mock_emulator.instruction_count = 0
        self.mock_emulator.get_cpu_state.return_value = {
            'pc': 0x100,
            'a': 0x00,
            'b': 0x00,
            'ba': 0x0000,
            'i': 0x0000,
            'x': 0x000000,
            'y': 0x000000,
            'u': 0x000000,
            's': 0x000000,
            'flags': {'z': 0, 'c': 0},
            'cycles': 0
        }
        
        # Mock LCD
        mock_image = Mock()
        mock_image.save = Mock()
        self.mock_emulator.lcd.get_combined_display.return_value = mock_image
        
        # Replace global emulator
        app.emulator = self.mock_emulator
    
    def tearDown(self):
        """Clean up after tests."""
        # Ensure emulator is stopped
        app.emulator_state["is_running"] = False
        if app.emulator_thread:
            app.emulator_thread.join(timeout=1)
            app.emulator_thread = None
    
    def test_update_emulator_state_basic(self):
        """Test basic state update functionality."""
        # Call update
        update_emulator_state()
        
        # Verify state was updated
        state = app.emulator_state
        self.assertEqual(state['registers']['pc'], 0x100)
        self.assertEqual(state['flags']['z'], 0)
        self.assertEqual(state['flags']['c'], 0)
        self.assertEqual(state['instruction_count'], 0)
        
        # Verify screen was captured
        self.assertTrue(state['screen'].startswith('data:image/png;base64,'))
        self.mock_emulator.lcd.get_combined_display.assert_called_once_with(zoom=1)
        
        # Verify timestamp was updated
        self.assertGreater(state['last_update_time'], 0)
        self.assertEqual(state['last_update_instructions'], 0)
    
    def test_update_emulator_state_no_emulator(self):
        """Test update when emulator is None."""
        app.emulator = None
        
        # Should not crash
        update_emulator_state()
        
        # State should remain unchanged
        state = app.emulator_state
        self.assertEqual(state['registers'], {})
    
    def test_time_based_update_threshold(self):
        """Test time-based update threshold."""
        # Set up initial state
        app.emulator_state["is_running"] = True
        app.emulator_state["last_update_time"] = time.time()
        
        # Mock time to control elapsed time
        with patch('app.time.time') as mock_time:
            # Initial time
            current_time = app.emulator_state["last_update_time"]
            mock_time.return_value = current_time
            
            # Run one iteration - should not update (not enough time elapsed)
            with patch('app.update_emulator_state') as mock_update:
                # Simulate run loop iteration
                app.emulator_state["is_running"] = True
                thread = threading.Thread(target=emulator_run_loop)
                thread.start()
                
                # Wait a bit
                time.sleep(0.05)
                
                # Stop thread
                app.emulator_state["is_running"] = False
                thread.join(timeout=1)
                
                # Should not have updated
                mock_update.assert_not_called()
            
            # Now advance time past threshold
            mock_time.return_value = current_time + UPDATE_TIME_THRESHOLD + 0.01
            
            # Run another iteration - should update
            with patch('app.update_emulator_state') as mock_update:
                app.emulator_state["is_running"] = True
                thread = threading.Thread(target=emulator_run_loop)
                thread.start()
                
                # Wait a bit
                time.sleep(0.05)
                
                # Stop thread
                app.emulator_state["is_running"] = False
                thread.join(timeout=1)
                
                # Should have updated
                mock_update.assert_called()
    
    def test_instruction_based_update_threshold(self):
        """Test instruction-based update threshold."""
        # Set up initial state
        app.emulator_state["is_running"] = True
        app.emulator_state["last_update_instructions"] = 0
        app.emulator_state["last_update_time"] = time.time()
        
        # Set instruction count just below threshold
        self.mock_emulator.instruction_count = UPDATE_INSTRUCTION_THRESHOLD - 1
        
        with patch('app.update_emulator_state') as mock_update:
            # Simulate run loop iteration
            app.emulator_state["is_running"] = True
            thread = threading.Thread(target=emulator_run_loop)
            thread.start()
            
            # Wait a bit
            time.sleep(0.05)
            
            # Stop thread
            app.emulator_state["is_running"] = False
            thread.join(timeout=1)
            
            # Should not have updated
            mock_update.assert_not_called()
        
        # Now set instruction count past threshold
        self.mock_emulator.instruction_count = UPDATE_INSTRUCTION_THRESHOLD + 1
        
        with patch('app.update_emulator_state') as mock_update:
            app.emulator_state["is_running"] = True
            thread = threading.Thread(target=emulator_run_loop)
            thread.start()
            
            # Wait a bit
            time.sleep(0.05)
            
            # Stop thread
            app.emulator_state["is_running"] = False
            thread.join(timeout=1)
            
            # Should have updated
            mock_update.assert_called()
    
    def test_emulator_run_loop_exception_handling(self):
        """Test run loop handles exceptions gracefully."""
        # Make step() raise an exception
        self.mock_emulator.step.side_effect = Exception("Test error")
        
        # Run the loop
        app.emulator_state["is_running"] = True
        thread = threading.Thread(target=emulator_run_loop)
        thread.start()
        
        # Wait for thread to finish
        thread.join(timeout=1)
        
        # Should have stopped running
        self.assertFalse(app.emulator_state["is_running"])
    
    def test_emulator_run_loop_stops_when_emulator_none(self):
        """Test run loop stops when emulator becomes None."""
        app.emulator = None
        
        # Run the loop
        app.emulator_state["is_running"] = True
        thread = threading.Thread(target=emulator_run_loop)
        thread.start()
        
        # Should exit quickly
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
    
    def test_state_update_with_different_cpu_values(self):
        """Test state update with various CPU register values."""
        test_cases = [
            {
                'pc': 0xFFFFFF, 'a': 0xFF, 'b': 0xFF, 'ba': 0xFFFF,
                'i': 0xFFFF, 'x': 0xFFFFFF, 'y': 0xFFFFFF,
                'u': 0xFFFFFF, 's': 0xFFFFFF,
                'flags': {'z': 1, 'c': 1}, 'cycles': 1000000
            },
            {
                'pc': 0x000000, 'a': 0x00, 'b': 0x00, 'ba': 0x0000,
                'i': 0x0000, 'x': 0x000000, 'y': 0x000000,
                'u': 0x000000, 's': 0x000000,
                'flags': {'z': 0, 'c': 0}, 'cycles': 0
            },
            {
                'pc': 0x123456, 'a': 0xAB, 'b': 0xCD, 'ba': 0xABCD,
                'i': 0x1234, 'x': 0x123456, 'y': 0x789ABC,
                'u': 0xDEF012, 's': 0x345678,
                'flags': {'z': 1, 'c': 0}, 'cycles': 12345
            }
        ]
        
        for test_state in test_cases:
            self.mock_emulator.get_cpu_state.return_value = test_state
            self.mock_emulator.instruction_count = test_state['cycles']
            
            update_emulator_state()
            
            state = app.emulator_state
            for reg in ['pc', 'a', 'b', 'ba', 'i', 'x', 'y', 'u', 's']:
                self.assertEqual(state['registers'][reg], test_state[reg])
            
            self.assertEqual(state['flags']['z'], test_state['flags']['z'])
            self.assertEqual(state['flags']['c'], test_state['flags']['c'])
            self.assertEqual(state['instruction_count'], test_state['cycles'])
    
    def test_concurrent_state_updates(self):
        """Test thread safety of state updates."""
        # This test verifies that concurrent access doesn't crash
        update_count = 0
        
        def update_worker():
            nonlocal update_count
            for _ in range(10):
                with app.emulator_lock:
                    update_emulator_state()
                    update_count += 1
                time.sleep(0.001)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have completed all updates
        self.assertEqual(update_count, 50)


if __name__ == '__main__':
    unittest.main()