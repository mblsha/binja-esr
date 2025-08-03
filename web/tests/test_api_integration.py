"""Integration tests for PC-E500 web emulator API."""

import unittest
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, Mock

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set FORCE_BINJA_MOCK before importing app
os.environ['FORCE_BINJA_MOCK'] = '1'

import app as app_module
from app import app, initialize_emulator


class TestAPIIntegration(unittest.TestCase):
    """Test Flask API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        # Initialize emulator for tests
        try:
            initialize_emulator()
        except Exception as e:
            print(f"Warning: Could not initialize emulator: {e}")
    
    def test_get_state(self):
        """Test GET /api/v1/state endpoint."""
        response = self.client.get('/api/v1/state')
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/json')
        
        # Parse JSON
        data = json.loads(response.data)
        
        # Verify state structure
        self.assertIn('screen', data)
        self.assertIn('registers', data)
        self.assertIn('flags', data)
        self.assertIn('instruction_count', data)
        self.assertIn('is_running', data)
        
        # Verify registers
        registers = data['registers']
        expected_registers = ['pc', 'a', 'b', 'ba', 'i', 'x', 'y', 'u', 's']
        for reg in expected_registers:
            self.assertIn(reg, registers)
            self.assertIsInstance(registers[reg], int)
        
        # Verify flags
        flags = data['flags']
        self.assertIn('z', flags)
        self.assertIn('c', flags)
        self.assertIsInstance(flags['z'], int)
        self.assertIsInstance(flags['c'], int)
        
        # Verify screen is base64 PNG
        self.assertTrue(data['screen'].startswith('data:image/png;base64,'))
    
    def test_control_reset(self):
        """Test POST /api/v1/control with reset command."""
        response = self.client.post('/api/v1/control',
                                   json={'command': 'reset'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'reset')
    
    def test_control_step(self):
        """Test POST /api/v1/control with step command."""
        # Ensure emulator is paused
        self.client.post('/api/v1/control', json={'command': 'pause'})
        
        # Get initial state
        initial_state = json.loads(self.client.get('/api/v1/state').data)
        initial_instructions = initial_state['instruction_count']
        
        # Step
        response = self.client.post('/api/v1/control',
                                   json={'command': 'step'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'stepped')
        
        # Verify instruction count increased
        new_state = json.loads(self.client.get('/api/v1/state').data)
        self.assertEqual(new_state['instruction_count'], initial_instructions + 1)
    
    def test_control_run_pause(self):
        """Test POST /api/v1/control with run and pause commands."""
        # Start running
        response = self.client.post('/api/v1/control',
                                   json={'command': 'run'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'running')
        
        # Verify is_running state
        state = json.loads(self.client.get('/api/v1/state').data)
        self.assertTrue(state['is_running'])
        
        # Pause
        response = self.client.post('/api/v1/control',
                                   json={'command': 'pause'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'paused')
        
        # Verify is_running state
        state = json.loads(self.client.get('/api/v1/state').data)
        self.assertFalse(state['is_running'])
    
    def test_control_invalid_command(self):
        """Test POST /api/v1/control with invalid command."""
        response = self.client.post('/api/v1/control',
                                   json={'command': 'invalid'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Unknown command', data['error'])
    
    def test_control_missing_command(self):
        """Test POST /api/v1/control without command."""
        response = self.client.post('/api/v1/control',
                                   json={},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Missing command', data['error'])
    
    def test_key_press(self):
        """Test POST /api/v1/key with press action."""
        response = self.client.post('/api/v1/key',
                                   json={'key_code': 'KEY_A', 'action': 'press'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        
        # Should return debug info
        self.assertIn('debug', data)
        debug = data['debug']
        self.assertIn('pressed_keys', debug)
        self.assertIn('KEY_A', debug['pressed_keys'])
    
    def test_key_release(self):
        """Test POST /api/v1/key with release action."""
        # First press a key
        self.client.post('/api/v1/key',
                        json={'key_code': 'KEY_B', 'action': 'press'})
        
        # Then release it
        response = self.client.post('/api/v1/key',
                                   json={'key_code': 'KEY_B', 'action': 'release'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        
        # Verify key is no longer pressed
        debug = data['debug']
        self.assertNotIn('KEY_B', debug['pressed_keys'])
    
    def test_key_default_action(self):
        """Test POST /api/v1/key defaults to press action."""
        response = self.client.post('/api/v1/key',
                                   json={'key_code': 'KEY_C'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        
        # Should have pressed the key
        debug = data['debug']
        self.assertIn('KEY_C', debug['pressed_keys'])
    
    def test_key_invalid_action(self):
        """Test POST /api/v1/key with invalid action."""
        response = self.client.post('/api/v1/key',
                                   json={'key_code': 'KEY_D', 'action': 'invalid'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Invalid action', data['error'])
    
    def test_key_missing_key_code(self):
        """Test POST /api/v1/key without key_code."""
        response = self.client.post('/api/v1/key',
                                   json={'action': 'press'},
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Missing key_code', data['error'])
    
    def test_index_page(self):
        """Test GET / returns HTML page."""
        response = self.client.get('/')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)
        self.assertIn(b'PC-E500 Web Emulator', response.data)
        self.assertIn(b'virtual-keyboard', response.data)
    
    def test_static_files(self):
        """Test static file serving."""
        # Test CSS
        response = self.client.get('/static/style.css')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/css', response.content_type)
        
        # Test JavaScript
        response = self.client.get('/static/app.js')
        self.assertEqual(response.status_code, 200)
        self.assertIn('javascript', response.content_type)
    
    def test_emulator_not_initialized(self):
        """Test API behavior when emulator is not initialized."""
        # Save original values
        original_emulator = app_module.emulator
        original_keyboard = app_module.keyboard_handler
        
        try:
            # Make emulator and keyboard handler None
            app_module.emulator = None
            app_module.keyboard_handler = None
            
            # Key endpoint should return error
            response = self.client.post('/api/v1/key',
                                       json={'key_code': 'KEY_E'},
                                       content_type='application/json')
            
            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertIn('error', data)
            self.assertIn('not initialized', data['error'])
            
            # Control endpoint (except reset) should return error
            response = self.client.post('/api/v1/control',
                                       json={'command': 'step'},
                                       content_type='application/json')
            
            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertIn('error', data)
            
        finally:
            # Restore original values
            app_module.emulator = original_emulator
            app_module.keyboard_handler = original_keyboard


if __name__ == '__main__':
    unittest.main()