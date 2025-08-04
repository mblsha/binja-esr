"""Unit tests for PC-E500 keyboard simulation with debouncing."""

import unittest
import sys
import time
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.keyboard import PCE500KeyboardHandler, DEFAULT_DEBOUNCE_READS


class TestKeyboardSimulation(unittest.TestCase):
    """Test keyboard simulation with debouncing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = PCE500KeyboardHandler()
    
    def test_key_queue_basic(self):
        """Test basic key queue operations."""
        # Initially queue should be empty
        self.assertEqual(len(self.handler.key_queue), 0)
        
        # Press a key
        self.handler.press_key('KEY_A')
        self.assertEqual(len(self.handler.key_queue), 1)
        
        # Check queued key properties
        queued_key = self.handler.key_queue[0]
        self.assertEqual(queued_key.key_code, 'KEY_A')
        self.assertEqual(queued_key.read_count, 0)
        self.assertEqual(queued_key.target_reads, DEFAULT_DEBOUNCE_READS)
        
        # Release the key - it should remain in queue until debounce completes
        self.handler.release_key('KEY_A')
        self.assertEqual(len(self.handler.key_queue), 1)
    
    def test_key_debouncing(self):
        """Test key debouncing behavior."""
        # Press KEY_Q (column 0, row 1)
        self.handler.press_key('KEY_Q')
        
        # Set KOL to select column 0 (bit 0)
        self.handler.handle_register_write(0xF0, 0x01)  # KOL bit 0
        
        # Read KIL multiple times
        for i in range(DEFAULT_DEBOUNCE_READS):
            kil = self.handler.handle_register_read(0xF2)
            # Should return the key's KIL value (bit 1 set - active high)
            self.assertEqual(kil, 0x02)  # (1 << 1)
            
            # Check read count
            self.assertEqual(self.handler.key_queue[0].read_count, i + 1)
        
        # Release the key
        self.handler.release_key('KEY_Q')
        
        # Key is marked as released but still in queue
        self.assertEqual(len(self.handler.key_queue), 1)
        
        # Next read will still show the key (it's complete but needs one more read to be removed)
        kil = self.handler.handle_register_read(0xF2)
        self.assertEqual(kil, 0x02)  # Key still active until removed from queue
        
        # Now key should be removed from queue
        self.assertEqual(len(self.handler.key_queue), 0)
        
        # And next read should show no keys
        kil = self.handler.handle_register_read(0xF2)
        self.assertEqual(kil, 0x00)  # No keys pressed (active high)
    
    def test_multiple_keys_queue(self):
        """Test multiple keys in queue."""
        # Press multiple keys
        self.handler.press_key('KEY_A')  # Column 0, row 3
        self.handler.press_key('KEY_B')  # Column 2, row 5
        self.handler.press_key('KEY_C')  # Column 1, row 5
        
        self.assertEqual(len(self.handler.key_queue), 3)
        
        # Select column 0 (for KEY_A)
        self.handler.handle_register_write(0xF0, 0x01)  # KOL bit 0
        
        # Read should only process KEY_A
        kil = self.handler.handle_register_read(0xF2)
        self.assertEqual(kil, 0x08)  # (1 << 3) - active high
        
        # Only KEY_A should have incremented read count
        self.assertEqual(self.handler.key_queue[0].read_count, 1)  # KEY_A
        self.assertEqual(self.handler.key_queue[1].read_count, 0)  # KEY_B
        self.assertEqual(self.handler.key_queue[2].read_count, 0)  # KEY_C
    
    def test_key_already_queued(self):
        """Test that pressing same key twice doesn't create duplicate."""
        self.handler.press_key('KEY_A')
        self.assertEqual(len(self.handler.key_queue), 1)
        
        # Press same key again
        self.handler.press_key('KEY_A')
        self.assertEqual(len(self.handler.key_queue), 1)  # Still only one
    
    def test_custom_debounce_reads(self):
        """Test custom number of debounce reads."""
        # Press key with custom target reads
        self.handler.press_key('KEY_A', target_reads=5)
        
        queued_key = self.handler.key_queue[0]
        self.assertEqual(queued_key.target_reads, 5)
        
        # Set appropriate KOL for column 0
        self.handler.handle_register_write(0xF0, 0x01)  # KOL bit 0 for column 0
        
        # Read 5 times
        for i in range(5):
            kil = self.handler.handle_register_read(0xF2)
            self.assertEqual(kil, 0x08)  # Key still active (row 3 bit set)
        
        # Release the key
        self.handler.release_key('KEY_A')
        
        # Key is marked as released but still in queue
        self.assertEqual(len(self.handler.key_queue), 1)
        
        # Next read will still show the key (it's complete but needs one more read to be removed)
        kil = self.handler.handle_register_read(0xF2)
        self.assertEqual(kil, 0x08)  # Key still active until removed
        
        # Now key should be removed from queue
        self.assertEqual(len(self.handler.key_queue), 0)
        
        # And next read should show no keys
        kil = self.handler.handle_register_read(0xF2)
        self.assertEqual(kil, 0x00)  # No keys
    
    def test_stuck_key_detection(self):
        """Test stuck key detection in get_queue_info."""
        # Press a key
        self.handler.press_key('KEY_A')
        
        # Get initial queue info
        queue_info = self.handler.get_queue_info()
        self.assertEqual(len(queue_info), 1)
        self.assertFalse(queue_info[0]['is_stuck'])
        
        # Simulate time passing without reads
        # Manually adjust queued_time to simulate stuck key
        self.handler.key_queue[0].queued_time = time.time() - 2.0  # 2 seconds ago
        
        # Get queue info again
        queue_info = self.handler.get_queue_info()
        self.assertTrue(queue_info[0]['is_stuck'])
    
    def test_queue_info_format(self):
        """Test get_queue_info return format."""
        # Press a key
        self.handler.press_key('KEY_Q')
        
        # Get queue info
        queue_info = self.handler.get_queue_info()
        self.assertEqual(len(queue_info), 1)
        
        info = queue_info[0]
        self.assertEqual(info['key_code'], 'KEY_Q')
        self.assertEqual(info['column'], 0)
        self.assertEqual(info['row'], 1)
        self.assertEqual(info['kol'], '0x01')  # Column 0 requires KOL bit 0
        self.assertEqual(info['koh'], '0x00')
        self.assertEqual(info['kil'], '0x02')  # Row 1 sets bit 1
        self.assertEqual(info['read_count'], 0)
        self.assertEqual(info['target_reads'], DEFAULT_DEBOUNCE_READS)
        self.assertEqual(info['progress'], f'0/{DEFAULT_DEBOUNCE_READS}')
        self.assertFalse(info['is_stuck'])
        self.assertIn('queued_time', info)
        self.assertIn('age_seconds', info)
    
    def test_kol_koh_combination(self):
        """Test keys requiring both KOL and KOH."""
        # Most keys only need KOL or KOH, but test the logic anyway
        # Press KEY_0 (column 5, row 7)
        self.handler.press_key('KEY_0')
        
        # Set KOL to select column 5
        self.handler.handle_register_write(0xF0, 0x20)  # KOL bit 5
        
        # Read KIL
        kil = self.handler.handle_register_read(0xF2)
        self.assertEqual(kil, 0x80)  # (1 << 7) - active high
        
        # Clear KOL, key should not be read
        self.handler.handle_register_write(0xF0, 0x00)
        kil = self.handler.handle_register_read(0xF2)
        self.assertEqual(kil, 0x00)  # No keys
    
    def test_release_all_keys(self):
        """Test releasing all keys clears queue."""
        # Press multiple keys
        self.handler.press_key('KEY_A')
        self.handler.press_key('KEY_B')
        self.handler.press_key('KEY_C')
        self.assertEqual(len(self.handler.key_queue), 3)
        
        # Release all
        self.handler.release_all_keys()
        self.assertEqual(len(self.handler.key_queue), 0)
    
    def test_get_debug_info_includes_queue(self):
        """Test that get_debug_info includes key queue info."""
        # Press a key
        self.handler.press_key('KEY_A')
        
        debug_info = self.handler.get_debug_info()
        self.assertIn('key_queue', debug_info)
        self.assertEqual(len(debug_info['key_queue']), 1)


if __name__ == '__main__':
    unittest.main()