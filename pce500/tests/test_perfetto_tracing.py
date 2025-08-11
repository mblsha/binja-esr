"""Tests for the Perfetto tracing system."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add path for retrobus-perfetto
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "retrobus-perfetto" / "py"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.tracing.perfetto_tracing import PerfettoTracer, tracer, RETROBUS_AVAILABLE


class TestPerfettoTracer(unittest.TestCase):
    """Test the Perfetto tracer implementation."""
    
    def setUp(self):
        """Set up test environment."""
        if not RETROBUS_AVAILABLE:
            self.skipTest("retrobus-perfetto not available")
        self.temp_dir = tempfile.mkdtemp()
        self.trace_path = os.path.join(self.temp_dir, "test.perfetto-trace")
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up any trace files
        if os.path.exists(self.trace_path):
            os.remove(self.trace_path)
        os.rmdir(self.temp_dir)
        
    def test_tracer_disabled_by_default(self):
        """Test that tracer is disabled by default."""
        t = PerfettoTracer()
        self.assertFalse(t.enabled)
        
    def test_start_stop_creates_file(self):
        """Test that starting and stopping creates a trace file."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        self.assertTrue(t.enabled)
        
        # Add some events
        t.instant("Test", "event1", {"value": 42})
        t.counter("Test", "counter1", 100)
        
        t.stop()
        self.assertFalse(t.enabled)
        
        # Check file was created
        self.assertTrue(os.path.exists(self.trace_path))
        
        # Check file size is reasonable (protobuf files should have some content)
        file_size = os.path.getsize(self.trace_path)
        self.assertGreater(file_size, 100)  # Should be at least 100 bytes
        
    def test_events_not_emitted_when_disabled(self):
        """Test that events are not emitted when tracer is disabled."""
        t = PerfettoTracer()
        
        # Try to emit without starting - should not crash
        t.instant("Test", "should_not_appear")
        t.counter("Test", "should_not_appear", 0)
        
        # Should not have created a file
        self.assertFalse(os.path.exists(self.trace_path))
        
    def test_instant_events_work(self):
        """Test that instant events can be created."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        # Create multiple instant events
        t.instant("TestTrack", "TestEvent1", {"key": "value", "number": 123})
        t.instant("TestTrack", "TestEvent2")
        t.instant("AnotherTrack", "TestEvent3", {"data": [1, 2, 3]})
        
        t.stop()
        
        # Verify file was created with content
        self.assertTrue(os.path.exists(self.trace_path))
        self.assertGreater(os.path.getsize(self.trace_path), 100)
        
    def test_counter_events_work(self):
        """Test that counter events can be created."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        # Create counter samples
        for i in range(10):
            t.counter("Metrics", "test_counter", float(i * 10))
            time.sleep(0.001)
        
        t.stop()
        
        # Verify file was created with content
        self.assertTrue(os.path.exists(self.trace_path))
        self.assertGreater(os.path.getsize(self.trace_path), 100)
        
    def test_slice_events_work(self):
        """Test begin/end slice events."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        # Create nested slices
        t.begin_slice("SliceTrack", "OuterSlice", {"level": 1})
        time.sleep(0.001)
        t.begin_slice("SliceTrack", "InnerSlice", {"level": 2})
        time.sleep(0.001)
        t.end_slice("SliceTrack")  # End inner
        time.sleep(0.001)
        t.end_slice("SliceTrack")  # End outer
        
        t.stop()
        
        # Verify file was created with content
        self.assertTrue(os.path.exists(self.trace_path))
        self.assertGreater(os.path.getsize(self.trace_path), 100)
        
    def test_multiple_tracks(self):
        """Test that multiple tracks can be created."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        # Use different tracks
        t.instant("Track1", "event1")
        t.instant("Track2", "event2")
        t.instant("Track3", "event3")
        t.counter("Counters", "metric1", 100)
        t.begin_slice("CPU", "function1")
        t.end_slice("CPU")
        
        t.stop()
        
        # Verify file was created with content
        self.assertTrue(os.path.exists(self.trace_path))
        self.assertGreater(os.path.getsize(self.trace_path), 100)
        
    def test_global_tracer_singleton(self):
        """Test that the global tracer is accessible."""
        self.assertIsNotNone(tracer)
        self.assertIsInstance(tracer, PerfettoTracer)
        
    def test_complex_trace_scenario(self):
        """Test a complex tracing scenario."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        # Simulate a realistic trace
        for i in range(5):
            t.begin_slice("CPU", f"function_{i}")
            t.instant("Execution", f"instruction_{i}", {"pc": i * 0x100})
            t.counter("metrics", "instructions", float(i * 1000))
            time.sleep(0.001)
            t.end_slice("CPU")
            
        t.stop()
        
        # Verify file was created with reasonable size
        self.assertTrue(os.path.exists(self.trace_path))
        self.assertGreater(os.path.getsize(self.trace_path), 500)


if __name__ == "__main__":
    unittest.main()