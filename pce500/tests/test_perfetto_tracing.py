"""Tests for the Perfetto tracing system."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pce500.tracing.perfetto_tracing import PerfettoTracer, tracer


class TestPerfettoTracer(unittest.TestCase):
    """Test the Perfetto tracer implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.trace_path = os.path.join(self.temp_dir, "test.trace.json")
        
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
        
        # Verify JSON structure
        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            
        self.assertIn("traceEvents", data)
        self.assertIn("displayTimeUnit", data)
        self.assertEqual(data["displayTimeUnit"], "us")
        
        # Check events are present
        events = data["traceEvents"]
        self.assertGreater(len(events), 0)
        
        # Find our test events
        event_names = [e.get("name") for e in events]
        self.assertIn("event1", event_names)
        self.assertIn("counter1", event_names)
        
    def test_events_not_emitted_when_disabled(self):
        """Test that events are not emitted when tracer is disabled."""
        t = PerfettoTracer()
        
        # Try to emit without starting
        t.instant("Test", "should_not_appear")
        t.counter("Test", "should_not_appear", 0)
        
        # Start, stop immediately, then try to emit
        t.start(self.trace_path)
        t.stop()
        
        t.instant("Test", "should_not_appear_either")
        
        # Check file contents
        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            
        # Should only have metadata events
        events = data["traceEvents"]
        test_events = [e for e in events if "should_not_appear" in e.get("name", "")]
        self.assertEqual(len(test_events), 0)
        
    def test_instant_event_format(self):
        """Test instant event format."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        t.instant("TestTrack", "TestEvent", {"key": "value", "number": 123})
        
        t.stop()
        
        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            
        # Find the instant event
        instant_event = None
        for event in data["traceEvents"]:
            if event.get("name") == "TestEvent" and event.get("ph") == "i":
                instant_event = event
                break
                
        self.assertIsNotNone(instant_event)
        self.assertEqual(instant_event["ph"], "i")
        self.assertEqual(instant_event["s"], "t")
        self.assertIn("ts", instant_event)
        self.assertIn("pid", instant_event)
        self.assertIn("tid", instant_event)
        self.assertEqual(instant_event["args"]["key"], "value")
        self.assertEqual(instant_event["args"]["number"], 123)
        
    def test_counter_event_format(self):
        """Test counter event format."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        t.counter("CounterTrack", "test_counter", 42.5, {"extra": "data"})
        
        t.stop()
        
        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            
        # Find the counter event
        counter_event = None
        for event in data["traceEvents"]:
            if event.get("name") == "test_counter" and event.get("ph") == "C":
                counter_event = event
                break
                
        self.assertIsNotNone(counter_event)
        self.assertEqual(counter_event["ph"], "C")
        self.assertIn("ts", counter_event)
        self.assertEqual(counter_event["args"]["test_counter"], 42.5)
        self.assertEqual(counter_event["args"]["extra"], "data")
        
    def test_slice_events(self):
        """Test begin/end slice events."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        t.begin_slice("SliceTrack", "TestSlice", {"param": 1})
        time.sleep(0.001)  # Small delay to ensure different timestamps
        t.end_slice("SliceTrack")
        
        t.stop()
        
        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            
        # Find begin and end events
        begin_event = None
        end_event = None
        for event in data["traceEvents"]:
            if event.get("name") == "TestSlice" and event.get("ph") == "B":
                begin_event = event
            elif event.get("name") == "TestSlice" and event.get("ph") == "E":
                end_event = event
                
        self.assertIsNotNone(begin_event)
        self.assertIsNotNone(end_event)
        self.assertEqual(begin_event["args"]["param"], 1)
        self.assertLess(begin_event["ts"], end_event["ts"])
        
    def test_multiple_tracks(self):
        """Test that multiple tracks get different TIDs."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        t.instant("Track1", "event1")
        t.instant("Track2", "event2")
        t.instant("Track1", "event3")
        
        t.stop()
        
        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            
        # Find events and their TIDs
        track1_tids = set()
        track2_tids = set()
        
        for event in data["traceEvents"]:
            if event.get("name") in ("event1", "event3"):
                track1_tids.add(event.get("tid"))
            elif event.get("name") == "event2":
                track2_tids.add(event.get("tid"))
                
        # Each track should have a consistent TID
        self.assertEqual(len(track1_tids), 1)
        self.assertEqual(len(track2_tids), 1)
        
        # Different tracks should have different TIDs
        self.assertNotEqual(track1_tids.pop(), track2_tids.pop())
        
    def test_global_tracer_singleton(self):
        """Test that the global tracer is accessible."""
        self.assertIsNotNone(tracer)
        self.assertIsInstance(tracer, PerfettoTracer)
        
    def test_timestamps_are_monotonic(self):
        """Test that timestamps increase monotonically."""
        t = PerfettoTracer()
        t.start(self.trace_path)
        
        # Emit several events
        for i in range(10):
            t.instant("Test", f"event{i}")
            time.sleep(0.0001)  # Small delay
            
        t.stop()
        
        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            
        # Get all timestamps for our events
        timestamps = []
        for event in data["traceEvents"]:
            if event.get("name", "").startswith("event"):
                timestamps.append(event["ts"])
                
        # Check monotonic increase
        for i in range(1, len(timestamps)):
            self.assertGreaterEqual(timestamps[i], timestamps[i-1])


if __name__ == "__main__":
    unittest.main()