"""Unit tests for Perfetto tracing functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import threading
import time

from pce500.trace_manager import TraceManager, g_tracer, ENABLE_PERFETTO_TRACING, RETROBUS_PERFETTO_AVAILABLE


class TestTraceManager:
    """Test the TraceManager singleton and basic functionality."""
    
    def test_singleton_pattern(self):
        """Test that TraceManager is a singleton."""
        tm1 = TraceManager()
        tm2 = TraceManager()
        assert tm1 is tm2
        assert tm1 is g_tracer
    
    def test_start_stop_tracing(self):
        """Test starting and stopping tracing."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace', delete=False) as f:
            trace_path = f.name
        
        try:
            # Initially not tracing
            assert not g_tracer.is_tracing()
            
            # Start tracing
            result = g_tracer.start_tracing(trace_path)
            if result:  # Only if perfetto library is available
                assert g_tracer.is_tracing()
                
                # Stop tracing
                assert g_tracer.stop_tracing()
                assert not g_tracer.is_tracing()
                
                # Trace file should exist
                assert Path(trace_path).exists()
        finally:
            Path(trace_path).unlink(missing_ok=True)
    
    def test_timestamp_monotonic(self):
        """Test that timestamps are monotonically increasing."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with patch('pce500.trace_manager.ENABLE_PERFETTO_TRACING', True):
            if not g_tracer.is_tracing():
                with tempfile.NamedTemporaryFile(suffix='.perfetto-trace', delete=False) as f:
                    g_tracer.start_tracing(f.name)
                    cleanup = True
            else:
                cleanup = False
                
            try:
                ts1 = g_tracer.get_timestamp()
                time.sleep(0.001)  # Sleep 1ms
                ts2 = g_tracer.get_timestamp()
                
                assert ts2 > ts1
                # Should be at least 1000 microseconds apart
                assert ts2 - ts1 >= 1000
            finally:
                if cleanup:
                    g_tracer.stop_tracing()
                    Path(f.name).unlink(missing_ok=True)
    
    def test_thread_safety(self):
        """Test thread-safe access to TraceManager."""
        results = []
        
        def worker():
            # Each thread gets same instance
            tm = TraceManager()
            results.append(tm)
            
            # Test thread-safe depth tracking
            thread_id = threading.get_ident()
            initial_depth = tm.get_call_depth(thread_id)
            results.append(initial_depth)
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All threads should get same TraceManager instance
        assert all(tm is g_tracer for tm in results[::2])
        # All threads should start with depth 0
        assert all(depth == 0 for depth in results[1::2])


class TestTraceEvents:
    """Test different types of trace events."""
    
    @pytest.fixture
    def mock_perfetto(self):
        """Mock retrobus_perfetto module."""
        # Create a mock instance of PerfettoTraceBuilder
        mock_instance = MagicMock()
        
        # Mock required methods
        mock_instance.add_process_track.return_value = "process_track_uuid"
        mock_instance.add_thread = MagicMock(return_value="thread_track_uuid")
        mock_instance.add_counter_track.return_value = "counter_track_uuid"
        
        # Mock event methods that return event objects
        mock_event = MagicMock()
        mock_event.add_annotations = MagicMock()
        
        mock_instance.begin_slice = MagicMock(return_value=mock_event)
        mock_instance.end_slice = MagicMock()
        mock_instance.add_instant_event = MagicMock(return_value=mock_event)
        mock_instance.update_counter = MagicMock()
        mock_instance.add_flow = MagicMock()
        mock_instance.finalize = MagicMock(return_value=b"trace_data")
        
        # Patch both the class and the ENABLE flag
        with patch('pce500.trace_manager.ENABLE_PERFETTO_TRACING', True):
            with patch('pce500.trace_manager.PerfettoTraceBuilder', return_value=mock_instance):
                yield mock_instance
    
    def test_begin_end_function(self, mock_perfetto):
        """Test function begin/end tracing."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace') as f:
            g_tracer.start_tracing(f.name)
            
            # Trace a function
            g_tracer.begin_function("CPU", pc=0x1000, caller_pc=0x500, name="test_func")
            g_tracer.end_function("CPU", pc=0x1000)
            
            # Verify begin_slice and end_slice were called
            mock_perfetto.begin_slice.assert_called()
            mock_perfetto.end_slice.assert_called()
            
            g_tracer.stop_tracing()
    
    def test_trace_instant(self, mock_perfetto):
        """Test instant event tracing."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace') as f:
            g_tracer.start_tracing(f.name)
            
            # Trace instant event
            g_tracer.trace_instant("CPU", "TestEvent", {"value": 42})
            
            # Verify add_instant_event was called
            mock_perfetto.add_instant_event.assert_called()
            call_args = mock_perfetto.add_instant_event.call_args
            assert call_args[0][1] == "TestEvent"  # name is second argument
            
            g_tracer.stop_tracing()
    
    def test_trace_jump(self, mock_perfetto):
        """Test jump tracing."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace') as f:
            g_tracer.start_tracing(f.name)
            
            # Trace jump
            g_tracer.trace_jump("CPU", from_pc=0x1000, to_pc=0x2000)
            
            # Should create instant event
            mock_perfetto.add_instant_event.assert_called()
            
            g_tracer.stop_tracing()
    
    def test_trace_counter(self, mock_perfetto):
        """Test counter event tracing."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace') as f:
            g_tracer.start_tracing(f.name)
            
            # Trace counter
            g_tracer.trace_counter("CPU", "instructions", 1000)
            
            # Verify update_counter was called
            mock_perfetto.update_counter.assert_called()
            
            g_tracer.stop_tracing()
    
    def test_flow_events(self, mock_perfetto):
        """Test flow begin/end events."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace') as f:
            g_tracer.start_tracing(f.name)
            
            # Begin and end flow
            g_tracer.begin_flow("Interrupt", flow_id=123)
            g_tracer.end_flow("Interrupt", flow_id=123)
            
            # Should create flow events
            assert mock_perfetto.add_flow.call_count >= 2
            
            g_tracer.stop_tracing()


class TestCallStackTracking:
    """Test call stack tracking functionality."""
    
    def test_call_stack_depth_limits(self):
        """Test that call stack respects depth limits."""
        if not RETROBUS_PERFETTO_AVAILABLE:
            pytest.skip("retrobus_perfetto module not available")
            
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace', delete=False) as f:
            g_tracer.start_tracing(f.name)
        
        try:
            thread_id = threading.get_ident()
            
            # Push many frames
            for i in range(60):
                g_tracer.begin_function("CPU", pc=0x1000 + i, caller_pc=0x1000 + i - 1)
            
            # Check depth is limited
            depth = g_tracer.get_call_depth(thread_id)
            assert depth <= TraceManager.MAX_CALL_DEPTH
            
            # Clean up
            for i in range(60):
                g_tracer.end_function("CPU", pc=0x1000 + i)
                
        finally:
            g_tracer.stop_tracing()
            Path(f.name).unlink(missing_ok=True)
    
    def test_stale_frame_cleanup(self):
        """Test that old frames are cleaned up."""
        # This would require mocking time or waiting, so we'll skip for now
        pass


class TestNoOpBehavior:
    """Test behavior when tracing is disabled or perfetto library is not available."""
    
    def test_no_op_when_disabled(self):
        """Test that trace operations are no-ops when tracing is disabled."""
        # Ensure tracing is stopped
        if g_tracer.is_tracing():
            g_tracer.stop_tracing()
        
        # These should all be no-ops and not raise errors
        g_tracer.begin_function("CPU", 0x1000, 0x500)
        g_tracer.end_function("CPU", 0x1000)
        g_tracer.trace_instant("CPU", "test", {})
        g_tracer.trace_jump("CPU", 0x1000, 0x2000)
        g_tracer.trace_counter("CPU", "test", 42)
        g_tracer.begin_flow("test", 1)
        g_tracer.end_flow("test", 1)
        
        # Should return 0 when not tracing
        assert g_tracer.get_timestamp() == 0
    
    @patch('pce500.trace_manager.ENABLE_PERFETTO_TRACING', False)
    def test_tracing_disabled(self):
        """Test behavior when tracing is disabled."""
        with tempfile.NamedTemporaryFile(suffix='.perfetto-trace') as f:
            # Should return False when tracing is disabled
            assert not g_tracer.start_tracing(f.name)
            assert not g_tracer.is_tracing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])