"""
End-to-end test for heartbeat retry behavior.

Uses a real HTTP server (no mocks) to verify that:
1. The monitor thread doesn't block when trigger returns 502
2. finish() completes promptly even when the trigger endpoint is down
3. The heartbeat uses max_retries=0 and short timeout (fire-and-forget)

This test would have caught the production incident where 502s on the
trigger endpoint caused _worker_monitor to block for 7-135 seconds per
cycle (4 retries × 32s timeout), making finish() hang.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

from pluto.api import make_compat_trigger_v1
from pluto.iface import ServerInterface
from pluto.op import OpMonitor
from pluto.sets import Settings


def _make_502_server():
    """Start a local HTTP server that returns 502 on trigger, 200 elsewhere."""
    trigger_requests = {'count': 0, 'timestamps': []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if '/api/runs/trigger' in self.path:
                trigger_requests['count'] += 1
                trigger_requests['timestamps'].append(time.time())
                self.send_response(502)
                self.end_headers()
                self.wfile.write(b'<html>502 Bad Gateway</html>')
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({}).encode())

        def log_message(self, format, *args):
            pass  # Silence request logs during tests

    server = HTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, trigger_requests


def _make_settings(port):
    """Create Settings pointing at local test server."""
    settings = Settings()
    settings._op_id = 1
    settings._op_name = 'test-heartbeat'
    settings.project = 'test'
    settings._auth = 'test-token'
    settings._op_status = -1

    base = f'http://127.0.0.1:{port}'
    settings.url_app = base
    settings.url_api = base
    settings.url_ingest = base
    settings.url_py = base
    settings.update_url()

    # Fast heartbeats for testing (1s instead of default 4s)
    settings.x_sys_sampling_interval = 1
    # Keep default retry settings so we test the override, not global config
    # x_file_stream_retry_max = 4 (default)
    # x_file_stream_timeout_seconds = 32 (default)

    # Provide a minimal _sys mock that returns real dicts
    mock_sys = MagicMock()
    mock_sys.monitor.return_value = {'sys/cpu.utilization': 25.0}
    settings._sys = mock_sys

    return settings


class TestHeartbeatE2E:
    """End-to-end tests using a real HTTP server, no mocking of retry logic."""

    def test_monitor_stops_quickly_despite_502_trigger(self):
        """
        _worker_monitor should stop within seconds even when every
        heartbeat gets a 502. Before the fix, each heartbeat blocked
        for 7+ seconds of retries; stopping took 30+ seconds.
        """
        server, port, trigger_stats = _make_502_server()

        try:
            settings = _make_settings(port)
            iface = ServerInterface({}, settings)

            # Build a minimal op-like object with the real ServerInterface
            mock_op = MagicMock()
            mock_op.settings = settings
            mock_op._iface = iface
            mock_op._sync_manager = None

            monitor = OpMonitor(mock_op)
            monitor.start()

            # Let monitor run for 3s — should fire multiple heartbeats
            time.sleep(3)
            assert trigger_stats['count'] > 0, 'Monitor should have sent heartbeats'

            # Stop and measure — this is the critical assertion
            start = time.time()
            monitor.stop()
            elapsed = time.time() - start

            # With the fix: stops in <10s (1 heartbeat cycle + 5s timeout max)
            # Without fix: would take 30s (thread join timeout, stuck in retries)
            assert elapsed < 10, (
                f'Monitor.stop() took {elapsed:.1f}s, expected <10s. '
                f'Heartbeat retries are likely blocking the thread.'
            )

            iface.close()
        finally:
            server.shutdown()

    def test_heartbeats_are_not_retried_on_502(self):
        """
        Each heartbeat cycle should make exactly 1 HTTP request to the
        trigger endpoint, not 4 (the default retry count for data uploads).
        """
        server, port, trigger_stats = _make_502_server()

        try:
            settings = _make_settings(port)
            iface = ServerInterface({}, settings)

            mock_op = MagicMock()
            mock_op.settings = settings
            mock_op._iface = iface
            mock_op._sync_manager = None

            monitor = OpMonitor(mock_op)
            monitor.start()

            # Run for 4s with 1s heartbeat interval → expect ~3-4 heartbeats
            time.sleep(4)
            monitor.stop()

            # Without the fix: 4 retries per heartbeat → 12-16 requests for 3-4 cycles
            # With the fix: 1 request per heartbeat → 3-4 requests for 3-4 cycles
            # Allow some slack but should be well below the retry-heavy count
            assert trigger_stats['count'] < 10, (
                f'Got {trigger_stats["count"]} trigger requests in 4s '
                f'with 1s interval. Expected <10 (1 per cycle). '
                f'Heartbeats are being retried.'
            )

            iface.close()
        finally:
            server.shutdown()

    def test_heartbeat_spacing_is_not_inflated_by_retries(self):
        """
        Time between consecutive heartbeats should be ~1s (the sampling
        interval), not 7+ seconds (sampling interval + retry backoff).
        """
        server, port, trigger_stats = _make_502_server()

        try:
            settings = _make_settings(port)
            iface = ServerInterface({}, settings)

            mock_op = MagicMock()
            mock_op.settings = settings
            mock_op._iface = iface
            mock_op._sync_manager = None

            monitor = OpMonitor(mock_op)
            monitor.start()

            # Run long enough to get several heartbeats
            time.sleep(5)
            monitor.stop()

            timestamps = trigger_stats['timestamps']
            assert len(timestamps) >= 3, (
                f'Expected >=3 heartbeats in 5s, got {len(timestamps)}'
            )

            # Check spacing between consecutive heartbeats
            gaps = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps)

            # With the fix: avg gap ≈ 1s (sampling interval) + small overhead
            # Without fix: avg gap ≈ 8s+ (1s interval + 7s retry backoff)
            assert avg_gap < 4, (
                f'Average heartbeat gap is {avg_gap:.1f}s, expected <4s. '
                f'Retries are inflating the interval. Gaps: {gaps}'
            )

            iface.close()
        finally:
            server.shutdown()

    def test_data_uploads_still_retry_with_default_settings(self):
        """
        Verify that non-heartbeat calls (e.g. status update) still use
        the full retry behavior — the fix should only affect heartbeats.
        """
        server, port, trigger_stats = _make_502_server()

        try:
            settings = _make_settings(port)
            # Use very fast retry backoff for test speed
            settings.x_file_stream_retry_wait_min_seconds = 0.01
            settings.x_file_stream_retry_wait_max_seconds = 0.01
            settings.x_file_stream_retry_max = 3

            iface = ServerInterface({}, settings)

            # Call _post_v1 WITHOUT max_retries override (like update_status does)
            # This should make 1 initial attempt + 3 retries = 4 total requests
            mock_trigger = make_compat_trigger_v1(settings)
            result = iface._post_v1(
                settings.url_trigger,
                iface.headers,
                mock_trigger,
                client=iface.client_api,
                name='status-update',
                # No max_retries override — uses default settings
            )

            assert result is None  # All retries should fail (502)
            # x_file_stream_retry_max=3: 1 initial attempt + 3 retries = 4 total
            assert trigger_stats['count'] == 4

            iface.close()
        finally:
            server.shutdown()
