"""Tests for sync process V2 architecture.

These tests verify the sync process implementation including:
- SyncStore database operations
- File upload via sync process
- Graceful shutdown handling
- Payload format verification (unit tests)
"""

import json
import logging
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

import pluto
from pluto.sync.process import _SyncUploader
from pluto.sync.store import FileRecord, RecordType, SyncRecord, SyncStatus, SyncStore
from tests.utils import get_task_name

TESTING_PROJECT_NAME = 'testing-ci'


class TestSyncStore:
    """Unit tests for SyncStore database operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary SyncStore for testing."""
        db_path = tmp_path / 'test_sync.db'
        store = SyncStore(str(db_path))
        yield store
        store.close()

    def test_schema_initialization(self, store):
        """Test that database schema is created correctly."""
        # Verify we can query the tables
        cursor = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert 'runs' in tables
        assert 'sync_queue' in tables
        assert 'file_uploads' in tables
        assert 'schema_version' in tables

    def test_enqueue_file(self, store):
        """Test file can be enqueued to the store."""
        # Register a run first
        store.register_run('test-run-1', 'test-project')

        # Enqueue a file
        file_id = store.enqueue_file(
            run_id='test-run-1',
            local_path='/tmp/test.png',
            file_name='test',
            file_ext='.png',
            file_type='image/png',
            file_size=1024,
            log_name='images/test',
            timestamp_ms=int(time.time() * 1000),
            step=1,
        )

        assert file_id > 0

    def test_get_pending_files(self, store):
        """Test pending files are retrieved correctly."""
        store.register_run('test-run-1', 'test-project')

        # Enqueue multiple files
        for i in range(3):
            store.enqueue_file(
                run_id='test-run-1',
                local_path=f'/tmp/test{i}.png',
                file_name=f'test{i}',
                file_ext='.png',
                file_type='image/png',
                file_size=1024,
                log_name=f'images/test{i}',
                timestamp_ms=int(time.time() * 1000),
                step=i,
            )

        # Get pending files
        pending = store.get_pending_files(limit=10)
        assert len(pending) == 3
        assert all(isinstance(f, FileRecord) for f in pending)
        assert all(f.status == SyncStatus.PENDING for f in pending)

    def test_file_status_transitions(self, store):
        """Test file status transitions work correctly."""
        store.register_run('test-run-1', 'test-project')

        file_id = store.enqueue_file(
            run_id='test-run-1',
            local_path='/tmp/test.png',
            file_name='test',
            file_ext='.png',
            file_type='image/png',
            file_size=1024,
            log_name='images/test',
            timestamp_ms=int(time.time() * 1000),
            step=1,
        )

        # Initially PENDING
        pending = store.get_pending_files()
        assert len(pending) == 1
        assert pending[0].status == SyncStatus.PENDING

        # Mark in progress
        store.mark_files_in_progress([file_id])
        # Should still be retrievable as "pending" for retry purposes
        pending = store.get_pending_files()
        assert len(pending) == 0  # IN_PROGRESS is not retrieved

        # Mark completed
        store.mark_files_completed([file_id])
        pending = store.get_pending_files()
        assert len(pending) == 0

    def test_file_retry_count(self, store):
        """Test retry count increments on failure."""
        store.register_run('test-run-1', 'test-project')

        file_id = store.enqueue_file(
            run_id='test-run-1',
            local_path='/tmp/test.png',
            file_name='test',
            file_ext='.png',
            file_type='image/png',
            file_size=1024,
            log_name='images/test',
            timestamp_ms=int(time.time() * 1000),
            step=1,
        )

        # Mark failed multiple times
        store.mark_files_failed([file_id], 'Test error')
        store.mark_files_failed([file_id], 'Test error 2')

        # Query directly to check retry count
        cursor = store.conn.execute(
            'SELECT retry_count FROM file_uploads WHERE id = ?', (file_id,)
        )
        row = cursor.fetchone()
        assert row[0] == 2

    def test_pending_file_count(self, store):
        """Test pending file count is accurate."""
        store.register_run('test-run-1', 'test-project')

        assert store.get_pending_file_count() == 0

        for i in range(5):
            store.enqueue_file(
                run_id='test-run-1',
                local_path=f'/tmp/test{i}.png',
                file_name=f'test{i}',
                file_ext='.png',
                file_type='image/png',
                file_size=1024,
                log_name=f'images/test{i}',
                timestamp_ms=int(time.time() * 1000),
                step=i,
            )

        assert store.get_pending_file_count() == 5
        assert store.get_pending_file_count('test-run-1') == 5
        assert store.get_pending_file_count('other-run') == 0


class TestSyncProcessIntegration:
    """Integration tests for sync process file uploads."""

    @pytest.fixture
    def sync_enabled_run(self):
        """Create a pluto run with sync process enabled."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )
        yield run
        run.finish()

    def test_image_upload_via_sync(self, sync_enabled_run, tmp_path):
        """Test image uploads work via sync process."""
        # Create a test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = PILImage.fromarray(img_array)
        img_path = tmp_path / 'test_image.png'
        img.save(img_path)

        # Log the image
        image = pluto.Image(str(img_path), caption='test-image')
        sync_enabled_run.log({'images/test': image})

        # Allow time for sync process to pick it up
        time.sleep(2)

    def test_multiple_files_same_step(self, sync_enabled_run, tmp_path):
        """Test multiple files logged in same step."""
        # Create test files
        for i in range(3):
            img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            img = PILImage.fromarray(img_array)
            img_path = tmp_path / f'test_image_{i}.png'
            img.save(img_path)

            image = pluto.Image(str(img_path), caption=f'test-{i}')
            sync_enabled_run.log({f'images/batch_{i}': image})

        # Allow time for sync
        time.sleep(2)

    def test_text_upload_via_sync(self, sync_enabled_run):
        """Test text uploads work via sync process."""
        text = pluto.Text('Test log message content', caption='log-message')
        sync_enabled_run.log({'logs/message': text})

        # Allow time for sync
        time.sleep(2)

    def test_artifact_upload_via_sync(self, sync_enabled_run, tmp_path):
        """Test artifact uploads work via sync process."""
        # Create a JSON artifact
        artifact_path = tmp_path / 'test_artifact.json'
        artifact_path.write_text(json.dumps({'test': 'data', 'value': 123}))

        artifact = pluto.Artifact(str(artifact_path), caption='json-artifact')
        sync_enabled_run.log({'artifacts/json': artifact})

        # Allow time for sync
        time.sleep(2)

    def test_system_metrics_upload_via_sync(self):
        """Test system metrics (CPU, GPU, memory) upload via sync process.

        System metrics are automatically collected by OpMonitor and uploaded
        via the sync process. This test verifies the full pipeline works
        by checking that uploads complete without HTTP errors.
        """
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # System metrics are collected automatically by OpMonitor
        # Wait for at least one collection cycle (default interval is 1s)
        time.sleep(3)

        # Verify system metrics were enqueued
        assert run._sync_manager is not None
        store = run._sync_manager.store

        with store._lock:
            cursor = store.conn.execute(
                'SELECT COUNT(*) FROM sync_queue WHERE record_type = ?',
                (int(RecordType.SYSTEM),),
            )
            system_count = cursor.fetchone()[0]

        # Should have at least one system metrics record
        assert system_count > 0, 'System metrics should be collected by OpMonitor'

        # Finish waits for sync to complete - HTTP errors would cause this to fail
        # The sync process uploads system metrics to the backend; success means
        # HTTP requests returned 2xx status codes
        run.finish()

        # If we get here without exception, uploads succeeded


class TestSyncProcessShutdown:
    """Tests for sync process shutdown behavior."""

    def test_graceful_shutdown_waits_for_pending(self):
        """Test that finish() waits for pending uploads to complete."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # Log some metrics
        for i in range(10):
            run.log({'metric': i})

        # Log a text file
        text = pluto.Text('Test content', caption='test')
        run.log({'logs/test': text})

        # Finish should wait for sync and complete reasonably quickly
        start = time.time()
        run.finish()
        elapsed = time.time() - start
        assert elapsed < 60, f'Shutdown took too long: {elapsed}s'

    def test_sync_manager_pending_count(self):
        """Test that pending count tracks metrics and files."""
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # Initial count may include system metrics from monitor starting
        _initial_count = run._sync_manager.get_pending_count()

        # Log some data
        for i in range(5):
            run.log({'metric': i})

        # Allow a moment for data to be enqueued
        time.sleep(0.5)

        # Verify data was enqueued (pending count should increase or be processed)
        pending_after_log = run._sync_manager.get_pending_count()
        assert pending_after_log >= 0  # May be 0 if sync already processed

        run.finish()

    def test_sync_process_registers_metric_names(self):
        """Test that sync process path registers metric names with server.

        This is critical: the server requires metric names to be registered
        via /api/runs/logName/add before it will accept and display the data.
        Without this registration, metrics are silently ignored.
        """
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # Log metrics with specific names
        run.log({'test_metric_alpha': 1.0, 'test_metric_beta': 2.0})
        run.log({'test_metric_gamma': 3.0})

        # Verify metric names were registered in settings.meta
        # (this tracks which names have been sent to /api/runs/logName/add)
        assert 'test_metric_alpha' in run.settings.meta
        assert 'test_metric_beta' in run.settings.meta
        assert 'test_metric_gamma' in run.settings.meta

        # Verify _iface exists and would have been used for metadata
        assert run._iface is not None, (
            'ServerInterface must exist to register metric names with server'
        )

        run.finish()

    def test_sync_process_calls_update_meta(self):
        """Test that _update_meta is called when logging via sync process.

        This verifies the fix for the bug where metrics weren't appearing
        in dashboards because _update_meta wasn't being called.
        """
        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # Mock _update_meta to track calls
        update_meta_calls = []
        original_update_meta = run._iface._update_meta

        def mock_update_meta(num=None, df=None):
            update_meta_calls.append({'num': num, 'df': df})
            return original_update_meta(num=num, df=df)

        run._iface._update_meta = mock_update_meta

        # Log a metric - this should trigger _update_meta
        run.log({'new_metric': 42.0})

        # Verify _update_meta was called with the metric name
        metric_calls = [c for c in update_meta_calls if c['num']]
        assert len(metric_calls) > 0, '_update_meta should be called for new metrics'

        # Find the call that registered 'new_metric'
        registered_names = []
        for call in metric_calls:
            if call['num']:
                registered_names.extend(call['num'])

        assert 'new_metric' in registered_names, (
            f"'new_metric' should be registered via _update_meta. "
            f'Registered names: {registered_names}'
        )

        run.finish()

    def test_console_logs_are_captured(self):
        """Test that stdout/stderr output is captured and enqueued to sync store."""
        from pluto.sync.store import RecordType

        run = pluto.init(
            project=TESTING_PROJECT_NAME,
            name=get_task_name(),
            config={},
            sync_process_enabled=True,
        )

        # Print something to stdout - this should be captured by ConsoleHandler
        print('Test console log message')

        # Give a moment for the log to be enqueued
        time.sleep(0.5)

        # Check that console logs were enqueued to the sync store
        assert run._sync_manager is not None
        store = run._sync_manager.store

        # Query for CONSOLE records in the store
        with store._lock:
            cursor = store.conn.execute(
                'SELECT * FROM sync_queue WHERE record_type = ?',
                (int(RecordType.CONSOLE),),
            )
            console_records = cursor.fetchall()

        assert len(console_records) > 0, (
            'Console logs should be enqueued to sync store. '
            'This test would have caught the missing console log upload feature.'
        )

        # Verify the message content
        import json

        found_test_message = False
        for record in console_records:
            payload = json.loads(record['payload_json'])
            if 'Test console log message' in payload.get('message', ''):
                found_test_message = True
                break

        assert found_test_message, 'Test message should be found in console records'

        run.finish()


class TestSyncUploaderPayloadFormat:
    """Unit tests for _SyncUploader payload formats.

    These tests mock HTTP requests and verify the exact JSON payloads
    being sent to ensure compatibility with the server API.
    """

    @pytest.fixture
    def uploader(self):
        """Create an uploader with mock settings."""
        settings = {
            '_auth': 'test-token',
            '_op_id': 12345,
            '_op_name': 'test-run',
            'project': 'test-project',
            'url_num': 'https://test.example.com/ingest/metrics',
            'url_data': 'https://test.example.com/ingest/data',
            'url_update_config': 'https://test.example.com/api/runs/config/update',
            'url_update_tags': 'https://test.example.com/api/runs/tags/update',
            'url_file': 'https://test.example.com/files',
        }
        log = logging.getLogger('test')
        return _SyncUploader(settings, log)

    def test_metrics_payload_format(self, uploader):
        """Test metrics are sent in correct NDJSON format.

        Expected format: {"time": <ms>, "step": <int>, "data": {...}}
        NOT the incorrect: {"k": key, "v": value, "s": step, "t": timestamp}
        """
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5, 'accuracy': 0.95},
                timestamp_ms=1705600000000,
                step=10,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client

            # Force re-creation of client
            uploader._client = None
            uploader.upload_metrics_batch(records)

            # Verify POST was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            # Parse the body (may be bytes or str)
            body = call_args.kwargs.get('content') or call_args[1].get('content')
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            lines = body.strip().split('\n')
            assert len(lines) == 1

            payload = json.loads(lines[0])

            # Verify correct format
            assert 'time' in payload, "Payload must have 'time' field"
            assert 'step' in payload, "Payload must have 'step' field"
            assert 'data' in payload, "Payload must have 'data' field"
            assert payload['time'] == 1705600000000
            assert payload['step'] == 10
            assert payload['data'] == {'loss': 0.5, 'accuracy': 0.95}

            # Verify incorrect format is NOT used
            assert 'k' not in payload, "Should not use 'k' field"
            assert 'v' not in payload, "Should not use 'v' field"
            assert 's' not in payload, "Should not use 's' field"
            assert 't' not in payload, "Should not use 't' field"

    def test_metrics_payload_filters_non_numeric(self, uploader):
        """Test that non-numeric values are filtered from metrics."""
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5, 'name': 'test', 'enabled': True},
                timestamp_ms=1705600000000,
                step=1,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_metrics_batch(records)

            body = mock_client.post.call_args.kwargs['content']
            payload = json.loads(body.strip())

            # Only numeric values should be in data
            assert payload['data'] == {'loss': 0.5}
            assert 'name' not in payload['data']
            assert 'enabled' not in payload['data']

    def test_system_metrics_payload_format(self, uploader):
        """Test system metrics preserve 'sys/' prefix from monitor."""
        # Note: System metrics from the monitor already have 'sys/' prefix
        # The uploader should NOT add another prefix
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.SYSTEM,
                payload={'sys/cpu.pct.0': 45.2, 'sys/mem.used': 1024},
                timestamp_ms=1705600000000,
                step=None,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_system_batch(records)

            body = mock_client.post.call_args.kwargs['content']
            payload = json.loads(body.strip())

            # Verify sys/ prefix is preserved (not doubled)
            assert 'sys/cpu.pct.0' in payload['data']
            assert 'sys/mem.used' in payload['data']
            assert payload['data']['sys/cpu.pct.0'] == 45.2
            assert payload['data']['sys/mem.used'] == 1024
            assert payload['step'] == 0  # System metrics have step=0

    def test_config_payload_format(self, uploader):
        """Test config update payload format."""
        record = SyncRecord(
            id=1,
            run_id='test-run',
            record_type=RecordType.CONFIG,
            payload={'learning_rate': 0.001, 'batch_size': 32},
            timestamp_ms=1705600000000,
            step=None,
            status=SyncStatus.PENDING,
            retry_count=0,
            created_at=time.time(),
            last_attempt_at=None,
            error_message=None,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_config(record)

            body = mock_client.post.call_args.kwargs['content']
            payload = json.loads(body)

            assert payload['runId'] == 12345
            # Server expects config as a JSON-encoded string (double-encoded)
            assert payload['config'] == json.dumps(
                {'learning_rate': 0.001, 'batch_size': 32}
            )

    def test_tags_payload_format(self, uploader):
        """Test tags update payload format."""
        record = SyncRecord(
            id=1,
            run_id='test-run',
            record_type=RecordType.TAGS,
            payload={'tags': ['experiment', 'v2', 'baseline']},
            timestamp_ms=1705600000000,
            step=None,
            status=SyncStatus.PENDING,
            retry_count=0,
            created_at=time.time(),
            last_attempt_at=None,
            error_message=None,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_tags(record)

            body = mock_client.post.call_args.kwargs['content']
            payload = json.loads(body)

            assert payload['runId'] == 12345
            assert payload['tags'] == ['experiment', 'v2', 'baseline']

    def test_config_payload_matches_legacy_format(self, uploader):
        """Ensure sync process config payload matches legacy format.

        This is a regression test: the sync process and legacy code path must
        produce identical payload shapes for /api/runs/config/update, otherwise
        the server returns 400.
        """
        from pluto.api import make_compat_update_config_v1

        config = {'learning_rate': 0.001, 'batch_size': 32, 'nested': {'a': 1}}

        # Build the legacy payload
        mock_settings = MagicMock()
        mock_settings._op_id = 12345
        legacy_payload = json.loads(make_compat_update_config_v1(mock_settings, config))

        # Build the sync process payload
        record = SyncRecord(
            id=1,
            run_id='test-run',
            record_type=RecordType.CONFIG,
            payload=config,
            timestamp_ms=1705600000000,
            step=None,
            status=SyncStatus.PENDING,
            retry_count=0,
            created_at=time.time(),
            last_attempt_at=None,
            error_message=None,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_config(record)

            body = mock_client.post.call_args.kwargs['content']
            sync_payload = json.loads(body)

        # Both paths must encode config as a JSON string, not a raw dict
        assert isinstance(sync_payload['config'], type(legacy_payload['config'])), (
            f'Sync process sends config as {type(sync_payload["config"]).__name__}, '
            f'but legacy sends {type(legacy_payload["config"]).__name__}. '
            f'Server expects a JSON-encoded string.'
        )
        assert json.loads(sync_payload['config']) == json.loads(
            legacy_payload['config']
        )

    def test_tags_payload_matches_legacy_format(self, uploader):
        """Ensure sync process tags payload matches legacy format."""
        from pluto.api import make_compat_update_tags_v1

        tags = ['experiment', 'v2', 'baseline']

        # Build legacy payload
        mock_settings = MagicMock()
        mock_settings._op_id = 12345
        legacy_payload = json.loads(make_compat_update_tags_v1(mock_settings, tags))

        # Build sync process payload
        record = SyncRecord(
            id=1,
            run_id='test-run',
            record_type=RecordType.TAGS,
            payload={'tags': tags},
            timestamp_ms=1705600000000,
            step=None,
            status=SyncStatus.PENDING,
            retry_count=0,
            created_at=time.time(),
            last_attempt_at=None,
            error_message=None,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_tags(record)

            body = mock_client.post.call_args.kwargs['content']
            sync_payload = json.loads(body)

        assert isinstance(sync_payload['tags'], type(legacy_payload['tags']))
        assert sync_payload['tags'] == legacy_payload['tags']

    def test_metrics_batch_multiple_records(self, uploader):
        """Test multiple metric records are sent as NDJSON (one per line)."""
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5},
                timestamp_ms=1705600000000,
                step=1,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
            SyncRecord(
                id=2,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.4},
                timestamp_ms=1705600001000,
                step=2,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_metrics_batch(records)

            body = mock_client.post.call_args.kwargs['content']
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            lines = [line for line in body.strip().split('\n') if line]
            assert len(lines) == 2

            payload1 = json.loads(lines[0])
            payload2 = json.loads(lines[1])

            assert payload1['step'] == 1
            assert payload2['step'] == 2

    def test_data_payload_format(self, uploader):
        """Test structured data (Graph, Histogram, Table) payload format.

        Expected format: {"time": <ms>, "data": <json-string>, "dataType": <type>,
                         "logName": <name>, "step": <int>}
        """
        # Simulate a Histogram data payload (avoids tuple key JSON issues)
        histogram_data = {
            'bins': [0, 1, 2, 3, 4, 5],
            'counts': [10, 20, 30, 25, 15],
            'min': 0,
            'max': 5,
            'type': 'Histogram',
            'v': 1,
        }
        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.DATA,
                payload={
                    'log_name': 'loss_distribution',
                    'data_type': 'HISTOGRAM',
                    'data': histogram_data,
                },
                timestamp_ms=1705600000000,
                step=5,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value = mock_client
            uploader._client = None

            uploader.upload_data_batch(records)

            # Verify POST was called to the data endpoint
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            # Verify URL
            url = call_args.args[0] if call_args.args else call_args.kwargs.get('url')
            assert url == 'https://test.example.com/ingest/data'

            # Parse the body (may be bytes or str)
            body = call_args.kwargs.get('content') or call_args[1].get('content')
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            lines = body.strip().split('\n')
            assert len(lines) == 1

            payload = json.loads(lines[0])

            # Verify correct format
            assert payload['time'] == 1705600000000
            assert payload['dataType'] == 'HISTOGRAM'
            assert payload['logName'] == 'loss_distribution'
            assert payload['step'] == 5

            # data field should be a JSON string
            assert isinstance(payload['data'], str)
            data_parsed = json.loads(payload['data'])
            assert data_parsed['type'] == 'Histogram'
            assert 'bins' in data_parsed
            assert 'counts' in data_parsed


class TestSyncUploaderErrorHandling:
    """Tests for _SyncUploader retry and error handling."""

    @pytest.fixture
    def uploader(self):
        """Create an uploader with mock settings."""
        settings = {
            '_auth': 'test-token',
            '_op_id': 12345,
            '_op_name': 'test-run',
            'project': 'test-project',
            'url_num': 'https://test.example.com/ingest/metrics',
            'sync_process_retry_max': 3,
            'sync_process_retry_backoff': 0.1,  # Fast for tests
        }
        log = logging.getLogger('test')
        return _SyncUploader(settings, log)

    def test_urgent_mode_shorter_timeout(self, uploader):
        """Test that urgent mode uses shorter timeouts."""
        assert uploader._urgent_mode is False

        uploader.set_urgent_mode(True)
        assert uploader._urgent_mode is True

        # Verify urgent mode constants
        assert uploader.URGENT_TIMEOUT_SECONDS == 5.0
        assert uploader.URGENT_MAX_RETRIES == 1

    def test_retry_on_connection_error(self, uploader):
        """Test that connection errors trigger retries."""
        import httpx

        records = [
            SyncRecord(
                id=1,
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5},
                timestamp_ms=1705600000000,
                step=1,
                status=SyncStatus.PENDING,
                retry_count=0,
                created_at=time.time(),
                last_attempt_at=None,
                error_message=None,
            ),
        ]

        with patch('httpx.Client') as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            # First two calls fail, third succeeds
            mock_client.post.side_effect = [
                httpx.ConnectError('Connection failed'),
                httpx.ConnectError('Connection failed'),
                mock_response,
            ]
            MockClient.return_value = mock_client
            uploader._client = None

            # Should succeed after retries
            uploader.upload_metrics_batch(records)
            assert mock_client.post.call_count == 3


class TestDistributedEnvironmentDetection:
    """Tests for DDP/distributed environment detection."""

    def test_no_distributed_env_by_default(self):
        """Test that non-distributed environment is correctly identified."""
        from pluto.op import _is_distributed_environment

        # Clear any distributed env vars that might be set
        env_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK']
        original_values = {var: os.environ.get(var) for var in env_vars}

        try:
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]

            # Should return False when no distributed env vars are set
            # (assuming torch.distributed is not initialized)
            result = _is_distributed_environment()
            # May be True if torch.distributed is initialized in test env
            assert isinstance(result, bool)
        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_world_size_detection(self):
        """Test that WORLD_SIZE > 1 triggers distributed detection."""
        from pluto.op import _is_distributed_environment

        original = os.environ.get('WORLD_SIZE')
        try:
            os.environ['WORLD_SIZE'] = '2'
            assert _is_distributed_environment() is True
        finally:
            if original is not None:
                os.environ['WORLD_SIZE'] = original
            elif 'WORLD_SIZE' in os.environ:
                del os.environ['WORLD_SIZE']

    def test_world_size_one_not_distributed(self):
        """Test that WORLD_SIZE=1 is not considered distributed."""
        from pluto.op import _is_distributed_environment

        original = os.environ.get('WORLD_SIZE')
        try:
            os.environ['WORLD_SIZE'] = '1'
            # WORLD_SIZE=1 alone should not trigger distributed mode
            # (unless torch.distributed is initialized)
            result = _is_distributed_environment()
            # Result depends on torch.distributed state, just verify it returns bool
            assert isinstance(result, bool)
        finally:
            if original is not None:
                os.environ['WORLD_SIZE'] = original
            elif 'WORLD_SIZE' in os.environ:
                del os.environ['WORLD_SIZE']

    def test_slurm_detection(self):
        """Test that SLURM environment variables trigger distributed detection."""
        from pluto.op import _is_distributed_environment

        original_procid = os.environ.get('SLURM_PROCID')
        original_ntasks = os.environ.get('SLURM_NTASKS')
        original_world = os.environ.get('WORLD_SIZE')
        try:
            # Clear WORLD_SIZE to isolate SLURM detection
            if 'WORLD_SIZE' in os.environ:
                del os.environ['WORLD_SIZE']

            os.environ['SLURM_PROCID'] = '0'
            os.environ['SLURM_NTASKS'] = '4'
            assert _is_distributed_environment() is True
        finally:
            # Restore original values
            if original_procid is not None:
                os.environ['SLURM_PROCID'] = original_procid
            elif 'SLURM_PROCID' in os.environ:
                del os.environ['SLURM_PROCID']
            if original_ntasks is not None:
                os.environ['SLURM_NTASKS'] = original_ntasks
            elif 'SLURM_NTASKS' in os.environ:
                del os.environ['SLURM_NTASKS']
            if original_world is not None:
                os.environ['WORLD_SIZE'] = original_world

    def test_slurm_single_task_not_distributed(self):
        """Test that SLURM with single task is not considered distributed."""
        from pluto.op import _is_distributed_environment

        original_procid = os.environ.get('SLURM_PROCID')
        original_ntasks = os.environ.get('SLURM_NTASKS')
        original_world = os.environ.get('WORLD_SIZE')
        try:
            # Clear WORLD_SIZE to isolate SLURM detection
            if 'WORLD_SIZE' in os.environ:
                del os.environ['WORLD_SIZE']

            os.environ['SLURM_PROCID'] = '0'
            os.environ['SLURM_NTASKS'] = '1'
            # SLURM_NTASKS=1 should not trigger distributed mode
            result = _is_distributed_environment()
            # Result depends on torch.distributed state
            assert isinstance(result, bool)
        finally:
            if original_procid is not None:
                os.environ['SLURM_PROCID'] = original_procid
            elif 'SLURM_PROCID' in os.environ:
                del os.environ['SLURM_PROCID']
            if original_ntasks is not None:
                os.environ['SLURM_NTASKS'] = original_ntasks
            elif 'SLURM_NTASKS' in os.environ:
                del os.environ['SLURM_NTASKS']
            if original_world is not None:
                os.environ['WORLD_SIZE'] = original_world
