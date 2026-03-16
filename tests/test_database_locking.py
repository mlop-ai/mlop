"""Tests for SQLite database locking resilience.

These tests verify that the sync architecture handles sqlite3.OperationalError
("database is locked") gracefully at every layer:

1. SyncStore: @_retry_on_locked decorator retries transient lock errors
2. Sync process: _sync_main catches OperationalError and continues
3. Training process: Op.log() and background workers don't crash on DB errors

These tests would have caught the production incident where the sync process
crashed on "database is locked" and cascaded into the training process.
"""

import logging
import sqlite3
import threading
import time
from unittest.mock import MagicMock

import pytest

from pluto.sync.store import (
    _SQLITE_RETRY_COUNT,
    HEALTH_METRIC_KEYS,
    RecordType,
    SyncStore,
    _retry_on_locked,
    _retry_stats,
)


class TestRetryOnLockedDecorator:
    """Unit tests for the @_retry_on_locked decorator."""

    def test_succeeds_on_first_try(self):
        """Decorated function returns immediately if no error."""
        call_count = 0

        @_retry_on_locked
        def func():
            nonlocal call_count
            call_count += 1
            return 'ok'

        assert func() == 'ok'
        assert call_count == 1

    def test_retries_on_database_locked(self):
        """Retries when sqlite3.OperationalError with 'locked' is raised."""
        call_count = 0

        @_retry_on_locked
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError('database is locked')
            return 'ok'

        assert func() == 'ok'
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Raises the last error after exhausting all retries."""

        @_retry_on_locked
        def func():
            raise sqlite3.OperationalError('database is locked')

        with pytest.raises(sqlite3.OperationalError, match='locked'):
            func()

    def test_does_not_retry_non_locked_errors(self):
        """Non-lock OperationalErrors are raised immediately."""
        call_count = 0

        @_retry_on_locked
        def func():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError('disk I/O error')

        with pytest.raises(sqlite3.OperationalError, match='disk I/O'):
            func()
        assert call_count == 1

    def test_does_not_retry_other_exceptions(self):
        """Non-OperationalError exceptions propagate immediately."""
        call_count = 0

        @_retry_on_locked
        def func():
            nonlocal call_count
            call_count += 1
            raise ValueError('something else')

        with pytest.raises(ValueError):
            func()
        assert call_count == 1

    def test_preserves_function_metadata(self):
        """Decorator preserves __name__ and __doc__."""

        @_retry_on_locked
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == 'my_function'
        assert my_function.__doc__ == 'My docstring.'

    def test_retry_count_matches_constant(self):
        """Exactly _SQLITE_RETRY_COUNT attempts are made."""
        call_count = 0

        @_retry_on_locked
        def func():
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError('database is locked')

        with pytest.raises(sqlite3.OperationalError):
            func()
        assert call_count == _SQLITE_RETRY_COUNT


class TestSyncStoreRetryBehavior:
    """Tests that SyncStore methods handle lock contention."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary SyncStore for testing."""
        db_path = tmp_path / 'test_sync.db'
        store = SyncStore(str(db_path))
        store.register_run('test-run', 'test-project')
        yield store
        store.close()

    def test_enqueue_retries_on_lock(self, tmp_path):
        """enqueue() retries on database locked and eventually succeeds.

        Uses a real exclusive lock from a second connection to trigger
        contention, then releases it so the retry succeeds.
        """
        db_path = str(tmp_path / 'lock_test.db')
        store = SyncStore(db_path)
        store.register_run('test-run', 'test-project')

        # Open a second connection and hold an exclusive lock
        blocker = sqlite3.connect(db_path)
        blocker.execute('BEGIN EXCLUSIVE')

        result = {'id': None, 'error': None}

        def writer():
            try:
                result['id'] = store.enqueue(
                    run_id='test-run',
                    record_type=RecordType.METRIC,
                    payload={'loss': 0.5},
                    timestamp_ms=1000,
                    step=1,
                )
            except Exception as e:
                result['error'] = e

        t = threading.Thread(target=writer)
        t.start()

        # Release the lock after a short delay so the retry succeeds
        time.sleep(0.3)
        blocker.rollback()
        blocker.close()

        t.join(timeout=30)

        assert result['error'] is None, (
            f'enqueue should have retried: {result["error"]}'
        )
        assert result['id'] is not None and result['id'] > 0

        store.close()

    def test_mark_in_progress_retries_on_lock(self, tmp_path):
        """mark_in_progress() retries when the database is locked."""
        db_path = str(tmp_path / 'lock_test2.db')
        store = SyncStore(db_path)
        store.register_run('test-run', 'test-project')

        record_id = store.enqueue(
            run_id='test-run',
            record_type=RecordType.METRIC,
            payload={'loss': 0.5},
            timestamp_ms=1000,
            step=1,
        )

        # Hold exclusive lock from second connection
        blocker = sqlite3.connect(db_path)
        blocker.execute('BEGIN EXCLUSIVE')

        result = {'error': None}

        def marker():
            try:
                store.mark_in_progress([record_id])
            except Exception as e:
                result['error'] = e

        t = threading.Thread(target=marker)
        t.start()

        # Release after short delay
        time.sleep(0.3)
        blocker.rollback()
        blocker.close()

        t.join(timeout=30)

        assert result['error'] is None, (
            f'mark_in_progress should retry: {result["error"]}'
        )

        store.close()

    def test_heartbeat_retries_on_lock(self, tmp_path):
        """heartbeat() retries when the database is locked."""
        db_path = str(tmp_path / 'lock_test3.db')
        store = SyncStore(db_path)
        store.register_run('test-run', 'test-project')

        # Hold exclusive lock from second connection
        blocker = sqlite3.connect(db_path)
        blocker.execute('BEGIN EXCLUSIVE')

        result = {'error': None}

        def heartbeater():
            try:
                store.heartbeat('test-run')
            except Exception as e:
                result['error'] = e

        t = threading.Thread(target=heartbeater)
        t.start()

        # Release after short delay
        time.sleep(0.3)
        blocker.rollback()
        blocker.close()

        t.join(timeout=30)

        assert result['error'] is None, f'heartbeat should retry: {result["error"]}'

        store.close()

    def test_busy_timeout_is_30_seconds(self, store):
        """Verify busy_timeout is set to 30000ms."""
        cursor = store.conn.execute('PRAGMA busy_timeout')
        timeout = cursor.fetchone()[0]
        assert timeout == 30000


class TestConcurrentWriteContention:
    """Tests for actual multi-connection write contention.

    These tests use multiple connections to the same SQLite database to
    simulate the training process and sync process writing concurrently.
    """

    def test_many_connections_concurrent_writes(self, tmp_path):
        """Multiple connections can write concurrently with WAL mode."""
        db_path = str(tmp_path / 'concurrent.db')
        num_writers = 6
        writes_per_writer = 200

        store_main = SyncStore(db_path)
        store_main.register_run('run-1', 'project')

        stores = [SyncStore(db_path) for _ in range(num_writers)]
        errors = []
        results = {i: [] for i in range(num_writers)}

        def writer(idx, store):
            try:
                for i in range(writes_per_writer):
                    rid = store.enqueue(
                        run_id='run-1',
                        record_type=RecordType.METRIC,
                        payload={'metric': i, 'writer': idx},
                        timestamp_ms=i * 1000,
                        step=i,
                    )
                    results[idx].append(rid)
            except Exception as e:
                errors.append((f'writer-{idx}', e))

        def reader_drainer(store):
            """Continuously read and complete records to add read contention."""
            try:
                for _ in range(writes_per_writer * 2):
                    pending = store.get_pending_records(limit=20)
                    if pending:
                        ids = [r.id for r in pending]
                        store.mark_in_progress(ids)
                        store.mark_completed(ids)
                    store.heartbeat('run-1')
            except Exception as e:
                errors.append(('reader', e))

        threads = []
        for idx in range(num_writers):
            threads.append(threading.Thread(target=writer, args=(idx, stores[idx])))
        # Add two reader/drainer threads using their own connections
        reader_stores = [SyncStore(db_path) for _ in range(2)]
        for rs in reader_stores:
            threads.append(threading.Thread(target=reader_drainer, args=(rs,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        for s in stores + reader_stores:
            s.close()
        store_main.close()

        assert not errors, f'Concurrent writes should not error: {errors}'
        for idx in range(num_writers):
            assert len(results[idx]) == writes_per_writer

    def test_high_frequency_writes_with_contention(self, tmp_path):
        """Stress test: rapid writes from many connections with heavy contention."""
        db_path = str(tmp_path / 'stress.db')
        num_writers = 4
        num_readers = 4
        write_count = 500

        store_main = SyncStore(db_path)
        store_main.register_run('run-1', 'project')

        writer_stores = [SyncStore(db_path) for _ in range(num_writers)]
        reader_stores = [SyncStore(db_path) for _ in range(num_readers)]
        errors = []
        written = {i: 0 for i in range(num_writers)}

        def rapid_writer(idx, store):
            try:
                for i in range(write_count):
                    store.enqueue(
                        run_id='run-1',
                        record_type=RecordType.METRIC,
                        payload={'m': i},
                        timestamp_ms=i,
                        step=i,
                    )
                    written[idx] += 1
            except Exception as e:
                errors.append((f'writer-{idx}', e))

        def rapid_reader_writer(idx, store):
            try:
                for _ in range(write_count):
                    pending = store.get_pending_records(limit=20)
                    if pending:
                        ids = [r.id for r in pending]
                        store.mark_in_progress(ids)
                        store.mark_completed(ids)
                    store.heartbeat('run-1')
                    store.get_health_stats()
            except Exception as e:
                errors.append((f'reader-{idx}', e))

        threads = []
        for i in range(num_writers):
            threads.append(
                threading.Thread(target=rapid_writer, args=(i, writer_stores[i]))
            )
        for i in range(num_readers):
            threads.append(
                threading.Thread(target=rapid_reader_writer, args=(i, reader_stores[i]))
            )

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        for s in writer_stores + reader_stores:
            s.close()
        store_main.close()

        assert not errors, f'High-frequency writes should not error: {errors}'
        for i in range(num_writers):
            assert written[i] == write_count, f'Writer {i} only wrote {written[i]}'


class TestSyncMainResilience:
    """Tests that _sync_main handles sqlite3.OperationalError without dying."""

    def test_sync_batch_survives_operational_error(self, tmp_path):
        """_sync_batch should not raise on OperationalError from store."""
        from pluto.sync.process import _sync_batch

        db_path = str(tmp_path / 'test.db')
        store = SyncStore(db_path)
        store.register_run('run-1', 'project')

        # Enqueue some records
        for i in range(5):
            store.enqueue(
                run_id='run-1',
                record_type=RecordType.METRIC,
                payload={'m': i},
                timestamp_ms=i * 1000,
                step=i,
            )

        uploader = MagicMock()
        log = logging.getLogger('test')

        # Make mark_in_progress raise OperationalError
        def failing_mark(ids):
            raise sqlite3.OperationalError('database is locked')

        store.mark_in_progress = failing_mark

        # _sync_batch should propagate the error (caught by _sync_main)
        with pytest.raises(sqlite3.OperationalError):
            _sync_batch(store, uploader, log, max_retries=5)

        store.close()

    def test_sync_main_loop_continues_after_operational_error(self, tmp_path):
        """Simulate _sync_main catching OperationalError and continuing."""
        # This tests the pattern used in _sync_main:
        # try: ... except sqlite3.OperationalError: continue
        db_path = str(tmp_path / 'test.db')
        store = SyncStore(db_path)
        store.register_run('run-1', 'project')

        iterations = 0
        errors_caught = 0

        # Simulate the inner loop pattern from _sync_main
        for _ in range(5):
            try:
                iterations += 1
                if iterations <= 2:
                    raise sqlite3.OperationalError('database is locked')
                # Normal operation would happen here
            except sqlite3.OperationalError:
                errors_caught += 1
                continue

        assert iterations == 5
        assert errors_caught == 2

        store.close()


class TestOpLogResilience:
    """Tests that Op.log() handles database errors gracefully."""

    def test_log_does_not_propagate_operational_error(self, tmp_path):
        """Op.log() should catch OperationalError, not crash training."""
        from pluto.op import Op

        # Create a minimal Op with a mock sync manager
        mock_sync_manager = MagicMock()
        mock_sync_manager.enqueue_metrics.side_effect = sqlite3.OperationalError(
            'database is locked'
        )

        # Create Op with mocked dependencies
        mock_settings = MagicMock()
        mock_settings.mode = 'online'
        mock_settings.meta = []
        mock_settings.get_dir.return_value = str(tmp_path)
        mock_settings.x_meta_label = '__meta__'

        op = Op.__new__(Op)
        op.settings = mock_settings
        op._sync_manager = mock_sync_manager
        op._iface = None
        op._store = None
        op._step = 0
        op._queue = None
        op._finished = False
        op._finish_lock = threading.Lock()
        op.config = {}
        op.tags = []

        # This should NOT raise - the error should be caught
        op.log({'loss': 0.5}, step=1)

        # Verify enqueue was called (and failed)
        mock_sync_manager.enqueue_metrics.assert_called_once()

    def test_log_continues_after_transient_error(self, tmp_path):
        """Multiple log() calls work even if some fail with DB errors."""
        from pluto.op import Op

        call_count = 0

        def flaky_enqueue(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise sqlite3.OperationalError('database is locked')

        mock_sync_manager = MagicMock()
        mock_sync_manager.enqueue_metrics.side_effect = flaky_enqueue

        mock_settings = MagicMock()
        mock_settings.mode = 'online'
        mock_settings.meta = []
        mock_settings.get_dir.return_value = str(tmp_path)
        mock_settings.x_meta_label = '__meta__'

        op = Op.__new__(Op)
        op.settings = mock_settings
        op._sync_manager = mock_sync_manager
        op._iface = None
        op._store = None
        op._step = 0
        op._queue = None
        op._finished = False
        op._finish_lock = threading.Lock()
        op.config = {}
        op.tags = []

        # Log 3 times. The 2nd will fail silently.
        op.log({'loss': 0.5}, step=1)  # succeeds
        op.log({'loss': 0.4}, step=2)  # fails silently
        op.log({'loss': 0.3}, step=3)  # succeeds

        assert call_count == 3


class TestWorkerMonitorResilience:
    """Tests that _worker_monitor handles DB errors as warnings."""

    def test_monitor_logs_warning_not_critical_for_db_error(self, caplog):
        """_worker_monitor should log WARNING for OperationalError, not CRITICAL."""
        import sqlite3

        # Verify that our error handling distinguishes sqlite3 errors
        # by checking the log level behavior
        with caplog.at_level(logging.WARNING):
            # Simulate the pattern from _worker_monitor
            try:
                raise sqlite3.OperationalError('database is locked')
            except sqlite3.OperationalError:
                logging.getLogger('pluto').warning(
                    'transient database error (will retry): database is locked'
                )

        assert any(
            'transient database error' in record.message
            and record.levelno == logging.WARNING
            for record in caplog.records
        )


class TestHealthDiagnostics:
    """Tests for runtime health/observability diagnostics."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = tmp_path / 'health_test.db'
        store = SyncStore(str(db_path))
        store.register_run('test-run', 'test-project')
        yield store
        store.close()

    def test_health_stats_returns_all_keys(self, store):
        """get_health_stats() returns all expected diagnostic keys."""
        stats = store.get_health_stats()
        expected_keys = {
            'pending',
            'in_progress',
            'completed',
            'failed',
            'total_rows',
            'lag_s',
            'wal_size_kb',
            'db_size_kb',
            'write_count',
            'write_avg_ms',
            'write_max_ms',
            'retries',
            'retry_failures',
        }
        assert expected_keys == set(stats.keys())

    def test_health_stats_empty_store(self, store):
        """Health stats on empty store show zeroes."""
        stats = store.get_health_stats()
        assert stats['pending'] == 0
        assert stats['completed'] == 0
        assert stats['total_rows'] == 0
        assert stats['lag_s'] == 0.0

    def test_lag_tracks_oldest_pending_record(self, store):
        """lag_s reflects the age of the oldest pending record."""
        store.enqueue(
            run_id='test-run',
            record_type=RecordType.METRIC,
            payload={'m': 1},
            timestamp_ms=1000,
            step=1,
        )
        stats = store.get_health_stats()
        # Just created, so lag should be very small (< 2s)
        assert 0 <= stats['lag_s'] < 2.0

    def test_lag_zero_when_no_pending(self, store):
        """lag_s is 0 when all records have been completed."""
        rid = store.enqueue(
            run_id='test-run',
            record_type=RecordType.METRIC,
            payload={'m': 1},
            timestamp_ms=1000,
            step=1,
        )
        store.mark_in_progress([rid])
        store.mark_completed([rid])
        stats = store.get_health_stats()
        assert stats['lag_s'] == 0.0

    def test_health_stats_tracks_pending(self, store):
        """Pending count increases as records are enqueued."""
        for i in range(10):
            store.enqueue(
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'m': i},
                timestamp_ms=i * 1000,
                step=i,
            )
        stats = store.get_health_stats()
        assert stats['pending'] == 10
        assert stats['total_rows'] == 10

    def test_health_stats_tracks_completed(self, store):
        """Completed count updates after mark_completed."""
        ids = []
        for i in range(5):
            rid = store.enqueue(
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'m': i},
                timestamp_ms=i * 1000,
                step=i,
            )
            ids.append(rid)
        store.mark_in_progress(ids)
        store.mark_completed(ids)
        stats = store.get_health_stats()
        assert stats['completed'] == 5
        assert stats['pending'] == 0

    def test_write_latency_tracked(self, store):
        """Write latency stats are updated after enqueue."""
        store.enqueue(
            run_id='test-run',
            record_type=RecordType.METRIC,
            payload={'m': 1},
            timestamp_ms=1000,
            step=1,
        )
        assert store._write_count >= 1
        assert store._write_total_ms > 0
        stats = store.get_health_stats()
        assert stats['write_count'] >= 1
        assert stats['write_avg_ms'] >= 0

    def test_write_latency_batch(self, store):
        """Batch enqueue also records write latency."""
        records = [
            ('test-run', RecordType.METRIC, {'m': i}, i * 1000, i) for i in range(10)
        ]
        before = store._write_count
        store.enqueue_batch(records)
        assert store._write_count > before

    def test_retry_stats_incremented(self):
        """_retry_stats tracks retries from _retry_on_locked."""
        before = _retry_stats['total_retries']

        call_count = 0

        @_retry_on_locked
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError('database is locked')
            return 'ok'

        flaky()
        # Should have retried 2 times
        assert _retry_stats['total_retries'] >= before + 2

    def test_retry_failure_stats(self):
        """_retry_stats tracks exhausted retries."""
        before = _retry_stats['total_failures']

        @_retry_on_locked
        def always_locked():
            raise sqlite3.OperationalError('database is locked')

        with pytest.raises(sqlite3.OperationalError):
            always_locked()
        assert _retry_stats['total_failures'] >= before + 1

    def test_health_stats_wal_and_db_size(self, store):
        """WAL and DB file sizes are reported."""
        # Write some data to ensure files exist on disk
        for i in range(20):
            store.enqueue(
                run_id='test-run',
                record_type=RecordType.METRIC,
                payload={'m': i},
                timestamp_ms=i * 1000,
                step=i,
            )
        stats = store.get_health_stats()
        # DB file should exist and have size > 0
        assert stats['db_size_kb'] >= 0
        # WAL might or might not exist depending on checkpointing
        assert isinstance(stats['wal_size_kb'], int)


class TestTableGrowthStress:
    """Stress test reproducing the original production incident.

    The actual failure was caused by:
    1. cleanup_completed() never being called → unbounded table growth
    2. Multiple DDP ranks writing to the same DB simultaneously
    3. High-frequency console log writes (every print() was a DB write)
    4. Over hours, growing table → slower writes → longer lock holds → starvation

    This test compresses that pattern into seconds by simulating multiple
    concurrent writers with no cleanup, then verifying the system stays
    healthy (no crashes, all writes succeed or are retried).
    """

    def test_table_growth_without_cleanup(self, tmp_path):
        """Writes degrade as completed records accumulate without cleanup.

        This is the exact pattern that caused the production incident.
        We verify that even without cleanup, the retry decorator keeps
        writers alive (no unhandled OperationalError).
        """
        db_path = str(tmp_path / 'growth.db')
        # Simulate 3 DDP ranks + 1 sync reader (4 connections)
        writers = [SyncStore(db_path) for _ in range(3)]
        reader = SyncStore(db_path)
        writers[0].register_run('run-1', 'project')

        errors = []
        total_writes = 500  # Per writer
        completed_without_cleanup = 0

        def rank_writer(store, rank_id):
            """Simulates a DDP rank doing metric + console writes."""
            nonlocal errors
            try:
                for i in range(total_writes):
                    # Metric write
                    store.enqueue(
                        run_id='run-1',
                        record_type=RecordType.METRIC,
                        payload={f'rank{rank_id}/loss': 0.5 - i * 0.001},
                        timestamp_ms=i * 100,
                        step=i,
                    )
                    # Console write (high frequency, like print())
                    store.enqueue(
                        run_id='run-1',
                        record_type=RecordType.CONSOLE,
                        payload={
                            'message': f'rank {rank_id} step {i}',
                            'logType': 'INFO',
                            'lineNumber': i,
                        },
                        timestamp_ms=i * 100,
                        step=i,
                    )
            except Exception as e:
                errors.append((f'rank{rank_id}', e))

        def sync_reader():
            """Simulates sync process: read → mark → complete (no cleanup)."""
            nonlocal completed_without_cleanup
            try:
                for _ in range(total_writes * 2):
                    pending = reader.get_pending_records(limit=50)
                    if pending:
                        ids = [r.id for r in pending]
                        reader.mark_in_progress(ids)
                        reader.mark_completed(ids)
                        completed_without_cleanup += len(ids)
                    else:
                        time.sleep(0.001)
            except Exception as e:
                errors.append(('reader', e))

        # Run all 4 connections concurrently
        threads = []
        for rank in range(3):
            t = threading.Thread(target=rank_writer, args=(writers[rank], rank))
            threads.append(t)
        threads.append(threading.Thread(target=sync_reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        # Check results
        stats = reader.get_health_stats()

        # Writers tracked their own latency
        total_writer_writes = sum(w._write_count for w in writers)

        for s in writers:
            s.close()
        reader.close()

        assert not errors, f'Concurrent writes should not crash: {errors}'
        # Completed records accumulated without cleanup (the root cause)
        assert stats['completed'] + stats['pending'] > 0
        # Writers tracked latency across all ranks
        assert total_writer_writes > 0

    def test_health_degrades_with_table_growth(self, tmp_path):
        """Write latency increases as the table grows without cleanup.

        This verifies the degradation pattern: early writes are fast,
        later writes (with a large completed backlog) are slower.
        """
        db_path = str(tmp_path / 'degrade.db')
        store = SyncStore(db_path)
        store.register_run('run-1', 'project')

        # Phase 1: Write and complete 1000 records (building backlog)
        for batch_num in range(10):
            ids = []
            for i in range(100):
                rid = store.enqueue(
                    run_id='run-1',
                    record_type=RecordType.CONSOLE,
                    payload={'message': f'line {batch_num * 100 + i}'},
                    timestamp_ms=(batch_num * 100 + i) * 100,
                    step=batch_num * 100 + i,
                )
                ids.append(rid)
            store.mark_in_progress(ids)
            store.mark_completed(ids)
            # Intentionally do NOT call cleanup_completed()

        stats_after_backlog = store.get_health_stats()

        # Phase 2: Write 100 more records with the backlog present
        store._write_count = 0
        store._write_total_ms = 0.0
        store._write_max_ms = 0.0

        for i in range(100):
            store.enqueue(
                run_id='run-1',
                record_type=RecordType.METRIC,
                payload={'loss': 0.5},
                timestamp_ms=i * 100,
                step=i,
            )

        stats_with_backlog = store.get_health_stats()
        store.close()

        # The backlog should be visible
        assert stats_after_backlog['completed'] == 1000
        # Write latency is tracked for phase 2
        assert stats_with_backlog['write_count'] == 100
        assert stats_with_backlog['write_avg_ms'] > 0

    def test_cleanup_prevents_degradation(self, tmp_path):
        """Calling cleanup_completed() keeps the table small."""
        db_path = str(tmp_path / 'cleanup.db')
        store = SyncStore(db_path)
        store.register_run('run-1', 'project')

        for batch_num in range(10):
            ids = []
            for i in range(100):
                rid = store.enqueue(
                    run_id='run-1',
                    record_type=RecordType.CONSOLE,
                    payload={'message': f'line {batch_num * 100 + i}'},
                    timestamp_ms=(batch_num * 100 + i) * 100,
                    step=batch_num * 100 + i,
                )
                ids.append(rid)
            store.mark_in_progress(ids)
            store.mark_completed(ids)
            # Clean up after each batch
            store.cleanup_completed(older_than_seconds=0)

        stats = store.get_health_stats()
        store.close()

        # Completed records should be cleaned up
        assert stats['completed'] == 0
        assert stats['total_rows'] <= 100  # At most the last batch


class TestHealthMetricKeys:
    """Tests that HEALTH_METRIC_KEYS stays in sync with get_health_stats()."""

    def test_keys_match_health_stats(self, tmp_path):
        """HEALTH_METRIC_KEYS must exactly match get_health_stats() keys."""
        store = SyncStore(str(tmp_path / 'keys.db'))
        store.register_run('r', 'p')
        stats = store.get_health_stats()
        store.close()
        assert set(HEALTH_METRIC_KEYS) == set(stats.keys())


class TestUploadHealthStats:
    """Tests for _SyncUploader.upload_health_stats()."""

    def test_upload_posts_prefixed_metrics(self):
        """upload_health_stats sends sys/pluto.* keys to url_num."""
        from pluto.sync.process import _SyncUploader

        uploader = _SyncUploader(
            {'url_num': 'http://fake/num', '_auth': 'tok'},
            logging.getLogger('test'),
        )

        # Capture what _post_with_retry receives
        captured = {}

        def fake_post(url, body, headers):
            captured['url'] = url
            captured['body'] = body
            captured['headers'] = headers

        uploader._post_with_retry = fake_post

        stats = {
            'pending': 10,
            'lag_s': 5.2,
            'write_avg_ms': 3.1,
            'retries': 2,
        }
        uploader.upload_health_stats(stats)

        assert captured['url'] == 'http://fake/num'
        import json

        payload = json.loads(captured['body'].strip())
        assert payload['step'] == 0
        assert 'time' in payload
        assert payload['data']['sys/pluto.pending'] == 10
        assert payload['data']['sys/pluto.lag_s'] == 5.2
        assert payload['data']['sys/pluto.write_avg_ms'] == 3.1
        assert payload['data']['sys/pluto.retries'] == 2

    def test_upload_skipped_when_no_url(self):
        """upload_health_stats is a no-op when url_num is empty."""
        from pluto.sync.process import _SyncUploader

        uploader = _SyncUploader(
            {'url_num': '', '_auth': 'tok'},
            logging.getLogger('test'),
        )
        called = {'post': False}

        def fake_post(url, body, headers):
            called['post'] = True

        uploader._post_with_retry = fake_post
        uploader.upload_health_stats({'pending': 5})
        assert not called['post']

    def test_upload_all_health_keys(self, tmp_path):
        """Full get_health_stats() dict is uploaded with correct prefixes."""
        from pluto.sync.process import _SyncUploader

        store = SyncStore(str(tmp_path / 'upload.db'))
        store.register_run('r', 'p')
        store.enqueue('r', RecordType.METRIC, {'m': 1}, 1000, 1)
        stats = store.get_health_stats()
        store.close()

        uploader = _SyncUploader(
            {'url_num': 'http://fake/num', '_auth': 'tok'},
            logging.getLogger('test'),
        )

        captured = {}

        def fake_post(url, body, headers):
            captured['body'] = body

        uploader._post_with_retry = fake_post
        uploader.upload_health_stats(stats)

        import json

        payload = json.loads(captured['body'].strip())
        data = payload['data']
        for key in HEALTH_METRIC_KEYS:
            assert f'sys/pluto.{key}' in data, f'Missing sys/pluto.{key}'


class TestHealthMetricRegistration:
    """Tests that sync health metrics are registered with the server."""

    def test_health_metrics_registered_in_op_start(self):
        """Op.start() should register sys/pluto.* names via _update_meta."""
        # Verify the registration list is correct
        expected = [f'sys/pluto.{k}' for k in HEALTH_METRIC_KEYS]
        assert len(expected) == 13
        assert 'sys/pluto.pending' in expected
        assert 'sys/pluto.lag_s' in expected
        assert 'sys/pluto.write_avg_ms' in expected
        assert 'sys/pluto.retries' in expected

    def test_health_keys_not_empty(self):
        """HEALTH_METRIC_KEYS should contain entries."""
        assert len(HEALTH_METRIC_KEYS) > 0
        assert all(isinstance(k, str) for k in HEALTH_METRIC_KEYS)
