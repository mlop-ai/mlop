"""
Comprehensive tests for Neptune-to-pluto compatibility layer.

These tests validate that:
1. Neptune API calls continue to work unchanged
2. pluto receives the logged data when configured
3. Neptune never fails due to pluto errors
4. Configuration via environment variables works
5. Fallback behavior is correct
"""

import logging

# CRITICAL: Set this FIRST, before any other imports that might configure logging.
# This prevents Python's logging module from printing error messages when
# handlers fail (e.g., "I/O operation on closed file" errors).
logging.raiseExceptions = False

# Suppress neptune_scale logging BEFORE importing neptune_scale
# This prevents "I/O operation on closed file" errors when pytest
# captures stdout/stderr and Neptune's async operations try to log
# after the streams are closed.
for logger_name in [
    'neptune_scale',
    'neptune_scale.util.logger',
    'neptune_scale.api.run',
    'neptune',
]:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.CRITICAL + 1)
    _logger.disabled = True
    _logger.addHandler(logging.NullHandler())

import os
import sys
import uuid
from typing import Any, Dict
from unittest import mock
from unittest.mock import call

import pytest

# Test both with and without neptune installed
pytest.importorskip('neptune_scale')

# Import after neptune is available
from neptune_scale.types import File as NeptuneFile
from neptune_scale.types import Histogram as NeptuneHistogram

from pluto.compat.neptune import _detect_media_type_from_bytes
from tests.utils import get_task_name


class TestMagicByteDetection:
    """Unit tests for _detect_media_type_from_bytes function."""

    def test_png_detection(self):
        """Test PNG magic bytes detection."""
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert _detect_media_type_from_bytes(png_bytes) == 'image'

    def test_jpeg_detection(self):
        """Test JPEG magic bytes detection."""
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert _detect_media_type_from_bytes(jpeg_bytes) == 'image'

    def test_gif87a_detection(self):
        """Test GIF87a magic bytes detection."""
        gif_bytes = b'GIF87a' + b'\x00' * 100
        assert _detect_media_type_from_bytes(gif_bytes) == 'image'

    def test_gif89a_detection(self):
        """Test GIF89a magic bytes detection."""
        gif_bytes = b'GIF89a' + b'\x00' * 100
        assert _detect_media_type_from_bytes(gif_bytes) == 'image'

    def test_bmp_detection(self):
        """Test BMP magic bytes detection."""
        bmp_bytes = b'BM' + b'\x00' * 100
        assert _detect_media_type_from_bytes(bmp_bytes) == 'image'

    def test_webp_detection(self):
        """Test WebP magic bytes detection."""
        webp_bytes = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100
        assert _detect_media_type_from_bytes(webp_bytes) == 'image'

    def test_wav_detection(self):
        """Test WAV magic bytes detection."""
        wav_bytes = b'RIFF\x00\x00\x00\x00WAVE' + b'\x00' * 100
        assert _detect_media_type_from_bytes(wav_bytes) == 'audio'

    def test_mp3_id3_detection(self):
        """Test MP3 ID3 header detection."""
        mp3_bytes = b'ID3' + b'\x00' * 100
        assert _detect_media_type_from_bytes(mp3_bytes) == 'audio'

    def test_flac_detection(self):
        """Test FLAC magic bytes detection."""
        flac_bytes = b'fLaC' + b'\x00' * 100
        assert _detect_media_type_from_bytes(flac_bytes) == 'audio'

    def test_ogg_detection(self):
        """Test OGG magic bytes detection (defaults to audio)."""
        ogg_bytes = b'OggS' + b'\x00' * 100
        assert _detect_media_type_from_bytes(ogg_bytes) == 'audio'

    def test_mp4_detection(self):
        """Test MP4/MOV magic bytes detection."""
        mp4_bytes = b'\x00\x00\x00\x20ftyp' + b'\x00' * 100
        assert _detect_media_type_from_bytes(mp4_bytes) == 'video'

    def test_unknown_bytes_returns_none(self):
        """Test that unknown byte sequences return None."""
        unknown_bytes = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d'
        assert _detect_media_type_from_bytes(unknown_bytes) is None

    def test_too_short_bytes_returns_none(self):
        """Test that bytes shorter than 12 return None."""
        short_bytes = b'\x89PNG\r\n\x1a'  # Only 7 bytes
        assert _detect_media_type_from_bytes(short_bytes) is None

    def test_non_bytes_returns_none(self):
        """Test that non-bytes input returns None."""
        assert _detect_media_type_from_bytes('not bytes') is None
        assert _detect_media_type_from_bytes(12345) is None
        assert _detect_media_type_from_bytes(None) is None


class MockNeptuneRun:
    """
    Mock Neptune Run for testing without actual Neptune backend.

    This simulates Neptune's behavior for testing the monkeypatch.
    """

    def __init__(self, *args, **kwargs):
        self.experiment_name = kwargs.get('experiment_name', 'test-experiment')
        self.run_id = kwargs.get('run_id') or uuid.uuid4().hex[:8]
        self._run_id = self.run_id
        self.project = kwargs.get('project', 'test/project')
        self.logged_metrics = []
        self.logged_configs = []
        self.logged_files = []
        self.logged_histograms = []
        self.tags = []
        self.closed = False
        self.terminated = False

    def log_metrics(self, data: Dict[str, float], step: int, timestamp=None, **kwargs):
        self.logged_metrics.append({'data': data, 'step': step, 'timestamp': timestamp})
        return None

    def log_configs(self, data: Dict[str, Any], **kwargs):
        self.logged_configs.append(data)
        return None

    def assign_files(self, files: Dict[str, Any], **kwargs):
        self.logged_files.append({'type': 'assign', 'files': files})
        return None

    def log_files(self, files: Dict[str, Any], step: int, timestamp=None, **kwargs):
        self.logged_files.append(
            {'type': 'log', 'files': files, 'step': step, 'timestamp': timestamp}
        )
        return None

    def log_histograms(
        self, histograms: Dict[str, Any], step: int, timestamp=None, **kwargs
    ):
        self.logged_histograms.append(
            {'histograms': histograms, 'step': step, 'timestamp': timestamp}
        )
        return None

    def add_tags(self, tags, **kwargs):
        self.tags.extend(tags)
        return None

    def remove_tags(self, tags, **kwargs):
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
        return None

    def close(self, **kwargs):
        self.closed = True
        return None

    def terminate(self, **kwargs):
        self.terminated = True
        return None

    def wait_for_submission(self, **kwargs):
        return None

    def wait_for_processing(self, **kwargs):
        return None

    def get_run_url(self):
        return f'https://neptune.ai/{self.project}/runs/{self.run_id or "test-run"}'

    def get_experiment_url(self):
        return f'https://neptune.ai/{self.project}/experiments/{self.experiment_name}'

    def log_string_series(
        self, data: Dict[str, str], step: int, timestamp=None, **kwargs
    ):
        # Not implemented in mock
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


@pytest.fixture
def mock_neptune_backend():
    """Replace the saved _original_neptune_run with our mock for testing."""
    import pluto.compat.neptune

    original_saved = pluto.compat.neptune._original_neptune_run
    # Replace the saved original with our mock
    pluto.compat.neptune._original_neptune_run = MockNeptuneRun
    yield
    # Restore
    pluto.compat.neptune._original_neptune_run = original_saved


@pytest.fixture
def clean_env():
    """Clean environment variables before each test."""
    env_vars = [
        'PLUTO_PROJECT',
        'PLUTO_API_KEY',
        'PLUTO_URL_APP',
        'PLUTO_URL_API',
        'PLUTO_URL_INGEST',
        'DISABLE_NEPTUNE_LOGGING',
    ]
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    # Neptune logging is disabled by default; explicitly enable for tests
    os.environ['DISABLE_NEPTUNE_LOGGING'] = 'false'

    yield

    # Restore
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


class TestNeptuneCompatBasic:
    """Test basic Neptune API functionality is preserved."""

    def test_neptune_import_without_pluto_config(self, mock_neptune_backend, clean_env):
        """Test that Neptune works normally when PLUTO_PROJECT is not set."""
        # Don't set PLUTO_PROJECT - should fall back to Neptune-only
        from neptune_scale import Run

        run = Run(experiment_name='test-exp')
        run.log_metrics({'loss': 0.5}, step=0)
        run.close()

        assert run._neptune_run.closed
        assert len(run._neptune_run.logged_metrics) == 1
        assert run._neptune_run.logged_metrics[0]['data'] == {'loss': 0.5}

    def test_neptune_metrics_logging(self, mock_neptune_backend, clean_env):
        """Test that Neptune metrics logging works unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='metrics-test')
        run.log_metrics({'acc': 0.95, 'loss': 0.1}, step=1)
        run.log_metrics({'acc': 0.96, 'loss': 0.09}, step=2)
        run.close()

        assert len(run._neptune_run.logged_metrics) == 2
        assert run._neptune_run.logged_metrics[0]['step'] == 1
        assert run._neptune_run.logged_metrics[1]['step'] == 2

    def test_neptune_configs_logging(self, mock_neptune_backend, clean_env):
        """Test that Neptune config logging works unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='config-test')
        run.log_configs({'lr': 0.001, 'batch_size': 32})
        run.close()

        assert len(run._neptune_run.logged_configs) == 1
        assert run._neptune_run.logged_configs[0] == {'lr': 0.001, 'batch_size': 32}

    def test_neptune_tags(self, mock_neptune_backend, clean_env):
        """Test that Neptune tags work unchanged."""
        from neptune_scale import Run

        run = Run(experiment_name='tag-test')
        run.add_tags(['experiment', 'baseline'])
        run.add_tags(['v1'])
        run.remove_tags(['baseline'])
        run.close()

        assert 'experiment' in run._neptune_run.tags
        assert 'v1' in run._neptune_run.tags
        assert 'baseline' not in run._neptune_run.tags

    def test_neptune_context_manager(self, mock_neptune_backend, clean_env):
        """Test that Neptune context manager protocol works."""
        from neptune_scale import Run

        with Run(experiment_name='context-test') as run:
            run.log_metrics({'loss': 0.3}, step=0)

        assert run._neptune_run.closed


class TestNeptuneCompatDualLogging:
    """Test dual-logging to both Neptune and pluto."""

    @pytest.fixture
    def pluto_config_env(self, clean_env):
        """Set up environment for pluto dual-logging."""
        os.environ['PLUTO_PROJECT'] = 'neptune-migration-test'
        # Don't set PLUTO_API_KEY - let it fall back to keyring or fail gracefully
        yield

    def test_dual_logging_metrics_with_env_config(
        self, mock_neptune_backend, pluto_config_env, monkeypatch
    ):
        """Test that metrics are logged to both Neptune and mlop when configured."""
        # Mock mlop.init to avoid actual API calls
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='dual-log-test')
            run.log_metrics({'loss': 0.5, 'acc': 0.9}, step=0)
            run.close()

            # Verify Neptune received the data
            assert len(run._neptune_run.logged_metrics) == 1
            assert run._neptune_run.logged_metrics[0]['data'] == {
                'loss': 0.5,
                'acc': 0.9,
            }

            # Verify mlop received the data with step parameter
            mock_mlop_run.log.assert_called_with({'loss': 0.5, 'acc': 0.9}, step=0)
            mock_mlop_run.finish.assert_called_once()

    def test_log_metrics_passes_step_to_mlop(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that log_metrics correctly passes step parameter to mlop."""
        # Mock mlop.init to avoid actual API calls
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='step-test')

            # Log metrics with various step values
            run.log_metrics({'loss': 0.5}, step=0)
            run.log_metrics({'acc': 0.9}, step=100)
            run.log_metrics({'f1': 0.85}, step=200)
            run.close()

            # Verify mlop received correct step values
            assert mock_mlop_run.log.call_count == 3
            calls = mock_mlop_run.log.call_args_list

            assert calls[0] == call({'loss': 0.5}, step=0)
            assert calls[1] == call({'acc': 0.9}, step=100)
            assert calls[2] == call({'f1': 0.85}, step=200)

    def test_dual_logging_configs(self, mock_neptune_backend, pluto_config_env):
        """Test that configs are logged to both Neptune and mlop."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='config-dual-test')
            run.log_configs({'lr': 0.001, 'epochs': 100})
            run.close()

            # Verify Neptune received the config
            assert len(run._neptune_run.logged_configs) == 1

            # Verify mlop config was updated
            assert mock_mlop_run.config['lr'] == 0.001
            assert mock_mlop_run.config['epochs'] == 100

    def test_dual_logging_configs_calls_server_sync(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that log_configs calls _iface._update_config to sync to server."""
        mock_iface = mock.MagicMock()
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run._iface = mock_iface
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='config-sync-test')
            run.log_configs({'lr': 0.001, 'model': 'resnet50'})
            run.close()

            # Verify _update_config was called on the interface
            mock_iface._update_config.assert_called_once_with(
                {'lr': 0.001, 'model': 'resnet50'}
            )


class TestNeptuneCompatErrorHandling:
    """Test that Neptune never fails due to mlop errors."""

    @pytest.fixture
    def pluto_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['PLUTO_PROJECT'] = 'error-test'
        yield

    def test_neptune_works_when_mlop_init_fails(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that Neptune continues working if mlop.init() fails."""
        # Make mlop.init() raise an exception
        with mock.patch('pluto.init', side_effect=Exception('mlop service down')):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-init-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert run._neptune_run.closed
            assert len(run._neptune_run.logged_metrics) == 1

    def test_neptune_works_when_mlop_log_fails(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that Neptune continues working if mlop.log() fails."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock(side_effect=Exception('Network error'))
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-log-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert len(run._neptune_run.logged_metrics) == 1
            assert run._neptune_run.closed

    def test_neptune_works_when_mlop_finish_fails(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that Neptune closes correctly even if mlop.finish() fails."""
        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock(side_effect=Exception('Finish error'))

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            # Should not raise - Neptune should work fine
            run = Run(experiment_name='mlop-finish-fail-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()  # Should not raise

            # Verify Neptune worked
            assert run._neptune_run.closed

    def test_neptune_works_when_mlop_not_installed(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that Neptune works when mlop is not installed."""
        # Simulate mlop import failure
        with mock.patch('pluto.compat.neptune._safe_import_pluto', return_value=None):
            from neptune_scale import Run

            run = Run(experiment_name='no-mlop-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # Verify Neptune worked
            assert run._neptune_run.closed
            assert len(run._neptune_run.logged_metrics) == 1


class TestNeptuneCompatFileConversion:
    """Test file type conversion from Neptune to mlop."""

    @pytest.fixture
    def pluto_config_env(self, clean_env):
        """Set up environment for mlop dual-logging."""
        os.environ['PLUTO_PROJECT'] = 'file-conversion-test'
        yield

    def test_image_file_conversion(
        self, mock_neptune_backend, pluto_config_env, tmp_path
    ):
        """Test that Neptune File objects are converted to mlop.Image."""
        # Create a test image
        img_path = tmp_path / 'test.png'
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n')  # PNG header

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='image-test')

            # Log file with Neptune File object
            neptune_file = NeptuneFile(source=str(img_path), mime_type='image/png')
            run.assign_files({'sample_image': neptune_file})
            run.close()

            # Verify Neptune received the file
            assert len(run._neptune_run.logged_files) == 1

            # Verify mlop.log was called (file conversion is internal)
            assert mock_mlop_run.log.called

    def test_raw_png_bytes_detected_as_image(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that raw PNG bytes are detected and converted to pluto.Image."""
        import pluto

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        # Create valid PNG bytes (header + minimal data)
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            with mock.patch.object(pluto, 'Image') as mock_image_class:
                mock_image_class.return_value = mock.MagicMock()

                from neptune_scale import Run

                run = Run(experiment_name='raw-bytes-test')

                # Log raw bytes directly (like matplotlib does with BytesIO)
                run.log_files({'sample_plot': png_bytes}, step=0)
                run.close()

                # Verify pluto.Image was called with the bytes
                mock_image_class.assert_called_once_with(png_bytes)

    def test_raw_jpeg_bytes_detected_as_image(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that raw JPEG bytes are detected and converted to pluto.Image."""
        import pluto

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        # Create valid JPEG bytes (header + minimal data)
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 100

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            with mock.patch.object(pluto, 'Image') as mock_image_class:
                mock_image_class.return_value = mock.MagicMock()

                from neptune_scale import Run

                run = Run(experiment_name='jpeg-bytes-test')

                # Log raw JPEG bytes
                run.assign_files({'sample_jpeg': jpeg_bytes})
                run.close()

                # Verify pluto.Image was called with the bytes
                mock_image_class.assert_called_once_with(jpeg_bytes)

    def test_unknown_bytes_become_artifact(
        self, mock_neptune_backend, pluto_config_env
    ):
        """Test that unknown byte formats fall back to pluto.Artifact."""
        import pluto

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        # Random bytes that don't match any known format
        unknown_bytes = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            with mock.patch.object(pluto, 'Artifact') as mock_artifact_class:
                mock_artifact_class.return_value = mock.MagicMock()

                from neptune_scale import Run

                run = Run(experiment_name='unknown-bytes-test')

                # Log unknown bytes
                run.assign_files({'unknown_data': unknown_bytes})
                run.close()

                # Verify pluto.Artifact was called (not Image)
                mock_artifact_class.assert_called_once_with(unknown_bytes)

    def test_histogram_conversion(self, mock_neptune_backend, pluto_config_env):
        """Test that Neptune Histogram objects are converted to mlop.Histogram."""
        import numpy as np

        mock_mlop_run = mock.MagicMock()
        mock_mlop_run.config = {}
        mock_mlop_run.log = mock.MagicMock()
        mock_mlop_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_mlop_run):
            from neptune_scale import Run

            run = Run(experiment_name='histogram-test')

            # Create Neptune histogram
            bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
            counts = np.array([10, 20, 15])
            neptune_hist = NeptuneHistogram(bin_edges=bin_edges, counts=counts)

            run.log_histograms({'layer1/activations': neptune_hist}, step=0)
            run.close()

            # Verify Neptune received the histogram
            assert len(run._neptune_run.logged_histograms) == 1

            # Verify mlop.log was called
            assert mock_mlop_run.log.called


@pytest.mark.skip(reason='Neptune is sunset - integration tests disabled')
class TestNeptuneCompatIntegration:
    """
    Integration tests with real mlop backend (requires auth).

    These tests should be run in CI with proper mlop credentials set.

    NOTE: These tests are skipped since Neptune's sunset. They are kept
    for reference in case a similar compat layer is needed in the future.
    """

    @pytest.mark.skipif(
        not os.environ.get('PLUTO_PROJECT') or not os.environ.get('CI'),
        reason='Requires PLUTO_PROJECT env var and CI environment',
    )
    def test_real_mlop_with_mock_neptune(self, mock_neptune_backend):
        """
        Integration test with real mlop backend and mocked Neptune.

        Tests that the compat layer correctly forwards data to a real mlop
        backend while Neptune is mocked. This verifies mlop integration
        without requiring Neptune credentials.

        This test requires:
        - PLUTO_PROJECT environment variable
        - Valid mlop credentials (keyring or PLUTO_API_KEY)
        - Network access to mlop service
        """
        from neptune_scale import Run

        task_name = get_task_name()

        # Log to mock Neptune and real mlop
        run = Run(experiment_name=task_name)

        # Log various data types
        run.log_configs({'lr': 0.001, 'batch_size': 32, 'model': 'resnet50'})
        run.log_metrics({'train/loss': 0.5, 'train/acc': 0.85}, step=0)
        run.log_metrics({'train/loss': 0.3, 'train/acc': 0.92}, step=1)
        run.add_tags(['integration-test', 'ci'])

        run.close()

        # Verify Neptune received all data
        assert len(run._neptune_run.logged_configs) == 1
        assert len(run._neptune_run.logged_metrics) == 2
        assert len(run._neptune_run.tags) == 2
        assert run._neptune_run.closed

        # pluto run should also be finished
        if run._pluto_run:
            # Verify pluto was initialized successfully
            assert run._pluto_run is not None
            print('✓ Integration test passed - logged to real pluto with mock Neptune')
        else:
            pytest.skip('pluto not configured, skipping integration validation')


class TestNeptuneCompatFallbackBehavior:
    """Test various fallback scenarios."""

    def test_no_pluto_project_env_var(self, mock_neptune_backend, clean_env):
        """
        Test that monkeypatch works but doesn't init pluto when
        PLUTO_PROJECT is not set.
        """
        # No PLUTO_PROJECT set
        from neptune_scale import Run

        run = Run(experiment_name='no-project-test')
        run.log_metrics({'loss': 0.5}, step=0)
        run.close()

        # Should have no pluto run
        assert run._pluto_run is None

        # Neptune should work fine
        assert run._neptune_run.closed
        assert len(run._neptune_run.logged_metrics) == 1

    def test_mlop_project_set_but_invalid_credentials(
        self, mock_neptune_backend, clean_env
    ):
        """Test fallback when mlop project is set but credentials are invalid."""
        os.environ['PLUTO_PROJECT'] = 'test-project'
        os.environ['PLUTO_API_KEY'] = 'invalid-key-123'

        # Mock mlop.init to fail with auth error
        with mock.patch('pluto.init', side_effect=Exception('Unauthorized')):
            from neptune_scale import Run

            run = Run(experiment_name='invalid-creds-test')
            run.log_metrics({'loss': 0.5}, step=0)
            run.close()

            # pluto should have failed silently
            assert run._pluto_run is None

            # Neptune should work fine
            assert run._neptune_run.closed

    def test_disable_neptune_logging(self, mock_neptune_backend, clean_env):
        """
        Test DISABLE_NEPTUNE_LOGGING environment variable.

        When set, Neptune calls should be skipped entirely.
        """
        os.environ['DISABLE_NEPTUNE_LOGGING'] = 'true'
        os.environ['PLUTO_PROJECT'] = 'test-project'

        # Mock mlop so we can verify it was called
        with mock.patch('pluto.init') as mock_mlop_init, mock.patch('pluto.log'):
            # Create mock mlop run
            mock_run = mock.MagicMock()
            mock_mlop_init.return_value = mock_run

            from neptune_scale import Run

            run = Run(experiment_name='neptune-disabled-test')

            # Verify Neptune was NOT initialized
            assert run._neptune_run is None
            assert run._neptune_disabled is True

            # Log some metrics
            run.log_metrics({'loss': 0.5}, step=0)
            run.log_configs({'lr': 0.001})
            run.add_tags(['test', 'disabled'])

            # Verify pluto was called (dual-logging to pluto only)
            assert run._pluto_run is not None

            # Close should work without errors
            run.close()

            # get_run_url should return placeholder
            url = run.get_run_url()
            assert url == 'neptune://disabled'


class TestNeptuneCompatAPIForwarding:
    """Test that unknown Neptune API methods are forwarded correctly."""

    def test_unknown_method_forwarding(self, mock_neptune_backend, clean_env):
        """Test that unknown methods are forwarded to Neptune."""
        from neptune_scale import Run

        run = Run(experiment_name='forward-test')

        # Call Neptune-specific methods
        url = run.get_run_url()
        assert 'neptune.ai' in url

        exp_url = run.get_experiment_url()
        assert 'neptune.ai' in exp_url

        run.close()

    def test_wait_methods_work(self, mock_neptune_backend, clean_env):
        """Test that Neptune's wait methods work."""
        from neptune_scale import Run

        run = Run(experiment_name='wait-test')
        run.log_metrics({'loss': 0.5}, step=0)

        # These should not raise
        run.wait_for_submission()
        run.wait_for_processing()

        run.close()


@pytest.mark.skip(reason='Neptune is sunset - integration tests no longer runnable')
class TestNeptuneRealBackend:
    """
    Integration tests with REAL Neptune backend.

    These tests validate that the monkeypatch works correctly with the
    actual Neptune client and API. Requires Neptune credentials.

    Set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT to run these tests.

    NOTE: These tests are skipped since Neptune's sunset. They are kept
    for reference in case a similar compat layer is needed in the future.
    """

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN and NEPTUNE_PROJECT env vars',
    )
    def test_real_neptune_without_pluto(self, clean_env, tmp_path):
        """
        Test with real Neptune backend, no mlop dual-logging.

        Validates that the monkeypatch doesn't break Neptune functionality.
        """
        # Ensure mlop is NOT configured
        assert 'PLUTO_PROJECT' not in os.environ

        # Apply monkeypatch BEFORE importing Run
        import pluto.compat.neptune  # noqa: F401, I001
        import numpy as np
        from neptune_scale import Run

        task_name = get_task_name()

        # Create real Neptune run
        run = Run(experiment_name=task_name)

        # Log to real Neptune
        run.log_configs({'test': 'real-neptune', 'mode': 'neptune-only'})
        run.log_metrics({'test/metric': 1.0}, step=0)

        # Log multiple test images to Neptune at different steps
        try:
            from PIL import Image

            # Log 3 images at different steps
            for img_step in range(3):
                # Create unique test image for each step (grayscale gradient)
                intensity = 50 + (img_step * 80)  # 50, 130, 210
                test_image = np.full((64, 64, 3), intensity, dtype=np.int16)
                # Add some noise (use int16 to allow negative values, then clip)
                test_image = test_image + np.random.randint(-20, 20, (64, 64, 3))
                test_image = np.clip(test_image, 0, 255).astype(np.uint8)

                img_path = tmp_path / f'neptune_test_step_{img_step}.png'
                Image.fromarray(test_image).save(img_path)

                # Log image at this step
                run.log_files(
                    {'test/sample_image': NeptuneFile(str(img_path))},
                    step=img_step,
                )

            print('  ✓ Logged 3 images at steps 0, 1, 2')
        except ImportError:
            print('  ⚠ PIL not available, skipping image logging')

        run.add_tags(['real-neptune-test', 'neptune-only'])

        # Should have no pluto run
        assert run._pluto_run is None

        # Wait for Neptune to finish all operations before closing
        # Use verbose=False to prevent logging errors when pytest captures stdout
        run.wait_for_processing(verbose=False)

        # Close Neptune run
        run.close()

        # Verify Neptune run exists and has URL
        url = run.get_run_url()
        assert 'neptune.ai' in url
        print(f'✓ Real Neptune test passed - run URL: {url}')

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT')
        or not os.environ.get('PLUTO_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN, NEPTUNE_PROJECT, and PLUTO_PROJECT',
    )
    def test_real_neptune_with_pluto_dual_logging(self, tmp_path):
        """
        Full integration test with BOTH real Neptune and real pluto.

        This is the ultimate validation that dual-logging works in production.
        Requires both Neptune and pluto credentials.
        """
        # Apply monkeypatch BEFORE importing Run
        import pluto.compat.neptune  # noqa: F401, I001
        import numpy as np
        from neptune_scale import Run

        task_name = get_task_name()

        # Create dual-logged run
        run = Run(experiment_name=task_name)

        # Log to both systems
        run.log_configs(
            {'test': 'dual-logging', 'mode': 'production', 'framework': 'pytest'}
        )

        for step in range(3):
            run.log_metrics(
                {
                    'test/loss': 1.0 / (step + 1),
                    'test/accuracy': 0.5 + (step * 0.1),
                },
                step=step,
            )

        # Create and log multiple test images at different steps
        try:
            from PIL import Image

            # Log 3 images at different steps to test stepping functionality
            for img_step in range(3):
                # Create unique test image for each step (different colors)
                # Use step-based color to make images visually distinct
                base_color = [
                    (255, 0, 0),  # Red for step 0
                    (0, 255, 0),  # Green for step 1
                    (0, 0, 255),  # Blue for step 2
                ][img_step]

                test_image = np.zeros((64, 64, 3), dtype=np.int16)
                test_image[:, :] = base_color
                # Add some variation (use int16 to avoid overflow, then clip)
                test_image = test_image + np.random.randint(0, 50, (64, 64, 3))
                test_image = np.clip(test_image, 0, 255).astype(np.uint8)

                img_path = tmp_path / f'test_image_step_{img_step}.png'
                Image.fromarray(test_image).save(img_path)

                # Log image at this step
                run.log_files(
                    {'training/sample_image': NeptuneFile(str(img_path))},
                    step=img_step,
                )

            # Also log a static image (not associated with a step)
            static_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            static_path = tmp_path / 'static_image.png'
            Image.fromarray(static_img).save(static_path)
            run.assign_files({'summary/final_image': NeptuneFile(str(static_path))})

            print('  ✓ Logged 3 stepped images + 1 static image')
        except ImportError:
            print('  ⚠ PIL not available, skipping image logging')

        run.add_tags(['dual-logging-test', 'production', 'with-images'])

        # Both runs should be active
        assert run._neptune_run is not None
        assert run._pluto_run is not None

        # Wait for Neptune to finish all operations before closing
        # Use verbose=False to prevent logging errors when pytest captures stdout
        run.wait_for_processing(verbose=False)

        # Close both
        run.close()

        # Get URLs from both systems
        neptune_url = run.get_run_url()
        assert 'neptune.ai' in neptune_url

        print('✓ Full dual-logging test passed!')
        print(f'  Neptune: {neptune_url}')
        print('  pluto run successfully logged')

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN and NEPTUNE_PROJECT env vars',
    )
    def test_real_neptune_context_manager(self, clean_env):
        """Test context manager protocol with real Neptune."""
        # Apply monkeypatch BEFORE importing Run
        import pluto.compat.neptune  # noqa: F401, I001
        from neptune_scale import Run

        task_name = get_task_name()

        with Run(experiment_name=task_name) as run:
            run.log_metrics({'test/ctx': 1.0}, step=0)
            assert run._pluto_run is None  # No pluto configured

        # Neptune should be closed automatically
        url = run.get_run_url()
        assert 'neptune.ai' in url
        print('✓ Real Neptune context manager test passed')

    @pytest.mark.skipif(
        not os.environ.get('NEPTUNE_API_TOKEN')
        or not os.environ.get('NEPTUNE_PROJECT')
        or not os.environ.get('PLUTO_PROJECT'),
        reason='Requires NEPTUNE_API_TOKEN, NEPTUNE_PROJECT, and PLUTO_PROJECT',
    )
    def test_real_neptune_pluto_resilience(self):
        """
        Test that Neptune works even if mlop fails during the run.

        Simulates mlop service being down mid-run.
        """
        # Apply monkeypatch BEFORE importing Run
        import pluto.compat.neptune  # noqa: F401, I001
        from neptune_scale import Run

        task_name = get_task_name()

        run = Run(experiment_name=task_name)

        # Log some data - both should work
        run.log_configs({'test': 'resilience'})
        run.log_metrics({'test/metric1': 1.0}, step=0)

        # Simulate pluto failure by breaking the run object
        if run._pluto_run:
            run._pluto_run.log = mock.MagicMock(side_effect=Exception('pluto down'))

        # Neptune should still work!
        run.log_metrics({'test/metric2': 2.0}, step=1)
        run.log_metrics({'test/metric3': 3.0}, step=2)

        # Wait for Neptune to finish all operations before closing
        # Use verbose=False to prevent logging errors when pytest captures stdout
        run.wait_for_processing(verbose=False)

        # Close should still work
        run.close()

        url = run.get_run_url()
        assert 'neptune.ai' in url
        print('✓ Resilience test passed - Neptune worked despite mlop failure')


class TestNeptuneCompatSignalHandlingTransparency:
    """
    Test that the Neptune compat layer is TRANSPARENT with respect to signal handling.

    CRITICAL REQUIREMENT: When using pluto.compat.neptune, the exit/signal handling
    behavior must be IDENTICAL to using Neptune alone. This is especially important
    in multi-GPU (DDP/FSDP) settings where improper signal handling can cause hangs.

    These tests verify:
    1. Pluto's signal handlers are NOT registered when using the compat layer
    2. Pluto cleanup uses timeouts and never blocks Neptune
    3. Exit behavior is identical to Neptune-only usage
    """

    @pytest.fixture
    def pluto_config_env(self, clean_env):
        """Set up environment for pluto dual-logging."""
        os.environ['PLUTO_PROJECT'] = 'signal-test'
        yield

    def test_pluto_signal_handlers_disabled_when_neptune_enabled(
        self, mock_neptune_backend, pluto_config_env
    ):
        """
        Test that Pluto's signal handlers are DISABLED when Neptune is enabled.

        This is critical for DDP/FSDP where Neptune's signal handling should be
        preserved exactly.
        """
        import signal

        import pluto.op as pluto_op

        # Record initial signal handler state
        initial_sigint = signal.getsignal(signal.SIGINT)
        initial_sigterm = signal.getsignal(signal.SIGTERM)

        # Track if pluto's handler was registered
        pluto_op._signal_handler_registered = False

        # Mock pluto.init to verify x_disable_signal_handlers is passed
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        mock_pluto_run.finish = mock.MagicMock()

        captured_settings = {}

        def capture_init(**kwargs):
            captured_settings.update(kwargs.get('settings', {}))
            return mock_pluto_run

        with mock.patch('pluto.init', side_effect=capture_init):
            from neptune_scale import Run

            run = Run(experiment_name='signal-test')

            # Verify x_disable_signal_handlers is True when Neptune is enabled
            assert captured_settings.get('x_disable_signal_handlers') is True, (
                'x_disable_signal_handlers must be True when Neptune is enabled'
            )

            # Verify signal handlers were NOT changed by Pluto
            current_sigint = signal.getsignal(signal.SIGINT)
            current_sigterm = signal.getsignal(signal.SIGTERM)

            # The handlers should be the same as before (Pluto didn't register)
            assert current_sigint == initial_sigint, (
                'SIGINT handler should not be changed by Pluto compat layer'
            )
            assert current_sigterm == initial_sigterm, (
                'SIGTERM handler should not be changed by Pluto compat layer'
            )

            run.close()

    def test_pluto_signal_handlers_enabled_when_neptune_disabled(
        self, mock_neptune_backend, clean_env
    ):
        """
        Test that Pluto's signal handlers are ENABLED when Neptune is disabled.

        When DISABLE_NEPTUNE_LOGGING=true, Pluto is the only logging system,
        so it should handle signals for graceful shutdown.
        """
        os.environ['DISABLE_NEPTUNE_LOGGING'] = 'true'
        os.environ['PLUTO_PROJECT'] = 'signal-test'

        # Mock pluto.init to verify x_disable_signal_handlers is passed
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        mock_pluto_run.finish = mock.MagicMock()

        captured_settings = {}

        def capture_init(**kwargs):
            captured_settings.update(kwargs.get('settings', {}))
            return mock_pluto_run

        with mock.patch('pluto.init', side_effect=capture_init):
            from neptune_scale import Run

            run = Run(experiment_name='signal-test-neptune-disabled')

            # Verify Neptune is disabled
            assert run._neptune_disabled is True
            assert run._neptune_run is None

            # Verify x_disable_signal_handlers is False when Neptune is disabled
            # (Pluto should handle signals since Neptune won't)
            assert captured_settings.get('x_disable_signal_handlers') is False, (
                'x_disable_signal_handlers must be False when Neptune is disabled'
            )

            run.close()

    def test_signal_handler_setting_matches_neptune_state(
        self, mock_neptune_backend, clean_env
    ):
        """
        Test that x_disable_signal_handlers correctly reflects Neptune's state.

        - Neptune enabled (default): x_disable_signal_handlers = True
        - Neptune disabled: x_disable_signal_handlers = False
        """
        os.environ['PLUTO_PROJECT'] = 'signal-state-test'

        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        mock_pluto_run.finish = mock.MagicMock()

        # Test 1: Neptune enabled (default)
        captured_settings_enabled = {}

        def capture_init_enabled(**kwargs):
            captured_settings_enabled.update(kwargs.get('settings', {}))
            return mock_pluto_run

        with mock.patch('pluto.init', side_effect=capture_init_enabled):
            from neptune_scale import Run

            run1 = Run(experiment_name='neptune-enabled')
            assert run1._neptune_disabled is False
            assert captured_settings_enabled.get('x_disable_signal_handlers') is True
            run1.close()

        # Test 2: Neptune disabled
        os.environ['DISABLE_NEPTUNE_LOGGING'] = 'true'
        captured_settings_disabled = {}

        def capture_init_disabled(**kwargs):
            captured_settings_disabled.update(kwargs.get('settings', {}))
            return mock_pluto_run

        with mock.patch('pluto.init', side_effect=capture_init_disabled):
            from neptune_scale import Run

            run2 = Run(experiment_name='neptune-disabled')
            assert run2._neptune_disabled is True
            assert captured_settings_disabled.get('x_disable_signal_handlers') is False
            run2.close()

    def test_pluto_cleanup_has_timeout(self, mock_neptune_backend, pluto_config_env):
        """
        Test that Pluto cleanup uses a timeout to prevent blocking Neptune.
        """
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}

        # Make finish() hang forever
        import time

        def slow_finish():
            time.sleep(100)  # Would hang without timeout

        mock_pluto_run.finish = slow_finish

        with mock.patch('pluto.init', return_value=mock_pluto_run):
            from neptune_scale import Run

            run = Run(experiment_name='timeout-test')

            # close() should complete quickly due to timeout, not hang
            start = time.time()
            run.close()
            elapsed = time.time() - start

            # Should complete within timeout + buffer (not hang for 100s)
            assert elapsed < 10, (
                f'close() took {elapsed}s, should be < 10s (timeout should work)'
            )

    def test_pluto_cleanup_silent_on_error(
        self, mock_neptune_backend, pluto_config_env
    ):
        """
        Test that Pluto cleanup errors don't propagate to Neptune.
        """
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        mock_pluto_run.finish = mock.MagicMock(
            side_effect=Exception('Pluto cleanup failed catastrophically')
        )

        with mock.patch('pluto.init', return_value=mock_pluto_run):
            from neptune_scale import Run

            run = Run(experiment_name='error-test')

            # close() should NOT raise, even if Pluto fails
            run.close()

            # Neptune should be closed successfully
            assert run._neptune_run.closed

    def test_close_is_idempotent(self, mock_neptune_backend, pluto_config_env):
        """
        Test that close() can be called multiple times safely.

        This is important for signal handlers that might call close() after
        atexit already called it.
        """
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        mock_pluto_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_pluto_run):
            from neptune_scale import Run

            run = Run(experiment_name='idempotent-test')

            # Call close() multiple times
            run.close()
            run.close()
            run.close()

            # Pluto finish should only be called once (via timeout wrapper)
            # Neptune close should be called each time (normal behavior)
            assert run._neptune_run.closed

    def test_terminate_uses_shorter_timeout(
        self, mock_neptune_backend, pluto_config_env
    ):
        """
        Test that terminate() uses a shorter timeout than close().
        """
        import time

        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}

        # Track how long finish is allowed to run
        finish_start = None
        finish_end = None

        def tracked_finish():
            nonlocal finish_start, finish_end
            finish_start = time.time()
            time.sleep(10)  # Try to sleep 10s
            finish_end = time.time()

        mock_pluto_run.finish = tracked_finish

        with mock.patch('pluto.init', return_value=mock_pluto_run):
            from neptune_scale import Run

            run = Run(experiment_name='terminate-test')

            start = time.time()
            run.terminate()
            elapsed = time.time() - start

            # terminate() should use 2s timeout, so should complete < 5s
            assert elapsed < 5, f'terminate() took {elapsed}s, should use short timeout'

    def test_atexit_handler_registered(self, mock_neptune_backend, pluto_config_env):
        """
        Test that atexit handler is registered for Pluto cleanup.
        """
        import atexit

        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        mock_pluto_run.finish = mock.MagicMock()

        registered_handlers = []
        original_register = atexit.register

        def track_register(func, *args, **kwargs):
            registered_handlers.append(func)
            return original_register(func, *args, **kwargs)

        with mock.patch('atexit.register', side_effect=track_register):
            with mock.patch('pluto.init', return_value=mock_pluto_run):
                from neptune_scale import Run

                run = Run(experiment_name='atexit-test')

                # Verify an atexit handler was registered
                assert len(registered_handlers) > 0, (
                    'atexit handler should be registered'
                )

                # The handler should be the wrapper's cleanup method
                handler_names = [
                    h.__name__ for h in registered_handlers if hasattr(h, '__name__')
                ]
                assert '_atexit_cleanup_pluto' in handler_names, (
                    'atexit handler should be _atexit_cleanup_pluto'
                )

                run.close()

    def test_context_manager_uses_timeout(self, mock_neptune_backend, pluto_config_env):
        """
        Test that __exit__ also uses timeout for Pluto cleanup.
        """
        import time

        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}

        def slow_finish():
            time.sleep(100)

        mock_pluto_run.finish = slow_finish

        with mock.patch('pluto.init', return_value=mock_pluto_run):
            from neptune_scale import Run

            start = time.time()
            with Run(experiment_name='context-timeout-test') as run:
                run.log_metrics({'loss': 0.5}, step=0)
            elapsed = time.time() - start

            # Should complete quickly due to timeout
            assert elapsed < 10, (
                f'Context manager exit took {elapsed}s, should be < 10s'
            )

    def test_wrapper_closed_flag_prevents_double_cleanup(
        self, mock_neptune_backend, pluto_config_env
    ):
        """
        Test that the _closed flag prevents double cleanup of Pluto.
        """
        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        finish_call_count = 0

        def counting_finish():
            nonlocal finish_call_count
            finish_call_count += 1

        mock_pluto_run.finish = counting_finish

        with mock.patch('pluto.init', return_value=mock_pluto_run):
            from neptune_scale import Run

            run = Run(experiment_name='double-cleanup-test')

            # Simulate multiple cleanup paths (close, atexit, __exit__)
            run.close()
            run._atexit_cleanup_pluto()
            run.__exit__(None, None, None)

            # Pluto finish should only be called once
            assert finish_call_count == 1, (
                f'Pluto finish called {finish_call_count} times, should be 1'
            )


class TestNeptuneSeededRandom:
    """Test that Neptune compat layer isolates random state from user seeding."""

    def test_neptune_compat_preserves_random_state(
        self, mock_neptune_backend, clean_env
    ):
        """Random state should be preserved across Neptune Run init."""
        import random

        os.environ['PLUTO_PROJECT'] = 'test-project'

        mock_pluto_run = mock.MagicMock()
        mock_pluto_run.config = {}
        mock_pluto_run.finish = mock.MagicMock()

        with mock.patch('pluto.init', return_value=mock_pluto_run):
            from neptune_scale import Run

            random.seed(42)
            state_before = random.getstate()
            run = Run(experiment_name='state-test')
            state_after = random.getstate()
            run.close()

            assert state_before == state_after, (
                'Neptune compat layer should preserve global random state'
            )


@pytest.fixture()
def live_pluto_env():
    """Set up env for live Pluto-only tests (Neptune disabled)."""
    overrides = {
        'DISABLE_NEPTUNE_LOGGING': 'true',
        'PLUTO_PROJECT': 'testing-ci',
    }
    originals = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        os.environ[k] = v

    yield

    for k, v in originals.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.skipif(not os.environ.get('PLUTO_API_KEY'), reason='needs PLUTO_API_KEY')
class TestNeptuneCompatLiveBackend:
    """Live backend tests for Neptune compat layer run identity."""

    # Shared CI run tag so all runs from one execution can be correlated.
    # Uses GITHUB_RUN_ID in CI, falls back to a UUID for local runs.
    _ci_tag = 'ci:' + os.environ.get('GITHUB_RUN_ID', uuid.uuid4().hex[:8])
    _py_tag = f'py:{sys.version_info.major}.{sys.version_info.minor}'

    def _tag_run(self, run, test_tag):
        run._pluto_run.add_tags([test_tag, self._ci_tag, self._py_tag])

    def test_same_experiment_name_creates_separate_runs(self, live_pluto_env):
        """Two fresh Run() calls with identical experiment_name
        must create distinct server runs."""
        from pluto.compat.neptune import NeptuneRunWrapper

        tag = 'test:same_experiment_name_creates_separate_runs'
        exp_name = f'compat-fresh-{uuid.uuid4().hex[:8]}'

        run1 = NeptuneRunWrapper(project='testing-ci', experiment_name=exp_name)
        self._tag_run(run1, tag)
        id1 = run1._pluto_run.id
        run1.close()

        run2 = NeptuneRunWrapper(project='testing-ci', experiment_name=exp_name)
        self._tag_run(run2, tag)
        id2 = run2._pluto_run.id
        run2.close()

        assert id1 != id2, (
            'Fresh runs with same experiment_name must get distinct server IDs'
        )

    def test_resume_reattaches_to_existing_run(self, live_pluto_env):
        """Run(run_id=X) with resume should reattach to the same server run."""
        from pluto.compat.neptune import NeptuneRunWrapper

        tag = 'test:resume_reattaches_to_existing_run'
        exp_name = f'compat-resume-{uuid.uuid4().hex[:8]}'
        shared_run_id = f'resume-{uuid.uuid4().hex[:8]}'

        run1 = NeptuneRunWrapper(
            project='testing-ci', experiment_name=exp_name, run_id=shared_run_id
        )
        self._tag_run(run1, tag)
        id1 = run1._pluto_run.id
        run1.close()

        # Same run_id → compat layer sets resume=True → should reattach
        run2 = NeptuneRunWrapper(
            project='testing-ci', experiment_name=exp_name, run_id=shared_run_id
        )
        self._tag_run(run2, tag)
        id2 = run2._pluto_run.id
        run2.close()

        assert id1 == id2, 'Resume with same run_id must reattach to same server run'

    def test_no_run_id_means_no_external_id(self, live_pluto_env):
        """Fresh run without run_id should have no externalId on server."""
        from pluto.compat.neptune import NeptuneRunWrapper

        tag = 'test:no_run_id_means_no_external_id'
        exp_name = f'compat-no-ext-{uuid.uuid4().hex[:8]}'

        run = NeptuneRunWrapper(project='testing-ci', experiment_name=exp_name)
        self._tag_run(run, tag)
        assert run._pluto_run is not None
        run.close()
