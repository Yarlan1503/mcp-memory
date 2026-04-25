"""Tests for mcp_memory.embeddings module."""

import pytest
import numpy as np
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock, call
from pathlib import Path

sys.path.insert(0, "src")
from mcp_memory.embeddings import (
    EmbeddingEngine,
    serialize_f32,
    deserialize_f32,
    DIMENSION,
    _download_model_files,
    MODEL_REPO,
    _ROOT_FILES,
    _ONNX_FILE,
)


class TestPrepareEntityText:
    """Tests for prepare_entity_text static method."""

    def test_prepare_entity_text_no_observations(self):
        """Only name and type -> correct format with default status."""
        result = EmbeddingEngine.prepare_entity_text("TestEntity", "Testing", [])
        assert result == "TestEntity (Testing) [activo]"

    def test_prepare_entity_text_no_observations_custom_status(self):
        """Only name and type with custom status."""
        result = EmbeddingEngine.prepare_entity_text(
            "TestEntity", "Testing", [], status="archivado"
        )
        assert result == "TestEntity (Testing) [archivado]"

    def test_prepare_entity_text_with_observations(self):
        """Verifies observations are joined with ' | '."""
        result = EmbeddingEngine.prepare_entity_text("Test", "Type", ["obs1", "obs2"])
        assert "obs1" in result
        assert "obs2" in result
        assert " | " in result

    def test_prepare_entity_text_head_tail_selection(self):
        """With 15 obs, verifies head(3) + tail(7) + diverse middle selection."""
        observations = [f"obs{i}" for i in range(15)]
        result = EmbeddingEngine.prepare_entity_text("Entity", "Type", observations)
        # Should contain first 3 observations (head)
        assert "obs0" in result
        assert "obs1" in result
        assert "obs2" in result
        # Should contain last 7 observations (tail)
        assert "obs8" in result  # -7 means indices 8-14
        assert "obs14" in result
        # Total selected: 3 head + diverse middle + 7 tail
        # Header is "Entity (Type) [activo]: " with default status
        assert result.startswith("Entity (Type) [activo]: ")

    def test_prepare_entity_text_with_relations(self):
        """Verifies relations are appended as 'Rel: ...'."""
        relations = [
            {"relation_type": "knows", "target_name": "Alice"},
            {"relation_type": "works_at", "target_name": "Acme"},
        ]
        result = EmbeddingEngine.prepare_entity_text(
            "Bob", "Person", ["obs1"], relations=relations
        )
        assert "Rel:" in result
        assert "knows" in result
        assert "Alice" in result
        assert "works_at" in result
        assert "Acme" in result


class TestSerializeF32:
    """Tests for serialize_f32 / deserialize_f32 binary helpers."""

    def test_serialize_f32(self):
        """Vector numpy -> bytes, verifies length = 384*4 = 1536 bytes."""
        vector = np.random.randn(DIMENSION).astype(np.float32)
        serialized = serialize_f32(vector)
        assert len(serialized) == DIMENSION * 4 == 1536

    def test_deserialize_f32(self):
        """Roundtrip: serialize -> deserialize -> values equal."""
        original = np.random.randn(DIMENSION).astype(np.float32)
        serialized = serialize_f32(original)
        deserialized = deserialize_f32(serialized)
        np.testing.assert_array_almost_equal(original, deserialized)


class TestEmbeddingEngineSingleton:
    """Tests for EmbeddingEngine singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingEngine.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingEngine.reset()

    def test_embedding_engine_singleton(self):
        """Two calls to get_instance() return the same object."""
        engine1 = EmbeddingEngine.get_instance()
        engine2 = EmbeddingEngine.get_instance()
        assert engine1 is engine2

    def test_embedding_engine_reset(self):
        """reset() clears the singleton, next get_instance() creates new."""
        engine1 = EmbeddingEngine.get_instance()
        EmbeddingEngine.reset()
        engine2 = EmbeddingEngine.get_instance()
        assert engine1 is not engine2

    def test_embedding_engine_not_available_when_no_model(self):
        """When model files don't exist AND download fails, available=False."""
        with patch("mcp_memory.embeddings.MODEL_DIR", Path("/nonexistent/path")):
            with patch("mcp_memory.embeddings.Path.exists", return_value=False):
                with patch(
                    "mcp_memory.embeddings._download_model_files",
                    return_value=False,
                ) as mock_dl:
                    EmbeddingEngine.reset()
                    engine = EmbeddingEngine.get_instance()
                    mock_dl.assert_called_once()
                    assert engine.available is False


class TestDownloadModelFiles:
    """Unit tests for _download_model_files function."""

    def test_success_when_all_files_download(self, tmp_path):
        """Returns True when all files download successfully."""
        # Create a real file for shutil.copy2 to copy
        fake_onnx_src = tmp_path / "hf_cached_onnx"
        fake_onnx_src.write_text("fake-onnx-data")

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = str(fake_onnx_src)
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            result = _download_model_files(tmp_path)
        assert result is True
        # hf_hub_download called for each tokenizer file + ONNX
        assert mock_hf.hf_hub_download.call_count == len(_ROOT_FILES) + 1
        # Directory was created and model.onnx was copied
        assert (tmp_path / "model.onnx").exists()

    def test_returns_false_when_huggingface_hub_missing(self, tmp_path):
        """Returns False when huggingface_hub is not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("No module named 'huggingface_hub'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _download_model_files(tmp_path)
        assert result is False

    def test_returns_false_on_partial_failure(self, tmp_path):
        """Returns False when some files fail to download."""
        mock_hf = MagicMock()
        # First call succeeds, subsequent calls fail
        mock_hf.hf_hub_download.side_effect = [
            str(tmp_path / "tok1"),  # tokenizer.json ok
            Exception("network error"),  # tokenizer_config.json fails
            Exception("network error"),  # special_tokens_map.json fails
            Exception("network error"),  # model.onnx fails
        ]
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            result = _download_model_files(tmp_path)
        assert result is False

    def test_onnx_file_copied_to_correct_destination(self, tmp_path):
        """ONNX file is copied from HF cache to model_dir/model.onnx."""
        # Create a fake cached ONNX file
        fake_cache_dir = tmp_path / "hf_cache"
        fake_cache_dir.mkdir()
        fake_onnx = fake_cache_dir / "model.onnx"
        fake_onnx.write_text("fake-onnx-data")

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = str(fake_onnx)
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            result = _download_model_files(tmp_path)
        assert result is True
        assert (tmp_path / "model.onnx").exists()

    def test_hf_hub_download_called_with_correct_repo_and_filenames(self, tmp_path):
        """Verifies correct repo_id and filenames are passed to HF Hub."""
        fake_onnx_src = tmp_path / "hf_cached_onnx"
        fake_onnx_src.write_text("fake-onnx-data")

        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = str(fake_onnx_src)
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            _download_model_files(tmp_path)

        calls = mock_hf.hf_hub_download.call_args_list
        # Check tokenizer file calls
        for i, filename in enumerate(_ROOT_FILES):
            assert calls[i].kwargs["repo_id"] == MODEL_REPO
            assert calls[i].kwargs["filename"] == filename
            assert calls[i].kwargs["local_dir"] == tmp_path
        # Check ONNX call (last call)
        onnx_call = calls[-1]
        assert onnx_call.kwargs["repo_id"] == MODEL_REPO
        assert onnx_call.kwargs["filename"] == _ONNX_FILE


class TestAutoDownload:
    """Integration tests for auto-download behavior in EmbeddingEngine."""

    def setup_method(self):
        EmbeddingEngine.reset()

    def teardown_method(self):
        EmbeddingEngine.reset()

    def test_download_triggered_when_model_missing(self):
        """_download_model_files is called when model files don't exist."""
        with patch("mcp_memory.embeddings.MODEL_DIR", Path("/nonexistent/path")):
            with patch("mcp_memory.embeddings.Path.exists", return_value=False):
                with patch(
                    "mcp_memory.embeddings._download_model_files",
                    return_value=False,
                ) as mock_dl:
                    EmbeddingEngine()
                    mock_dl.assert_called_once_with(Path("/nonexistent/path"))

    def test_download_not_called_when_model_exists(self):
        """_download_model_files is NOT called when model files exist."""
        mock_ort = MagicMock()
        mock_sess = MagicMock()
        mock_sess.get_inputs.return_value = []
        mock_ort.InferenceSession.return_value = mock_sess
        mock_ort.SessionOptions.return_value.graph_optimization_level = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

        mock_tokenizer_instance = MagicMock()
        mock_tokenizers = MagicMock()
        mock_tokenizers.Tokenizer.from_file.return_value = mock_tokenizer_instance

        with patch.dict(
            "sys.modules",
            {"onnxruntime": mock_ort, "tokenizers": mock_tokenizers},
        ):
            with patch("mcp_memory.embeddings.MODEL_DIR", Path("/fake/models")):
                with patch("mcp_memory.embeddings.Path.exists", return_value=True):
                    with patch(
                        "mcp_memory.embeddings._download_model_files"
                    ) as mock_dl:
                        engine = EmbeddingEngine()
                        mock_dl.assert_not_called()

    def test_graceful_degradation_on_download_failure(self):
        """When download fails, available=False but no exception raised."""
        with patch("mcp_memory.embeddings.MODEL_DIR", Path("/nonexistent/path")):
            with patch("mcp_memory.embeddings.Path.exists", return_value=False):
                with patch(
                    "mcp_memory.embeddings._download_model_files",
                    return_value=False,
                ):
                    engine = EmbeddingEngine()
                    assert engine.available is False
                    # Non-semantic operations should not crash
                    assert engine.dimension == DIMENSION
                    text = EmbeddingEngine.prepare_entity_text("Test", "Type", ["obs1"])
                    assert "obs1" in text

    def test_successful_download_then_load_attempted(self):
        """After successful download, the engine attempts to load the model."""
        mock_ort = MagicMock()
        mock_sess = MagicMock()
        mock_sess.get_inputs.return_value = []
        mock_ort.InferenceSession.return_value = mock_sess
        mock_ort.SessionOptions.return_value.graph_optimization_level = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

        mock_tokenizer_instance = MagicMock()
        mock_tokenizers = MagicMock()
        mock_tokenizers.Tokenizer.from_file.return_value = mock_tokenizer_instance

        # Path.exists=False triggers download; _download_model_files=True
        # makes the code fall through to the ONNX loading section.
        with patch.dict(
            "sys.modules",
            {"onnxruntime": mock_ort, "tokenizers": mock_tokenizers},
        ):
            with patch("mcp_memory.embeddings.MODEL_DIR", Path("/fake/models")):
                with patch("mcp_memory.embeddings.Path.exists", return_value=False):
                    with patch(
                        "mcp_memory.embeddings._download_model_files",
                        return_value=True,
                    ) as mock_dl:
                        engine = EmbeddingEngine()
                        mock_dl.assert_called_once()
                        # Verify ONNX session was created (model load attempted)
                        mock_ort.InferenceSession.assert_called_once()

    def test_encode_raises_when_download_failed(self):
        """encode() raises RuntimeError when model unavailable after failed download."""
        with patch("mcp_memory.embeddings.MODEL_DIR", Path("/nonexistent/path")):
            with patch("mcp_memory.embeddings.Path.exists", return_value=False):
                with patch(
                    "mcp_memory.embeddings._download_model_files",
                    return_value=False,
                ):
                    engine = EmbeddingEngine()
                    with pytest.raises(RuntimeError, match="not available"):
                        engine.encode(["test"])


class TestEncodeIntegration:
    """Integration tests for encode() with real ONNX model."""

    @pytest.fixture(scope="class", autouse=True)
    def reset_singleton(self):
        """Reset singleton once before all tests in this class."""
        EmbeddingEngine.reset()
        yield
        EmbeddingEngine.reset()

    def test_encode_returns_correct_shape(self):
        """encode() returns arrays with expected shapes."""
        engine = EmbeddingEngine.get_instance()
        if not engine.available:
            pytest.skip("Embedding model not available")

        # Single text
        result = engine.encode(["Hello world"])
        assert result.shape == (1, DIMENSION)
        assert result.dtype == np.float32

        # Multiple texts
        result = engine.encode(["One", "Two", "Three"])
        assert result.shape == (3, DIMENSION)

        # Empty list
        result = engine.encode([])
        assert result.shape == (0, DIMENSION)

    def test_dynamic_padding_preserves_outputs(self):
        """Dynamic padding produces the same embeddings as fixed padding (512)."""
        engine = EmbeddingEngine.get_instance()
        if not engine.available:
            pytest.skip("Embedding model not available")

        texts = [
            "Hola mundo",
            "El rápido zorro marrón salta sobre el perro perezoso cada mañana.",
            "La inteligencia artificial está transformando rápidamente múltiples industrias.",
            "Python es genial.",
            "12345",
            "¿Cómo estás? ¡Bien!",
            "a",
            "Un texto medianamente largo que debería tener alrededor de treinta o cuarenta tokens en total.",
            "Embeddings semánticos para búsqueda vectorial en sqlite-vec.",
            "Último texto de prueba con acentos: áéíóúüñ.",
        ]

        # Current implementation (dynamic padding)
        dynamic = engine.encode(texts)
        assert dynamic.shape == (len(texts), DIMENSION)

        # Simulate fixed padding (512)
        engine._tokenizer.enable_padding(length=512)
        fixed = engine.encode(texts)
        assert fixed.shape == (len(texts), DIMENSION)

        # All outputs must be within atol=1e-4
        assert np.allclose(dynamic, fixed, atol=1e-4)


class _FakeEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _BorrowTrackingTokenizer:
    """Tokenizer double that fails if two encode() calls overlap."""

    def __init__(self):
        self._active = 0
        self._lock = threading.Lock()
        self.max_active = 0

    def _enter(self):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active > 1:
                raise RuntimeError("Already borrowed")

    def _exit(self):
        with self._lock:
            self._active -= 1

    def no_padding(self):
        pass

    def enable_padding(self, length: int):
        self.padding_length = length

    def encode_batch(self, texts: list[str]):
        self._enter()
        try:
            time.sleep(0.005)
            max_len = getattr(self, "padding_length", None)
            encodings = []
            for text in texts:
                length = max(1, min(len(text.split()), max_len or 512))
                encodings.append(_FakeEncoding(list(range(1, length + 1))))
            return encodings
        finally:
            self._exit()


class _BorrowTrackingSession:
    """ONNX session double that fails if session.run() overlaps."""

    def __init__(self):
        self._active = 0
        self._lock = threading.Lock()
        self.max_active = 0

    def run(self, _output_names, feed):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active > 1:
                raise RuntimeError("Already borrowed")
        try:
            time.sleep(0.005)
            input_ids = feed["input_ids"]
            batch, seq_len = input_ids.shape
            output = np.ones((batch, seq_len, DIMENSION), dtype=np.float32)
            return [output]
        finally:
            with self._lock:
                self._active -= 1


def test_encode_serializes_tokenizer_and_onnx_access():
    """Concurrent encode() calls share tokenizer/session without borrow races."""
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    tokenizer = _BorrowTrackingTokenizer()
    session = _BorrowTrackingSession()
    engine._available = True
    engine._tokenizer = tokenizer
    engine._session = session
    engine._input_names = ["input_ids", "attention_mask"]
    engine._encode_lock = threading.Lock()

    texts = [[f"texto {i} con palabras"] for i in range(20)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(engine.encode, texts))

    assert all(result.shape == (1, DIMENSION) for result in results)
    assert tokenizer.max_active == 1
    assert session.max_active == 1
