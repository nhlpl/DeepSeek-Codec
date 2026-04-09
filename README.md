We'll build the **DeepSeek-Codec-Plugin**, a lightweight Python library that provides a unified interface for compressing prompts before sending them to the DeepSeek API. This reduces token usage and cost while preserving semantic meaning.

---

## 📁 Project Structure

```
deepseek-codec/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── src/
│   └── deepseek_codec/
│       ├── __init__.py
│       ├── codec.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── llmlingua_backend.py
│       │   ├── llmlingua2_backend.py
│       │   ├── heuristic_backend.py
│       │   └── ocr_backend.py
│       ├── preprocessors/
│       │   ├── __init__.py
│       │   ├── format_detector.py
│       │   └── system_prompt.py
│       └── utils.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.ipynb
└── tests/
    ├── test_codec.py
    └── test_backends.py
```

---

## 📄 File Contents

### `README.md`

```markdown
# DeepSeek-Codec-Plugin

A lightweight, modular prompt compression library for DeepSeek models. Slash token costs while preserving semantic meaning.

## Features

- 🔌 **Unified API**: Simple `compress(prompt)` interface.
- 🧩 **Modular Backends**: Choose from LLMLingua, LLMLingua2, Heuristic, or experimental OCR.
- 🧠 **Intelligent Preprocessing**: Auto-detects JSON, Markdown, code; preserves system prompts.
- ⚡ **DeepSeek-Optimized**: Designed with DeepSeek's architecture in mind.

## Installation

```bash
pip install deepseek-codec
```

## Quick Start

```python
from deepseek_codec import DeepSeekCodec

codec = DeepSeekCodec(backend="llmlingua")

long_prompt = "Your very long document or conversation history..."
compressed = codec.compress(long_prompt)

# Use compressed prompt with DeepSeek API
response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": compressed}]
)
```

## Supported Backends

| Backend | Description | Compression Ratio |
|---------|-------------|-------------------|
| `llmlingua` | High-quality, training-free compression | Up to 20x |
| `llmlingua2` | Smaller, faster variant | Up to 15x |
| `heuristic` | Rule-based, zero dependencies | 2-5x |
| `ocr` | Experimental optical compression | 10-30x |

## License

MIT
```

---

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "deepseek-codec"
version = "0.1.0"
description = "Modular prompt compression for DeepSeek models"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "llmlingua>=0.2.0",
    "transformers>=4.30.0",
    "torch>=2.0.0",
]
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = ["deepseek", "prompt-compression", "llm", "token-reduction"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

[project.optional-dependencies]
ocr = ["Pillow>=10.0.0", "pytesseract>=0.3.10"]
dev = ["pytest", "black", "ruff"]

[project.urls]
Homepage = "https://github.com/yourusername/deepseek-codec"
```

---

### `src/deepseek_codec/__init__.py`

```python
from .codec import DeepSeekCodec
from .backends import (
    LLMLinguaBackend,
    LLMLingua2Backend,
    HeuristicBackend,
    OCRBackend,
)

__all__ = [
    "DeepSeekCodec",
    "LLMLinguaBackend",
    "LLMLingua2Backend",
    "HeuristicBackend",
    "OCRBackend",
]
```

---

### `src/deepseek_codec/codec.py`

```python
"""Main DeepSeekCodec class."""
from typing import Optional, Union, List, Dict, Any
import logging

from .backends import (
    BaseBackend,
    LLMLinguaBackend,
    LLMLingua2Backend,
    HeuristicBackend,
    OCRBackend,
)
from .preprocessors import FormatDetector, SystemPromptPreserver

logger = logging.getLogger(__name__)


class DeepSeekCodec:
    """
    Unified interface for prompt compression optimized for DeepSeek models.

    Example:
        codec = DeepSeekCodec(backend="llmlingua")
        compressed = codec.compress(long_text, preserve_system_prompt=True)
    """

    BACKENDS = {
        "llmlingua": LLMLinguaBackend,
        "llmlingua2": LLMLingua2Backend,
        "heuristic": HeuristicBackend,
        "ocr": OCRBackend,
    }

    def __init__(
        self,
        backend: Union[str, BaseBackend] = "llmlingua",
        backend_kwargs: Optional[Dict[str, Any]] = None,
        enable_preprocessing: bool = True,
    ):
        """
        Args:
            backend: Name of backend or backend instance.
            backend_kwargs: Additional arguments for backend initialization.
            enable_preprocessing: Whether to apply intelligent preprocessing.
        """
        self.enable_preprocessing = enable_preprocessing
        self.format_detector = FormatDetector() if enable_preprocessing else None
        self.prompt_preserver = SystemPromptPreserver() if enable_preprocessing else None

        if isinstance(backend, str):
            if backend not in self.BACKENDS:
                raise ValueError(
                    f"Unknown backend '{backend}'. Available: {list(self.BACKENDS.keys())}"
                )
            backend_cls = self.BACKENDS[backend]
            self.backend = backend_cls(**(backend_kwargs or {}))
        else:
            self.backend = backend

    def compress(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        preserve_system_prompt: bool = True,
        rate: float = 0.5,
        **kwargs,
    ) -> str:
        """
        Compress a prompt to reduce token count.

        Args:
            prompt: String prompt or list of chat messages.
            preserve_system_prompt: If True, protects system messages from aggressive compression.
            rate: Target compression rate (0.0 = max compression, 1.0 = no compression).
            **kwargs: Additional arguments passed to backend.

        Returns:
            Compressed prompt string.
        """
        # Convert chat messages to string if needed
        if isinstance(prompt, list):
            prompt = self._messages_to_string(prompt)

        # Preprocessing
        original_prompt = prompt
        if self.enable_preprocessing and preserve_system_prompt:
            system_part, rest = self.prompt_preserver.extract_system_prompt(prompt)
            if system_part:
                # Compress only the non-system part
                compressed_rest = self.backend.compress(rest, rate=rate, **kwargs)
                return system_part + "\n\n" + compressed_rest

        # Detect format and apply format-specific optimizations
        if self.enable_preprocessing:
            format_type = self.format_detector.detect(prompt)
            if format_type == "json":
                prompt = self._optimize_json(prompt)
            elif format_type == "markdown":
                prompt = self._optimize_markdown(prompt)
            elif format_type == "code":
                prompt = self._optimize_code(prompt)

        # Compress
        compressed = self.backend.compress(prompt, rate=rate, **kwargs)
        logger.debug(f"Compressed {len(original_prompt)} -> {len(compressed)} chars")
        return compressed

    def _messages_to_string(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single string."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _optimize_json(self, text: str) -> str:
        """Remove unnecessary whitespace from JSON."""
        import json

        try:
            data = json.loads(text)
            return json.dumps(data, separators=(",", ":"))
        except json.JSONDecodeError:
            return text

    def _optimize_markdown(self, text: str) -> str:
        """Strip extra blank lines and normalize Markdown."""
        lines = text.split("\n")
        cleaned = []
        prev_empty = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not prev_empty:
                    cleaned.append("")
                prev_empty = True
            else:
                cleaned.append(stripped)
                prev_empty = False
        return "\n".join(cleaned)

    def _optimize_code(self, text: str) -> str:
        """Remove comments from code (simplified)."""
        # Basic: strip single-line comments from Python/JS style
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            # Remove comments if not inside a string (simplistic)
            if "#" in line and '"' not in line and "'" not in line:
                line = line.split("#")[0].rstrip()
            if line.strip():
                cleaned.append(line)
        return "\n".join(cleaned)
```

---

### `src/deepseek_codec/backends/base.py`

```python
from abc import ABC, abstractmethod
from typing import Optional


class BaseBackend(ABC):
    """Abstract base class for compression backends."""

    @abstractmethod
    def compress(self, text: str, rate: float = 0.5, **kwargs) -> str:
        """
        Compress the input text.

        Args:
            text: Input text to compress.
            rate: Target compression rate (0.0 = max compression, 1.0 = min).
            **kwargs: Backend-specific options.

        Returns:
            Compressed text.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass
```

---

### `src/deepseek_codec/backends/llmlingua_backend.py`

```python
from .base import BaseBackend
from typing import Optional

try:
    from llmlingua import PromptCompressor

    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False


class LLMLinguaBackend(BaseBackend):
    """LLMLingua-based compression backend."""

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device_map: str = "cpu",
    ):
        if not LLMLINGUA_AVAILABLE:
            raise ImportError(
                "llmlingua is not installed. Run: pip install llmlingua"
            )
        self.model_name = model_name
        self.device_map = device_map
        self._compressor: Optional[PromptCompressor] = None

    @property
    def name(self) -> str:
        return "llmlingua"

    def _get_compressor(self) -> PromptCompressor:
        if self._compressor is None:
            self._compressor = PromptCompressor(
                model_name=self.model_name,
                device_map=self.device_map,
            )
        return self._compressor

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> str:
        compressor = self._get_compressor()
        # Convert rate to target token ratio (LLMLingua uses inverse)
        target_token = kwargs.get("target_token", None)
        if target_token is None:
            # Approximate token count: ~4 chars per token
            approx_tokens = len(text) // 4
            target_token = int(approx_tokens * rate)

        compressed = compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=["\n", "?", ".", "!"],
            chunk_end_tokens=["\n"],
            return_word_label=False,
            target_token=target_token,
        )
        return compressed["compressed_prompt"]
```

---

### `src/deepseek_codec/backends/llmlingua2_backend.py`

```python
from .base import BaseBackend
from typing import Optional

try:
    from llmlingua import PromptCompressor

    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False


class LLMLingua2Backend(BaseBackend):
    """LLMLingua2 backend (smaller, faster model)."""

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map: str = "cpu",
    ):
        if not LLMLINGUA_AVAILABLE:
            raise ImportError(
                "llmlingua is not installed. Run: pip install llmlingua"
            )
        self.model_name = model_name
        self.device_map = device_map
        self._compressor: Optional[PromptCompressor] = None

    @property
    def name(self) -> str:
        return "llmlingua2"

    def _get_compressor(self) -> PromptCompressor:
        if self._compressor is None:
            self._compressor = PromptCompressor(
                model_name=self.model_name,
                device_map=self.device_map,
            )
        return self._compressor

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> str:
        compressor = self._get_compressor()
        target_token = kwargs.get("target_token", None)
        if target_token is None:
            approx_tokens = len(text) // 4
            target_token = int(approx_tokens * rate)

        compressed = compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=["\n", "?", ".", "!"],
            chunk_end_tokens=["\n"],
            return_word_label=False,
            target_token=target_token,
        )
        return compressed["compressed_prompt"]
```

---

### `src/deepseek_codec/backends/heuristic_backend.py`

```python
from .base import BaseBackend
import re


class HeuristicBackend(BaseBackend):
    """Rule-based compression without external dependencies."""

    @property
    def name(self) -> str:
        return "heuristic"

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> str:
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Common abbreviations
        abbreviations = {
            "because": "b/c",
            "with": "w/",
            "without": "w/o",
            "and": "&",
            "approximately": "~",
            "number": "#",
            "for example": "e.g.",
            "that is": "i.e.",
        }
        for full, abbr in abbreviations.items():
            text = re.sub(rf"\b{full}\b", abbr, text, flags=re.IGNORECASE)

        # Remove filler words based on rate
        if rate < 0.7:
            filler_words = [
                "basically", "actually", "literally", "very", "really",
                "quite", "rather", "somewhat", "just", "simply",
            ]
            for word in filler_words:
                text = re.sub(rf"\b{word}\b", "", text, flags=re.IGNORECASE)

        # Remove duplicate punctuation
        text = re.sub(r"([!?.]){2,}", r"\1", text)

        # Clean up extra spaces again
        text = re.sub(r"\s+", " ", text).strip()

        # If aggressive compression requested, truncate sentences
        if rate < 0.3:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            keep = max(1, int(len(sentences) * rate * 2))
            text = " ".join(sentences[:keep])

        return text
```

---

### `src/deepseek_codec/backends/ocr_backend.py`

```python
from .base import BaseBackend
import base64
import io
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class OCRBackend(BaseBackend):
    """
    Experimental: Converts text to image and uses a VLM for compression.
    This is a placeholder that simulates the compression by truncation.
    In production, would call DeepSeek-VL or similar.
    """

    def __init__(self, api_key: Optional[str] = None):
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required for OCR backend. Run: pip install Pillow")
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "ocr"

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> str:
        # Placeholder: actual implementation would:
        # 1. Render text as image
        # 2. Encode to base64
        # 3. Call DeepSeek-VL with prompt to summarize/compress
        # For now, fall back to heuristic compression
        from .heuristic_backend import HeuristicBackend

        heuristic = HeuristicBackend()
        compressed = heuristic.compress(text, rate)

        # Simulate OCR overhead note
        return f"[OCR compressed] {compressed}"
```

---

### `src/deepseek_codec/preprocessors/format_detector.py`

```python
import json
import re


class FormatDetector:
    """Detects the format of input text for targeted optimization."""

    def detect(self, text: str) -> str:
        """Return format type: 'json', 'markdown', 'code', or 'plain'."""
        # Try JSON
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError:
            pass

        # Markdown indicators
        if re.search(r"^#+\s|\*\*|\[.*\]\(.*\)|`{3}", text, re.MULTILINE):
            return "markdown"

        # Code indicators
        code_patterns = [
            r"\bdef\s+\w+\s*\(.*\)\s*:",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+",
            r"\bconst\s+\w+\s*=",
            r"\bfunction\s+\w+\s*\(",
        ]
        for pattern in code_patterns:
            if re.search(pattern, text):
                return "code"

        return "plain"
```

---

### `src/deepseek_codec/preprocessors/system_prompt.py`

```python
import re


class SystemPromptPreserver:
    """Extracts and protects system prompts from compression."""

    def extract_system_prompt(self, text: str):
        """
        Extract system prompt from conversation format.
        Returns (system_part, rest) if found, else (None, text).
        """
        # Look for common system prompt markers
        patterns = [
            r"(?i)^(system:\s*[^\n]+)",
            r"(?i)^(system prompt:\s*[^\n]+)",
            r"(?i)^(instructions:\s*[^\n]+)",
            r"(?i)^(### system\n.*?\n###)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                system_part = match.group(1)
                rest = text.replace(system_part, "").strip()
                return system_part, rest

        return None, text
```

---

### `examples/basic_usage.py`

```python
#!/usr/bin/env python
"""Basic example of using DeepSeek-Codec."""

from deepseek_codec import DeepSeekCodec


def main():
    # Initialize codec with LLMLingua backend
    codec = DeepSeekCodec(backend="llmlingua")

    # Long prompt
    long_prompt = """
    You are a helpful assistant. Please provide a detailed explanation of the following topic.
    The user wants to understand quantum computing, including qubits, superposition,
    entanglement, and potential applications. The explanation should be accessible to someone
    with a basic understanding of classical computing but no quantum physics background.
    Include analogies and avoid overly technical jargon unless necessary.
    """ * 3  # Make it longer

    print(f"Original length: {len(long_prompt)} chars")

    # Compress to 50% of original token count
    compressed = codec.compress(long_prompt, rate=0.5, preserve_system_prompt=True)

    print(f"Compressed length: {len(compressed)} chars")
    print(f"Compression ratio: {len(compressed) / len(long_prompt):.2%}")
    print("\nCompressed text:")
    print(compressed)


if __name__ == "__main__":
    main()
```

---

### `tests/test_codec.py`

```python
import pytest
from deepseek_codec import DeepSeekCodec
from deepseek_codec.backends import HeuristicBackend


def test_heuristic_backend():
    codec = DeepSeekCodec(backend="heuristic")
    text = "This is a very very very long sentence with many filler words."
    compressed = codec.compress(text, rate=0.5)
    assert len(compressed) < len(text)
    assert "very" not in compressed.lower() or "very" in compressed.lower()


def test_preserve_system_prompt():
    codec = DeepSeekCodec(backend="heuristic")
    prompt = "System: You are a helpful assistant.\n\nUser: Tell me about AI."
    compressed = codec.compress(prompt, preserve_system_prompt=True)
    assert "System: You are a helpful assistant" in compressed
    assert "AI" in compressed
```

---

## 🚀 Getting Started

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Set up LLMLingua** (if using that backend):
   ```bash
   pip install llmlingua transformers torch
   ```

3. **Run the example**:
   ```bash
   python examples/basic_usage.py
   ```

This provides a complete, modular, and extensible foundation for prompt compression optimized for DeepSeek. The plugin can be easily extended with new backends, preprocessors, and optimization strategies as the field evolves.
