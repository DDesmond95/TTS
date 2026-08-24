"""Base class for all inference tasks in OmniVoice Studio."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


class Task[RequestT: BaseModel, ResultT](ABC):
    """
    Base class for all inference tasks.
    """

    @abstractmethod
    def validate(self, request: RequestT) -> RequestT:
        """Validate the request."""
        return request

    @abstractmethod
    async def run(self, engine: Any, request: RequestT) -> ResultT:
        """Execute the task (non-streaming)."""

    async def stream(self, _engine: Any, _request: RequestT) -> AsyncIterator[Any]:
        """Execute the task (streaming) - optional."""
        # Empty loop to satisfy generator requirement
        for _ in range(0):
            yield
        raise NotImplementedError("Streaming not implemented for this task")

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Splits text into sentences for streaming."""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _prepare_run(
        engine: Any, name: str, params: dict[str, Any]
    ) -> tuple[str, Path]:
        """Helper to create a run directory and write parameters."""
        run_id, run_dir = engine.outputs.new_run_dir(name)
        engine.outputs.write_params(run_dir, params)
        return run_id, run_dir

    def _get_stream_sentences(self, text: str | list[str]) -> list[str]:
        """Helper for streaming preamble: returns sentences if text is a string."""
        if isinstance(text, list):
            return []
        return self.split_sentences(text)

    async def _stream_loop(
        self,
        engine: Any,
        *,
        sentences: list[str],
        model_obj: Any,
        method_name: str,
        base_params: dict[str, Any],
        gen_params: dict[str, Any],
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Generic loop for streaming sentence chunks."""
        method = getattr(model_obj, method_name)
        for sent in sentences:
            async with engine.sem:
                wavs, sr = await asyncio.to_thread(
                    method, text=sent, **base_params, **gen_params
                )
                if wavs:
                    yield np.asarray(wavs[0]), int(sr)
