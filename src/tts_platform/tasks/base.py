from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

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
        pass

    async def stream(self, engine: Any, request: RequestT) -> AsyncIterator[Any]:
        """Execute the task (streaming) - optional."""
        # Yield nothing by default to make it an async generator
        if False:
            yield
        raise NotImplementedError("Streaming not implemented for this task")
