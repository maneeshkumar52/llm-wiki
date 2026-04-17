"""No-op telemetry implementation for Chroma."""

from __future__ import annotations

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpTelemetryClient(ProductTelemetryClient):
    """Disable Chroma product telemetry event emission."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        """Intentionally discard telemetry events."""
        return None