import logging
from typing import Optional

from image.settings.base import (ImageSettingsSnapshot,
                                 create_default_settings_snapshot)
from image.settings.validator import ImageSettingsValidator

logger = logging.getLogger(__name__)


class ImageSettingsServer:
    """
    Convenience wrapper that combines generic SettingsServer
    with ImageSettingsValidator.

    This is optional - you can use SettingsServer directly.
    """

    def __init__(self,
                 initial_settings: Optional[ImageSettingsSnapshot] = None,
                 router_endpoint: str = "tcp://127.0.0.1:7000",
                 pub_endpoint: str = "tcp://127.0.0.1:7001"):
        """
        Create image settings server with validators pre-configured.

        This is just a thin wrapper around the generic SettingsServer.
        """
        from pycore.settings.server import (
            SettingsServer)

        settings = initial_settings or create_default_settings_snapshot()
        self.server = SettingsServer(settings, router_endpoint, pub_endpoint)

        # Register validators
        ImageSettingsValidator.register_validators(self.server.validator)

        logger.info(
            "ImageSettingsServer created (wraps generic SettingsServer)")

    async def start(self):
        """Start the underlying generic server."""
        await self.server.start()

    async def stop(self):
        """Stop the underlying generic server."""
        await self.server.stop()

    @property
    def history(self):
        """Access audit trail."""
        return self.server.history

    @property
    def validator(self):
        """Access validator (for adding custom rules)."""
        return self.server.validator
