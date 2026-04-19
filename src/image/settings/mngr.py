"""
ImageSettingsManager - Bridge between PyQt6 ImageSettings and ImageSettingsServer
==================================================================================

Manages the complete lifecycle of distributed image settings:
- Starts/stops ImageSettingsServer automatically
- Syncs PyQt6 ImageSettings with server state
- Bidirectional updates: Qt → Server → All Clients
- Thread-safe and multiprocessing-safe
- Automatic reconnection and health monitoring
- Graceful degradation when server is unavailable

Architecture:
    ┌─────────────────┐
    │  Qt Main Thread │
    │  ImageSettings  │◄──┐
    └────────┬────────┘   │
             │            │ (Signal)
    ┌────────▼────────────┴───┐
    │ ImageSettingsManager    │
    │  - Async Bridge         │
    │  - Health Monitor       │
    │  - Auto Reconnect       │
    └────┬──────────────┬─────┘
         │              │
      Client         Subscriber
         │              │
    ┌────▼──────────────▼─────┐
    │  ImageSettingsServer    │
    │  (Async ZMQ)            │
    └─────────────────────────┘
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from enum import unique, StrEnum
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, pyqtSlot

from pycore.settings.server import SettingsServer
from pycore.settings.history import ChangeNotification
from pycore.settings.client import SettingsClient
from pycore.settings.subscriber import SettingsSubscriber
from qtcore.monitor import HealthMonitor

from image.settings.base import (
    ImageSettings
)
from image.settings.validator import (
    ImageSettingsValidator)

logger = logging.getLogger(__name__)

@unique
class ManagerState(StrEnum):
    """Manager lifecycle states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"  # Running without server
    STOPPING = "stopping"
    FAILED = "failed"


class ImageSettingsManager(QObject):
    """
    Manages ImageSettings lifecycle and synchronization with server.

    Features:
    - Automatic server lifecycle (start/stop)
    - Bidirectional sync: Qt ↔ Server
    - Thread-safe local and remote updates
    - Automatic reconnection on failure
    - Health monitoring
    - Graceful degradation

    Usage:
        # In Qt main thread
        qt_settings = ImageSettings()
        manager = ImageSettingsManager(qt_settings)
        manager.start()

        # Settings auto-sync
        qt_settings.update_setting('zoom', 2.5)  # → Server → All clients

        # Cleanup
        manager.stop()
    """

    # Qt Signals
    state_changed = pyqtSignal(ManagerState)
    server_connected = pyqtSignal()
    server_disconnected = pyqtSignal()
    sync_error = pyqtSignal(str)

    def __init__(self,
                 settings: ImageSettings,
                 router_endpoint: str = "tcp://127.0.0.1:7000",
                 pub_endpoint: str = "tcp://127.0.0.1:7001",
                 auto_start_server: bool = True,
                 parent: Optional[QObject] = None):
        """
        Initialize manager.

        Args:
            settings: PyQt6 ImageSettings instance to manage
            router_endpoint: Server ROUTER endpoint
            pub_endpoint: Server PUB endpoint
            auto_start_server: If True, start server if not running
            parent: Qt parent object
        """
        super().__init__(parent)

        self.qt_settings = settings
        self.router_endpoint = router_endpoint
        self.pub_endpoint = pub_endpoint
        self.auto_start_server = auto_start_server

        # State
        self._state = ManagerState.STOPPED
        self._server: Optional[SettingsServer] = None
        self._client: Optional[SettingsClient] = None
        self._subscriber: Optional[SettingsSubscriber] = None

        # Async runtime
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Health monitoring
        self.health = HealthMonitor()
        self._health_timer: Optional[QTimer] = None

        # Sync state
        self._syncing = False  # Prevent update loops
        self._last_server_sequence = -1

        logger.info(f"ImageSettingsManager created for {router_endpoint}")

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def start(self):
        """Start manager and connect to server"""
        if self._state != ManagerState.STOPPED:
            logger.warning(f"Cannot start: already in state {self._state}")
            return

        self._set_state(ManagerState.STARTING)

        # Start async event loop in background thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_async_loop,
                                        daemon=True)
        self._thread.start()

        # Connect Qt signals
        self.qt_settings.changed.connect(self._on_qt_settings_changed)

        # Start health monitoring
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._health_timer.start(2000)  # Check every 2s

        logger.info("ImageSettingsManager started")

    def stop(self):
        """Stop manager and cleanup resources"""
        if self._state == ManagerState.STOPPED:
            return

        self._set_state(ManagerState.STOPPING)

        # Stop health monitoring
        if self._health_timer:
            self._health_timer.stop()
            self._health_timer = None

        # Signal async loop to stop
        self._stop_event.set()

        # Wait for thread with timeout
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Async thread did not stop cleanly")

        self._set_state(ManagerState.STOPPED)
        logger.info("ImageSettingsManager stopped")

    def _run_async_loop(self):
        """Run async event loop in background thread"""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Run initialization
            self._loop.run_until_complete(self._async_start())

            # Run until stopped
            while not self._stop_event.is_set():
                self._loop.run_until_complete(asyncio.sleep(0.01))

            # Cleanup
            self._loop.run_until_complete(self._async_stop())

        except Exception as e:
            logger.error(f"Async loop error: {e}", exc_info=True)
            self._set_state(ManagerState.FAILED)
        finally:
            if self._loop:
                self._loop.close()

    async def _async_start(self):
        """Async initialization"""
        try:
            logger.critical("Starting settings server...")
            # Start server if needed
            if self.auto_start_server:
                await self._ensure_server_running()

            # Connect client
            await self._connect_client()

            # Start subscriber
            await self._start_subscriber()

            self._set_state(ManagerState.RUNNING)
            self.server_connected.emit()
            logger.critical("Started")

        except Exception as e:
            logger.error(f"Failed to start: {e}")
            self._set_state(ManagerState.DEGRADED)
            self.sync_error.emit(str(e))

    async def _async_stop(self):
        """Async cleanup"""
        try:
            # Stop subscriber
            if self._subscriber:
                await self._subscriber.stop()
                self._subscriber = None

            # Stop client
            if self._client:
                await self._client.stop()
                self._client = None

            # Stop server if we started it
            if self._server:
                await self._server.stop()
                self._server = None

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    # =========================================================================
    # Server Management
    # =========================================================================

    async def _ensure_server_running(self):
        """Start server if not already running"""
        # Try to connect first
        test_client = SettingsClient(self.router_endpoint, timeout=1.0)
        try:
            await test_client.start()
            await test_client.get()
            await test_client.stop()
            logger.info("Server already running")
            return
        except Exception:
            logger.info("Server not running, starting...")
            await test_client.stop()

        # Start new server
        initial_settings = self.qt_settings.get_copy()
        self._server = SettingsServer(
            initial_settings,
            self.router_endpoint,
            self.pub_endpoint
        )

        # Register validators
        ImageSettingsValidator.register_validators(self._server.validator)

        await self._server.start()
        await asyncio.sleep(0.1)  # Let server bind

        logger.info("Server started")

    async def _connect_client(self):
        """Connect to server"""
        self._client = SettingsClient(self.router_endpoint, timeout=2.0)
        await self._client.start()
        logger.info("Client connected")

    async def _start_subscriber(self):
        """Start subscriber for server updates"""
        self._subscriber = SettingsSubscriber(
            self.pub_endpoint,
            callback=self._on_server_update
        )
        await self._subscriber.start()
        self._subscriber.subscribe()  # Subscribe to all changes
        logger.info("Subscriber started")

    # =========================================================================
    # Synchronization
    # =========================================================================

    @pyqtSlot()
    def _on_qt_settings_changed(self):
        """Qt settings changed → Update server"""
        if self._syncing:
            print("already syncing")
            return  # Prevent update loop

        if self._state != ManagerState.RUNNING:
            snapshot = self.qt_settings.get_copy()
            logger.critical(f"Skipping sync: manager not running: {snapshot}")
            return

        # Schedule async update
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._sync_to_server(),
                self._loop
            )

    async def _sync_to_server(self):
        """Sync Qt settings to server"""
        if not self._client or self._syncing:
            print("returning, no client or sync enabled")
            return

        try:
            # Get current Qt state
            snapshot = self.qt_settings.get_copy()

            # Get server state
            server_snapshot = await self._client.get()

            # Find differences and update
            updates = {}
            for field in snapshot._fields:
                qt_value = getattr(snapshot, field)
                server_value = getattr(server_snapshot, field)

                if qt_value != server_value:
                    updates[field] = qt_value

            if updates:
                logger.critical(f"Syncing to server: {updates}")
                for field, value in updates.items():
                    await self._client.set(
                        field,
                        value,
                        changed_by='Qt-Local',
                        reason='Qt settings changed'
                    )

                self.health.record_success()

        except Exception as e:
            logger.error(f"Sync to server failed: {e}")
            self.health.record_failure()
            self.sync_error.emit(f"Sync to server: {e}")

    def _on_server_update(self, notification: ChangeNotification):
        """Server update received → Update Qt settings"""
        if self._syncing:
            return

        try:
            self._syncing = True

            meta = notification.metadata
            field_name = meta.field_path  # or field_name depending on your impl
            new_value = meta.new_value

            # Update Qt settings
            success = self.qt_settings.update_setting(field_name, new_value)

            if success:
                logger.debug(
                    f"Qt updated from server: {field_name}={new_value}")

            self.health.record_success()

        except Exception as e:
            logger.error(f"Failed to apply server update: {e}")
            self.health.record_failure()
        finally:
            self._syncing = False

    async def _sync_from_server(self):
        """Initial sync: Server → Qt"""
        if not self._client:
            return

        try:
            self._syncing = True

            # Get server snapshot
            snapshot = await self._client.get()

            # Update all Qt settings
            for field in snapshot._fields:
                value = getattr(snapshot, field)
                self.qt_settings.update_setting(field, value)

            logger.info("Initial sync from server complete")
            self.health.record_success()

        except Exception as e:
            logger.error(f"Initial sync failed: {e}")
            self.health.record_failure()
        finally:
            self._syncing = False

    # =========================================================================
    # Health Monitoring & Reconnection
    # =========================================================================

    def _check_health(self):
        """Periodic health check (called from Qt timer)"""
        if self._state != ManagerState.RUNNING:
            return

        if not self.health.is_healthy():
            logger.warning("Server appears unhealthy")

            if self.health.should_reconnect():
                logger.warning("Triggering reconnection")
                self._trigger_reconnect()

    def _trigger_reconnect(self):
        """Trigger reconnection in async thread"""
        if self._loop and self._state == ManagerState.RUNNING:
            asyncio.run_coroutine_threadsafe(
                self._reconnect(),
                self._loop
            )

    async def _reconnect(self):
        """Attempt to reconnect to server"""
        if self._state != ManagerState.RUNNING:
            return

        self._set_state(ManagerState.RECONNECTING)
        self.server_disconnected.emit()

        try:
            # Stop existing connections
            if self._subscriber:
                await self._subscriber.stop()
                self._subscriber = None

            if self._client:
                await self._client.stop()
                self._client = None

            # Wait a bit
            await asyncio.sleep(1.0)

            # Reconnect
            await self._connect_client()
            await self._start_subscriber()
            await self._sync_from_server()

            self.health.reset()
            self._set_state(ManagerState.RUNNING)
            self.server_connected.emit()

            logger.info("Reconnection successful")

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self._set_state(ManagerState.DEGRADED)
            self.sync_error.emit(f"Reconnection: {e}")

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_state(self, state: ManagerState):
        """Update manager state"""
        if self._state != state:
            old_state = self._state
            self._state = state
            logger.info(f"State: {old_state.value} → {state.value}")
            self.state_changed.emit(state)

    @property
    def state(self) -> ManagerState:
        """Current manager state"""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server"""
        return self._state == ManagerState.RUNNING

    # =========================================================================
    # Manual Control
    # =========================================================================

    def force_sync_to_server(self):
        """Manually trigger sync to server"""
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(
                self._sync_to_server(),
                self._loop
            )

    def force_sync_from_server(self):
        """Manually trigger sync from server"""
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(
                self._sync_from_server(),
                self._loop
            )

    def get_server_stats(self) -> dict:
        """Get server statistics"""
        return {
            'state': self._state.value,
            'connected': self.is_connected,
            'health_failures': self.health.consecutive_failures,
            'last_success': self.health.last_success,
        }


# =============================================================================
# Context Manager for Easy Usage
# =============================================================================
@asynccontextmanager
async def image_settings_manager(
        qt_settings: ImageSettings,
        router_endpoint: str = "tcp://127.0.0.1:7000",
        pub_endpoint: str = "tcp://127.0.0.1:7001"
):
    """
    Context manager for ImageSettingsManager.

    Usage:
        async with image_settings_manager(qt_settings) as manager:
            # Use manager
            pass
        # Auto cleanup
    """
    manager = ImageSettingsManager(
        qt_settings,
        router_endpoint,
        pub_endpoint
    )

    try:
        manager.start()
        # Wait for connection
        timeout = 5.0
        start = time.time()
        while not manager.is_connected and time.time() - start < timeout:
            await asyncio.sleep(0.1)

        if not manager.is_connected:
            raise TimeoutError("Failed to connect to server")

        yield manager

    finally:
        manager.stop()


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication


    def main():
        """Demo application"""
        app = QApplication(sys.argv)

        # Create Qt settings
        qt_settings = ImageSettings()

        # Create manager
        manager = ImageSettingsManager(qt_settings)

        # Connect to signals
        manager.state_changed.connect(
            lambda s: print(f"State: {s.value}")
        )
        manager.server_connected.connect(
            lambda: print("✓ Server connected")
        )
        manager.server_disconnected.connect(
            lambda: print("✗ Server disconnected")
        )
        manager.sync_error.connect(
            lambda e: print(f"⚠ Sync error: {e}")
        )

        # Start manager
        print("Starting manager...")
        manager.start()

        # Simulate Qt changes
        timer = QTimer()
        counter = [0]

        def update_settings():
            counter[0] += 1
            if counter[0] % 2 == 0:
                qt_settings.update_setting('zoom', 1.0 + counter[0] * 0.5)
            else:
                qt_settings.update_setting('brightness', counter[0] * 0.1)

            if counter[0] >= 10:
                timer.stop()
                print("\nStopping manager...")
                manager.stop()
                app.quit()

        timer.timeout.connect(update_settings)
        timer.start(1000)

        print("Running... (Qt settings will update periodically)")
        sys.exit(app.exec())


    main()
