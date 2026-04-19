import multiprocessing
import struct
import sys
from multiprocessing import shared_memory
from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt6.QtCore import QThread

import qt6_utils.image.utils.data


# -----------------------------------------------------------------------------
# 1. MOCKS (To simulate dependencies without full library)
# -----------------------------------------------------------------------------

class ImageMetadata:
    def __init__(self, timestamp=0.0, frame_id=0.0, shape=(100, 100),
                 dtype_str='uint8', processing_time_ms=0.0):
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.shape = shape
        self.dtype_str = dtype_str
        self.processing_time_ms = processing_time_ms


class ProcessingConfig:
    def __init__(self):
        self.colormap_enabled = False
        self.colormap_lut = None

    @classmethod
    def from_settings(cls, s): return cls()


# Mock Ring Buffer with Jitter Padding Logic
class MockSharedMemoryRingBuffer:
    def __init__(self, buffer_count=8):
        self.buffers = []

    def alloc_buffer(self, size_bytes):
        # Alloc 1.5x to simulate the Jitter padding logic in the real class
        alloc_size = int(size_bytes * 1.5)
        try:
            shm = shared_memory.SharedMemory(create=True, size=alloc_size)
        except FileExistsError:
            shm = shared_memory.SharedMemory(create=False, size=alloc_size)

        self.buffers.append(shm)
        # Return name and FULL view
        return shm.name, np.ndarray(alloc_size, dtype=np.uint8, buffer=shm.buf)

    def get_view(self, name):
        for buf in self.buffers:
            if buf.name == name:
                return np.ndarray(buf.size, dtype=np.uint8, buffer=buf.buf)
        # Try attach
        try:
            shm = shared_memory.SharedMemory(name=name)
            self.buffers.append(shm)
            return np.ndarray(shm.size, dtype=np.uint8, buffer=shm.buf)
        except:
            return None

    def cleanup(self):
        for buf in self.buffers:
            try:
                buf.close()
                buf.unlink()
            except:
                pass
        self.buffers = []


# -- PATCH MODULES --
mock_cp_core = MagicMock()
mock_cp_core.shm_ring.SharedMemoryRingBuffer = MockSharedMemoryRingBuffer
mock_cp_core.shm_ring.cleanup_shm_cache = lambda x, unlink: None
mock_cp_core.cpu_utils.set_high_priority = lambda x: None

mock_qt_utils = MagicMock()
mock_qt_utils.image.pipeline.metadata.FrameStats = ImageMetadata
mock_qt_utils.image.pipeline.config.ProcessingConfig = ProcessingConfig
qt6_utils.image.utils.data.ensure_contiguity = lambda x: np.ascontiguousarray(x)
mock_qt_utils.image.pipeline.backend.process._process_image_pipeline = lambda x, y, z: ImageMetadata(shape=x.shape)
mock_qt_utils.image.settings.lut.ColormapModel = MagicMock
mock_qt_utils.utils.has_qt_cpp_binding = lambda x: False
mock_qt_utils.core.reference.has_qt_cpp_binding = lambda x: False
mock_qt_utils.image.pipeline.monitor.PerformanceMonitor = MagicMock
mock_qt_utils.image.pipeline.monitor.PerfStats = MagicMock

def apply_patches():
    sys.modules['cross_platform.core.shm_ring'] = mock_cp_core.shm_ring
    sys.modules['cross_platform.core.cpu_utils'] = mock_cp_core.cpu_utils
    sys.modules['cross_platform.qt6_utils.image.pipeline.backend.utils'] = mock_qt_utils.image.pipeline.backend.utils
    sys.modules['cross_platform.qt6_utils.image.pipeline.backend.process'] = mock_qt_utils.image.pipeline.backend.process
    sys.modules['cross_platform.qt6_utils.image.pipeline.metadata'] = mock_qt_utils.image.pipeline.metadata
    sys.modules['cross_platform.qt6_utils.image.pipeline.config'] = mock_qt_utils.image.pipeline.config
    sys.modules['cross_platform.qt6_utils.image.settings.lut'] = mock_qt_utils.image.settings.lut
    sys.modules['cross_platform.qt6_utils.utils'] = mock_qt_utils.utils
    sys.modules['cross_platform.qt6_utils.core.reference'] = mock_qt_utils.core.reference
    sys.modules['cross_platform.qt6_utils.image.pipeline.monitor'] = mock_qt_utils.image.pipeline.monitor

apply_patches()
# -----------------------------------------------------------------------------
# 2. IMPORTS (Refactored Classes)
# -----------------------------------------------------------------------------
# Assuming your files are named submitter.py, worker.py, receiver.py
# Adjust if they are in a package structure
from cross_platform.qt6_utils.image.pipeline.submit import FrameSubmitter
from cross_platform.qt6_utils.image.pipeline.frame import FrameHeader
from cross_platform.qt6_utils.image.pipeline.worker import image_worker_entry
from cross_platform.qt6_utils.image.pipeline.receive import FrameReceiver


# -----------------------------------------------------------------------------
# 3. GLOBAL PROCESSORS (Picklable for Multiprocessing)
# -----------------------------------------------------------------------------

def worker_wrapper(input_q, output_q, buf_count, stop_evt, prio, proc_func):
    apply_patches()
    image_worker_entry(input_q, output_q, buf_count, stop_evt, prio, proc_func)

def add_ten_processor(img, out_buf, config):
    np.add(img, 10, out=out_buf, casting='unsafe')
    return ImageMetadata(timestamp=0, shape=img.shape, dtype_str=str(img.dtype))

def passthrough_processor(img, out, cfg):
    np.copyto(out, img, casting='unsafe')
    # FIX: Return out.dtype (float32) instead of img.dtype (uint8)
    return ImageMetadata(shape=img.shape, dtype_str=str(out.dtype))


# -----------------------------------------------------------------------------
# 4. TESTS
# -----------------------------------------------------------------------------

@pytest.fixture
def queues():
    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()
    return input_q, output_q


@pytest.fixture
def stop_event():
    return multiprocessing.Event()


def test_frame_header_structure():
    header = FrameHeader(1.0, 123.789, 1920, 1080, 3)
    packed = header.pack()
    assert len(packed) == 32
    unpacked = FrameHeader.unpack(packed)
    assert unpacked.width == 1920
    assert unpacked.timestamp == 123.789
    assert struct.unpack("<I", packed[-4:])[0] == 0


def test_submitter_writes_header_and_offset(queues):
    input_q, _ = queues
    submitter = FrameSubmitter(input_q, 2)
    try:
        img = np.zeros((100, 100), dtype=np.uint8)
        img[0, 0] = 255
        assert submitter.submit_image(img, {"timestamp": 99.9}) is True

        # macOS fix: Don't use qsize(), use get() with timeout
        try:
            shm_name, shape, dtype, _ = input_q.get(timeout=1.0)
        except multiprocessing.queues.Empty:
            pytest.fail("Queue Empty")

        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            # Copy bytes immediately to avoid holding a view
            hdr = FrameHeader.unpack(bytes(shm.buf[:32]))
            assert hdr.timestamp == 99.9

            # Create view, check, then DELETE view
            pix = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=32)
            try:
                assert pix[0, 0] == 255
            finally:
                del pix  # CRITICAL: Free the buffer view
        finally:
            shm.close()
            shm.unlink()
    finally:
        submitter.cleanup()


def test_worker_processing_flow(queues, stop_event):
    input_q, output_q = queues
    in_shm = shared_memory.SharedMemory(create=True, size=2500 + 32)
    try:
        in_shm.buf[:32] = FrameHeader(1.0, 500.0, 50, 50, 1).pack()

        # Populate inputs, then release view
        pix_in = np.ndarray((50, 50), dtype=np.uint8, buffer=in_shm.buf,
                            offset=32)
        pix_in.fill(10)
        del pix_in  # Release view

        input_q.put((in_shm.name, (50, 50), np.uint8, {}))

        p = multiprocessing.Process(target=worker_wrapper, args=(
        input_q, output_q, 2, stop_event, False, add_ten_processor))
        p.start()
        try:
            out_name, _ = output_q.get(timeout=4.0)
            out_shm = shared_memory.SharedMemory(name=out_name)
            try:
                # Copy header bytes
                assert FrameHeader.unpack(
                    bytes(out_shm.buf[:32])).timestamp == 500.0

                # Check pixel result, then release view
                out_pix = np.ndarray((50, 50), dtype=np.uint8,
                                     buffer=out_shm.buf, offset=32)
                try:
                    # 10 + 10 = 20.0
                    assert out_pix[0, 0] == 20
                finally:
                    del out_pix
            finally:
                out_shm.close()
                out_shm.unlink()
        finally:
            stop_event.set()
            p.join()
    finally:
        in_shm.close()
        in_shm.unlink()

#
def test_receiver_integration(qtbot, stop_event):
    out_q = multiprocessing.Queue()
    shm = shared_memory.SharedMemory(create=True, size=132)
    try:
        shm.buf[:32] = FrameHeader(2.0, 123.0, 10, 10, 1).pack()

        # Write test data, then release view
        pix = np.ndarray((10, 10), dtype=np.uint8, buffer=shm.buf, offset=32)
        pix.fill(42)
        del pix  # CRITICAL

        meta = ImageMetadata(timestamp=123.0, shape=(10, 10), dtype_str='uint8')
        out_q.put((shm.name, meta))

        receiver = FrameReceiver(out_q, stop_event)
        thread = QThread()
        receiver.moveToThread(thread)
        thread.started.connect(receiver.run)

        with qtbot.waitSignal(receiver.processed_img, timeout=2000) as blocker:
            thread.start()

        # Verification happens on COPY, so no buffer error here
        img_res = blocker.args[0]
        assert img_res[0, 0] == 42

        stop_event.set()
        thread.quit()
        thread.wait()
        receiver._cleanup_req.emit()

    finally:
        shm.close()
        shm.unlink()


def test_full_pipeline_sanity(qtbot, queues, stop_event):
    input_q, output_q = queues
    submitter = FrameSubmitter(input_q, 2)
    p = multiprocessing.Process(target=worker_wrapper, args=(
    input_q, output_q, 2, stop_event, False, passthrough_processor))
    p.start()
    receiver = FrameReceiver(output_q, stop_event)
    thread = QThread()
    receiver.moveToThread(thread)
    thread.started.connect(receiver.run)
    thread.start()

    try:
        submitter.submit_image(np.full((20, 20), 7, dtype=np.uint8),
                               {"timestamp": 777.0})
        with qtbot.waitSignal(receiver.processed_img, timeout=3000) as blocker:
            pass
        assert blocker.args[0][0, 0] == 7
    finally:
        stop_event.set()
        thread.quit()
        thread.wait()
        p.join()
        submitter.cleanup()
        receiver.cleanup()


def test_data_integrity_passthrough():
    """
    Verifies that the Worker PRESERVES the data type of the input
    when running a passthrough pipeline.
    """
    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    # 1. Create Input Data (Random uint8 noise)
    # Using specific values to check for casting errors (e.g. 255)
    shape = (100, 100)
    input_data = np.random.randint(0, 256, size=shape, dtype=np.uint8)

    # 2. Setup Input SHM
    # Need 32 bytes for header + data bytes
    in_shm_size = 32 + input_data.nbytes
    in_shm = shared_memory.SharedMemory(create=True, size=in_shm_size)

    try:
        # Write dummy header
        in_shm.buf[:32] = b'\x00' * 32
        # Write data at offset 32
        target_view = np.ndarray(shape, dtype=np.uint8, buffer=in_shm.buf,
                                 offset=32)
        np.copyto(target_view, input_data)
        del target_view  # release view

        # 3. Queue Task
        input_q.put((in_shm.name, shape, np.uint8, {}))

        # 4. Run Worker in Process
        p = multiprocessing.Process(
            target=image_worker_entry,
            args=(input_q, output_q, 2, stop_event, False, passthrough_processor)
        )
        p.start()

        try:
            # 5. Get Result
            out_name, meta = output_q.get(timeout=2.0)

            # 6. Verify Integrity
            out_shm = shared_memory.SharedMemory(name=out_name)
            try:
                # A. Verify Metadata Type
                print(f"\nInput Dtype: {input_data.dtype}")
                print(f"Meta  Dtype: {meta.dtype_str}")

                # If this assertion fails, the worker converted it to float32!
                assert "uint8" in meta.dtype_str, f"Worker changed type to {meta.dtype_str}"

                # B. Verify Byte-for-Byte Equality
                # Reconstruct view
                out_view = np.ndarray(shape, dtype=np.uint8, buffer=out_shm.buf,
                                      offset=32)

                # Check for equality
                are_equal = np.array_equal(input_data, out_view)

                if not are_equal:
                    print("First 5 pixels In: ", input_data.flatten()[:5])
                    print("First 5 pixels Out:", out_view.flatten()[:5])

                assert are_equal, "Output data does not match Input data!"

            finally:
                del out_view
                out_shm.close()
                out_shm.unlink()

        except Exception as e:
            pytest.fail(f"Test failed: {e}")

    finally:
        stop_event.set()
        p.join()
        in_shm.close()
        # in_shm.unlink()
