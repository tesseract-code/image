#!/usr/bin/env python3

"""
Simple camera capture script using OpenCV
"""
import logging
import time
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_single_frame(camera_index: int = 0, grayscale=False):
    """
    Capture a single frame and return as numpy array
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warm up camera
    for _ in range(100):
        cap.read()

    # Capture frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    if grayscale:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_generator(camera_index: int = 0) -> Iterator[np.ndarray]:
    """
    Yields frames one by one.
    Handles setup and cleanup automatically via try/finally.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return

    # Configuration for Mac M1/Performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.debug("Starting generator...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error reading frame")
                break

            # Convert BGR (OpenCV standard) to RGB (Standard for everything else)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Yield control back to the caller with the data
            yield frame_rgb

    except GeneratorExit:
        # This block runs if the caller closes the generator (e.g., break in loop)
        logger.info("Generator closed")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Ensures camera is released even if errors occur
        logger.debug("Releasing camera resource...")
        cap.release()


def show_continuous_capture():
    """
    Example of continuous capture without display window
    Returns frames as numpy arrays for processing
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.warning("Camera not available")
        return

    logger.debug("Starting continuous capture (Ctrl+C to stop)...")

    try:
        logger.debug("Initializing camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Error: Could not open camera")
            return

        # Set camera properties for better performance on M1 Mac
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        logger.debug("Camera initialized successfully!")
        logger.info("\nControls:\n\t"
                    "Press 'q' to quit\n\t"
                    "'g' to toggle grayscale\n\t"
                    "'s' to save frame\n")

        grayscale_mode = False
        frame_count = 0

        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                logger.error("Error: Could not read frame")
                break

            frame_count += 1

            # Choose which frame to display
            if grayscale_mode:
                # Create grayscale version
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = frame_gray
                current_array = frame_gray  # This is your numpy array
                logger.debug(
                    f"Frame {frame_count}: "
                    f"Grayscale array shape: {current_array.shape}, "
                    f"dtype: {current_array.dtype}")
            else:
                display_frame = frame  # Keep BGR for cv2.imshow
                # Convert BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_array = frame_rgb  # This is your numpy array (RGB)
                logger.debug(
                    f"Frame {frame_count}: "
                    f"Color array shape: {current_array.shape}, "
                    f"dtype: {current_array.dtype}")

            # Show the frame
            cv2.imshow('Camera Feed - Mac M1', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Quitting...")
                break
            elif key == ord('g'):
                grayscale_mode = not grayscale_mode
                mode_text = "grayscale" if grayscale_mode else "color"
                logger.info(f"Switched to {mode_text} mode")
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{int(time.time())}.jpg"
                if grayscale_mode:
                    cv2.imwrite(filename, frame_gray)
                else:
                    cv2.imwrite(filename, frame)
                logger.info(f"Saved frame as {filename}")

    except KeyboardInterrupt:
        logger.debug("Stopping capture...")
    finally:
        cap.release()


if __name__ == '__main__':
    show_continuous_capture()
