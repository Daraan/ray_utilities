"""Video processing utilities for Ray RLlib experiment recordings.

This module provides utilities for converting NumPy arrays to video files, which is
essential for logging episode recordings from Ray RLlib experiments. The videos can
be used for visualizing agent behavior during training and evaluation.

The main functionality includes converting multi-dimensional NumPy arrays (typically
episode recordings) into standard video formats like MP4, with support for different
color channels and frame configurations.

Key Features:
    - Convert NumPy arrays to MP4 video files
    - Handle different array shapes and color channels
    - Temporary video file creation for logging
    - Compatible with various logging frameworks (Wandb, Comet, TensorBoard)

Example:
    Basic video creation from episode recording::

        import numpy as np
        from ray_utilities.video.numpy_to_video import create_temp_video

        # Episode frames: (time, height, width, channels)
        episode_frames = np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8)

        # Create temporary video file
        video_path = create_temp_video(episode_frames)
        print(f"Video saved to: {video_path}")

    Using with Ray RLlib callbacks::

        from ray_utilities.postprocessing import save_videos

        # In a callback, convert video arrays to files
        save_videos(training_results)  # Modifies results in-place

Functions:
    :func:`numpy_to_video`: Convert NumPy array to video file
    :func:`create_temp_video`: Create temporary video with unique filename

See Also:
    :mod:`ray_utilities.postprocessing`: For integrating videos into result processing
    :mod:`ray_utilities.constants`: Video-related constants and configurations
    :mod:`cv2`: OpenCV library used for video encoding
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ray_utilities.temp_dir import TEMP_DIR_PATH


def numpy_to_video(
    data: list[float | int] | NDArray | list[NDArray], video_filename: str = "output.mp4", frame_rate: int = 10
) -> None:
    """Convert NumPy array data to an MP4 video file.

    This function takes episode recording data (typically from Ray RLlib environments)
    and converts it to a standard MP4 video file. It handles various input shapes
    and automatically transposes dimensions to match OpenCV's expected format.

    Args:
        data: Video data as NumPy array or list. Expected shapes:

            - **5D**: ``(batch=1, time, channel, height, width)`` - Standard Ray format
            - **4D**: ``(time, channel, height, width)`` or ``(time, height, width, channel)``
            - **List**: List of frame arrays

        video_filename: Output filename for the video. Should end with ``.mp4``.
        frame_rate: Frames per second for the output video.

    Note:
        **Shape Requirements**:

        Ray Tune's logging frameworks (WandB, TensorBoard) expect specific array shapes:

        - **3D array**: Single image ``(channel, height, width)``
        - **4D array**: Batch of images ``(batch, channel, height, width)``
        - **5D array**: Video ``(batch=1, time, channel, height, width)``

        This function automatically handles the conversion to the correct OpenCV format
        ``(time, height, width, channel)`` regardless of input shape.

    Example:
        >>> import numpy as np
        >>> # Create sample episode data (10 frames, 3 channels, 64x64)
        >>> frames = np.random.randint(0, 255, (10, 3, 64, 64), dtype=np.uint8)
        >>> numpy_to_video(frames, "episode.mp4", frame_rate=30)

        With 5D Ray format::

        >>> # Ray format: (1, time, channel, height, width)
        >>> ray_frames = np.random.randint(0, 255, (1, 20, 3, 128, 128), dtype=np.uint8)
        >>> numpy_to_video(ray_frames, "ray_episode.mp4")

    Raises:
        AssertionError: If the array doesn't have the expected channel dimensions
            (1 or 3 channels) in the correct positions.

    See Also:
        :func:`create_temp_video`: Wrapper that creates temporary filenames
        :mod:`cv2`: OpenCV video writing functionality
    """
    # Create a dummy video file in MP4 format
    video = np.squeeze(data)
    if video.shape[-1] not in (1, 3):
        assert video.shape[-3] in (1, 3)  # (L, C, H, W)
        # For CV2, the channel should be the last dimension
        video = video.transpose(0, 2, 3, 1)  # (L, H, W, C)
    _length, frame_height, frame_width, _colors = video.shape

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # type: ignore[attr-defined]
    out = cv2.VideoWriter(
        video_filename,
        fourcc=fourcc,
        fps=frame_rate,
        frameSize=(frame_width, frame_height),
    )

    if video.ndim == 4:
        for frame in video:
            out.write(frame)
    else:
        for frame in video.reshape(-1, *video.shape[-3:]):  # make 4D; might merges multiple videos
            out.write(frame)

    out.release()


def create_temp_video(
    data: list[float | int] | NDArray | list[NDArray],
    suffix: str = ".mp4",
    frame_rate: int = 10,
    dir: Optional[str] = None,
) -> str:
    """Create a temporary video file from NumPy array data with a unique filename.

    This is a convenience wrapper around :func:`numpy_to_video` that automatically
    generates a unique filename and saves the video to a temporary directory. It's
    particularly useful for creating videos that will be logged to experiment
    tracking platforms.

    Args:
        data: Video data as NumPy array or list. See :func:`numpy_to_video` for
            detailed shape requirements.
        suffix: File extension for the video. Currently only ``.mp4`` is supported.
        frame_rate: Frames per second for the output video.
        dir: Directory to save the video file. If ``None``, saves to the configured
            temporary directory (:data:`ray_utilities.temp_dir.TEMP_DIR_PATH`).

    Returns:
        Full path to the created temporary video file.

    Example:
        Create temporary video for logging::

            >>> import numpy as np
            >>> frames = np.random.randint(0, 255, (50, 64, 64, 3), dtype=np.uint8)
            >>> video_path = create_temp_video(frames, frame_rate=30)
            >>> print(video_path)
            '/tmp/ray_utilities/a1b2c3d4e5f6.mp4'

        Custom directory and frame rate::

            >>> video_path = create_temp_video(frames, dir="/my/videos", frame_rate=60)

    Note:
        - The filename is generated using :func:`uuid.uuid4` for uniqueness
        - Only MP4 format is currently supported
        - The temporary file is not automatically cleaned up

    See Also:
        :func:`numpy_to_video`: Underlying video conversion function
        :data:`ray_utilities.temp_dir.TEMP_DIR_PATH`: Default temporary directory
        :func:`ray_utilities.postprocessing.save_videos`: For batch video processing
    """
    assert suffix == ".mp4", "Only MP4 format is supported"
    path = os.path.join(TEMP_DIR_PATH if dir is None else dir, f"{uuid.uuid4().hex}{suffix}")
    numpy_to_video(data, path, frame_rate)
    return path
