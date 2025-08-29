"""Video creation utilities for converting NumPy arrays to video files.

Provides utilities for converting NumPy arrays (typically from environment
recordings or visualizations) into video files for experiment analysis and
visualization.

Main Components:
    - :func:`numpy_to_video`: Convert NumPy array frames to video file
    - :func:`create_temp_video`: Create temporary video files
"""

from .numpy_to_video import create_temp_video, numpy_to_video

__all__ = [
    "create_temp_video",
    "numpy_to_video",
]
