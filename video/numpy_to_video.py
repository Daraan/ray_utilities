import cv2
import numpy as np


def numpy_to_video(data: list, video_filename: str = "output.mp4", frame_rate: int = 10):
    """

    Note:
        About data.shape

        # Create a video from the images by simply stacking them AND
        # adding an extra B=1 dimension. Note that Tune's WandB logger currently
        # knows how to log the different data types by the following rules:
        # array is shape=3D -> An image (c, h, w).
        # array is shape=4D -> A batch of images (B, c, h, w).
        # array is shape=5D -> A video (batch=1, time, channel, height width),
        # where L is the length of the video.
        # -> Make our video ndarray a 5D one.
    """
    # Create a dummy video file in MP4 format
    video = np.squeeze(data)
    if video.shape[-1] not in (1, 3):
        # For CV2, the channel should be the last dimension
        assert video.shape[-3] in (1, 3)
        video = video.transpose(0, 2, 3, 1)
    _length, frame_height, frame_width, _ = video.shape

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
        for frame in video.reshape(-1, *video.shape[-3:]):
            out.write(frame)

    out.release()
