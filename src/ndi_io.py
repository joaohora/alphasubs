"""Thin wrapper around the NDIlib bindings: source discovery, receiving, sending."""

import numpy as np
import NDIlib as ndi


def initialize():
    if not ndi.initialize():
        raise RuntimeError("Failed to initialize the NDI runtime (libndi).")


def shutdown():
    ndi.destroy()


def find_sources(timeout_ms=3000):
    """Blocks for up to timeout_ms searching for NDI sources on the network and
    returns the list found."""
    finder = ndi.find_create_v2()
    if finder is None:
        raise RuntimeError("Failed to create the NDI source finder.")
    try:
        ndi.find_wait_for_sources(finder, timeout_ms)
        sources = list(ndi.find_get_current_sources(finder))
    finally:
        ndi.find_destroy(finder)
    return sources


def find_source_by_name(name_substring, timeout_ms=5000, rounds=5):
    """Searches repeatedly until it finds a source whose name contains
    name_substring (case-insensitive)."""
    needle = name_substring.lower()
    for _ in range(rounds):
        for src in find_sources(timeout_ms):
            if needle in src.ndi_name.lower():
                return src
    return None


class Receiver:
    """Receives video frames from an already-resolved NDI source (ndi.Source object)."""

    def __init__(self, source, recv_name="AlphaSubs Receiver"):
        create = ndi.RecvCreateV3()
        create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        create.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST
        create.ndi_recv_name = recv_name
        self._recv = ndi.recv_create_v3(create)
        if self._recv is None:
            raise RuntimeError("Failed to create the NDI receiver.")
        ndi.recv_connect(self._recv, source)

        # Last resolution/frame rate declared by the source (updated on every video frame).
        self.last_xres = None
        self.last_yres = None
        self.last_frame_rate = None  # (N, D)

    def read(self, timeout_ms=1000):
        """Returns a numpy HxWx4 uint8 array (BGRA/BGRX) for the next video
        frame, or None if nothing arrived within the timeout (e.g. audio-only,
        or no data)."""
        frame_type, video, audio, _meta = ndi.recv_capture_v2(
            self._recv, timeout_ms, want_metadata=False
        )

        if frame_type == ndi.FRAME_TYPE_VIDEO:
            frame = np.array(video.data, copy=True)
            self.last_xres = video.xres
            self.last_yres = video.yres
            self.last_frame_rate = (video.frame_rate_N, video.frame_rate_D)
            ndi.recv_free_video_v2(self._recv, video)
            return frame

        if frame_type == ndi.FRAME_TYPE_AUDIO:
            ndi.recv_free_audio_v2(self._recv, audio)

        return None

    def close(self):
        ndi.recv_destroy(self._recv)


class Sender:
    """Sends BGRA (with alpha) frames as a new NDI source."""

    def __init__(self, name, clock_video=True):
        settings = ndi.SendCreate()
        settings.ndi_name = name
        if hasattr(settings, "clock_video"):
            # Makes the NDI SDK pace the send calls to the frame_rate_N/D given
            # in each send(), instead of emitting as fast as frames arrive.
            settings.clock_video = clock_video
        self._send = ndi.send_create(settings)
        if self._send is None:
            raise RuntimeError("Failed to create the NDI sender.")
        self._frame = ndi.VideoFrameV2()
        self._frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA

    def send(self, bgra_frame, frame_rate_n=30000, frame_rate_d=1001):
        assert bgra_frame.dtype == np.uint8 and bgra_frame.shape[2] == 4
        h, w = bgra_frame.shape[:2]
        self._frame.data = bgra_frame
        self._frame.xres = w
        self._frame.yres = h
        self._frame.line_stride_in_bytes = w * 4
        self._frame.frame_rate_N = frame_rate_n
        self._frame.frame_rate_D = frame_rate_d
        ndi.send_send_video_v2(self._send, self._frame)

    def close(self):
        ndi.send_destroy(self._send)
