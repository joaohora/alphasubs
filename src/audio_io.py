"""Audio input devices (PortAudio, via sounddevice): discovery and capture as
16 kHz mono float32 — the format the speech engines expect."""

import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_S = 0.05
# WDM-KS devices on Windows are the same hardware already listed under
# MME/DirectSound/WASAPI, and tend to fail to open in shared use.
SKIPPED_HOSTAPIS = ("Windows WDM-KS",)


def _list_devices(channels_key, rescan):
    if rescan:
        sd._terminate()
        sd._initialize()
    hostapis = sd.query_hostapis()
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev[channels_key] <= 0:
            continue
        api = hostapis[dev["hostapi"]]["name"]
        if api in SKIPPED_HOSTAPIS:
            continue
        devices.append((idx, f"{dev['name']} ({api})", int(dev[channels_key])))
    return devices


def list_input_devices(rescan=False):
    """Returns [(device_index, label, max_input_channels)] for every device
    with input channels. The label carries the host API (MME/WASAPI/Core
    Audio/...) since Windows lists the same device once per host API.
    `rescan` re-initializes PortAudio so devices plugged in after startup
    show up (only safe while no stream is open)."""
    return _list_devices("max_input_channels", rescan)


def list_output_devices(rescan=False):
    """Like list_input_devices(), for devices with output channels."""
    return _list_devices("max_output_channels", rescan)


def _default_device(i):
    try:
        idx = sd.default.device[i]
    except (TypeError, IndexError):
        return None
    return idx if idx is not None and idx >= 0 else None


def default_input_device():
    return _default_device(0)


def default_output_device():
    return _default_device(1)


class _Resampler:
    """Streaming resampler to SAMPLE_RATE: windowed-sinc low-pass (anti-alias,
    only when downsampling) followed by linear interpolation. Keeps state
    between calls so block boundaries don't click."""

    def __init__(self, in_rate, out_rate=SAMPLE_RATE, taps=63):
        self.step = in_rate / out_rate
        self.pos = 0.0
        self.pending = np.zeros(0, dtype=np.float32)
        if in_rate > out_rate:
            cutoff = 0.45 * out_rate / in_rate
            n = np.arange(taps) - (taps - 1) / 2.0
            h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
            self.h = (h / h.sum()).astype(np.float32)
            self.hist = np.zeros(taps - 1, dtype=np.float32)
        else:
            self.h = None

    def process(self, x):
        if self.step == 1.0:
            return x
        if self.h is not None:
            buf = np.concatenate([self.hist, x])
            x = np.convolve(buf, self.h, mode="valid").astype(np.float32)
            self.hist = buf[-(len(self.h) - 1):]
        data = np.concatenate([self.pending, x])
        last = len(data) - 1
        if last < self.pos:
            self.pending = data
            return np.zeros(0, dtype=np.float32)
        n_out = int((last - self.pos) // self.step) + 1
        idx = self.pos + self.step * np.arange(n_out)
        out = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
        next_pos = self.pos + self.step * n_out
        consumed = int(next_pos)
        self.pending = data[consumed:]
        self.pos = next_pos - consumed
        return out


class AudioCapture:
    """Captures one input device. `channel` is a 0-based channel index, or
    None to mix the first two channels (mono mics/stereo pairs both work).
    read() returns 16 kHz mono float32 blocks; `level_db` holds the latest
    input level (dBFS) for a meter."""

    def __init__(self, device_index, channel=None):
        info = sd.query_devices(device_index)
        max_ch = int(info["max_input_channels"])
        if channel is not None and not 0 <= channel < max_ch:
            raise ValueError(f"Channel {channel + 1} not available (device has {max_ch}).")
        self.channel = channel
        n_open = channel + 1 if channel is not None else min(2, max_ch)
        self.native_rate = int(info["default_samplerate"])
        self.level_db = -120.0
        self._q = queue.Queue(maxsize=400)
        self._resampler = _Resampler(self.native_rate)
        self._stream = sd.InputStream(
            device=device_index, channels=n_open, samplerate=self.native_rate,
            dtype="float32", blocksize=int(self.native_rate * BLOCK_S),
            callback=self._callback,
        )

    def _callback(self, indata, _frames, _time_info, _status):
        if self.channel is None:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, self.channel]
        rms = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
        self.level_db = 20.0 * np.log10(max(rms, 1e-6))
        try:
            self._q.put_nowait(mono.copy())
        except queue.Full:
            pass  # consumer stalled: drop audio rather than grow without bound

    def start(self):
        self._stream.start()

    def read(self, timeout=0.1):
        """Returns the next block resampled to 16 kHz (possibly empty), or
        None if nothing arrived within the timeout."""
        try:
            block = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        return self._resampler.process(block)

    def close(self):
        try:
            self._stream.stop()
        finally:
            self._stream.close()
