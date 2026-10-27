"""Audio inputs: sound devices (PortAudio, via sounddevice) and the audio of
NDI sources, both captured as 16 kHz mono float32 — the format the speech
engines expect."""

import queue
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_S = 0.05
# NDI sources don't announce their audio channel count up front.
NDI_MAX_CHANNELS = 8
# Most audio the "listen" output may hold before dropping the oldest.
LISTEN_MAX_DELAY_S = 0.3
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
    input level (dBFS) for a meter. With `meter_only`, audio isn't kept at
    all — only the level is updated (the pre-Start signal check). Setting
    `listener` (a ListenOutput) also plays the input, meter_only or not."""

    def __init__(self, device_index, channel=None, meter_only=False):
        info = sd.query_devices(device_index)
        max_ch = int(info["max_input_channels"])
        if channel is not None and not 0 <= channel < max_ch:
            raise ValueError(f"Channel {channel + 1} not available (device has {max_ch}).")
        self.channel = channel
        self.meter_only = meter_only
        n_open = channel + 1 if channel is not None else min(2, max_ch)
        self.native_rate = int(info["default_samplerate"])
        self.level_db = -120.0
        self.listener = None
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
        self.level_db = _level_db(mono)
        listener = self.listener
        if listener is not None:
            listener.feed(self.native_rate, mono)
        if self.meter_only:
            return
        try:
            self._q.put_nowait(mono.copy())
        except queue.Full:
            pass  # consumer stalled: drop audio rather than grow without bound

    def start(self):
        self._stream.start()

    def read(self, timeout=0.1):
        """Returns ALL audio captured since the last call, resampled to
        16 kHz (possibly empty), or None if nothing arrived within the
        timeout. Draining the whole queue matters: a consumer that is slower
        than real time (e.g. a transcription pass) must catch up on its next
        read instead of falling further behind one block at a time."""
        blocks = _drain(self._q, timeout)
        if blocks is None:
            return None
        return self._resampler.process(np.concatenate(blocks))

    def close(self):
        try:
            self._stream.stop()
        finally:
            self._stream.close()


def _level_db(mono):
    rms = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
    return 20.0 * np.log10(max(rms, 1e-6))


def _drain(q, timeout):
    """All blocks queued in `q`, waiting up to `timeout` for the first one;
    None if nothing arrived."""
    try:
        blocks = [q.get(timeout=timeout)]
    except queue.Empty:
        return None
    while True:
        try:
            blocks.append(q.get_nowait())
        except queue.Empty:
            return blocks


class NDIAudioCapture:
    """Same interface as AudioCapture, for the audio of an NDI source — so
    an NDI feed can be captioned directly, without routing it through a
    virtual audio device. Receives audio only (no video over the network).
    `channel` is 0-based, or None to mix the first two channels."""

    def __init__(self, source, channel=None, meter_only=False):
        import NDIlib as ndi

        self._ndi = ndi
        create = ndi.RecvCreateV3()
        create.bandwidth = ndi.RECV_BANDWIDTH_AUDIO_ONLY
        create.ndi_recv_name = "AlphaSubs Audio Receiver"
        self._recv = ndi.recv_create_v3(create)
        if self._recv is None:
            raise RuntimeError("Failed to create the NDI audio receiver.")
        ndi.recv_connect(self._recv, source)
        self.channel = channel
        self.meter_only = meter_only
        self.level_db = -120.0
        self.listener = None
        self._q = queue.Queue(maxsize=400)
        self._rate = None
        self._resampler = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        ndi = self._ndi
        while not self._stop.is_set():
            frame_type, video, audio, _meta = ndi.recv_capture_v2(self._recv, 200, want_metadata=False)
            if frame_type == ndi.FRAME_TYPE_AUDIO:
                data = np.array(audio.data, dtype=np.float32, copy=True)  # planar: (channels, samples)
                rate = int(audio.sample_rate)
                ndi.recv_free_audio_v2(self._recv, audio)
                if data.ndim == 1:
                    data = data[None, :]
                if self.channel is not None and self.channel < data.shape[0]:
                    mono = data[self.channel]
                else:
                    mono = data[:2].mean(axis=0)
                self.level_db = _level_db(mono)
                listener = self.listener
                if listener is not None:
                    listener.feed(rate, mono)
                if not self.meter_only:
                    try:
                        self._q.put_nowait((rate, mono))
                    except queue.Full:
                        pass  # consumer stalled: drop audio rather than grow without bound
            elif frame_type == ndi.FRAME_TYPE_VIDEO:
                ndi.recv_free_video_v2(self._recv, video)

    def read(self, timeout=0.1):
        """Like AudioCapture.read(): all audio received since the last call,
        at 16 kHz, or None if nothing arrived within the timeout."""
        blocks = _drain(self._q, timeout)
        if blocks is None:
            return None
        rate = blocks[-1][0]
        if rate != self._rate:
            self._rate, self._resampler = rate, _Resampler(rate)
        return self._resampler.process(np.concatenate([b for r, b in blocks if r == rate]))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._ndi.recv_destroy(self._recv)


class ListenOutput:
    """Plays a capture's audio on an output device (the system default if
    `device_index` is None), so the operator can hear the input before
    Start. Attach it as a capture's `listener`; feed() is called from the
    capture thread with (rate, mono) blocks at the input's native rate."""

    def __init__(self, device_index=None):
        info = sd.query_devices(device_index, kind="output")
        self.name = info["name"]
        self.rate = int(info["default_samplerate"])
        self._lock = threading.Lock()
        self._chunks = []
        self._len = 0
        self._in_rate = None
        self._resampler = None
        # Input and output run on separate clocks: cap the backlog so a
        # slightly faster input can't build up an ever-growing delay.
        self._max_len = int(self.rate * LISTEN_MAX_DELAY_S)
        self._stream = sd.OutputStream(
            device=device_index, samplerate=self.rate, channels=min(2, int(info["max_output_channels"])),
            dtype="float32", latency="low", callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def feed(self, rate, mono):
        if rate != self._in_rate:
            self._in_rate, self._resampler = rate, _Resampler(rate, self.rate)
        # Copy first: a sound device's callback buffer is reused, and the
        # resampler returns its input as is when the rates match.
        out = self._resampler.process(np.array(mono, dtype=np.float32, copy=True))
        with self._lock:
            self._chunks.append(out)
            self._len += len(out)
            while self._len > self._max_len and len(self._chunks) > 1:
                self._len -= len(self._chunks.pop(0))

    def _callback(self, outdata, frames, _time_info, _status):
        block = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(frames - filled, len(chunk))
                block[filled:filled + take] = chunk[:take]
                filled += take
                if take == len(chunk):
                    self._chunks.pop(0)
                else:
                    self._chunks[0] = chunk[take:]
            self._len -= filled
        outdata[:] = np.clip(block, -1.0, 1.0)[:, None]

    def close(self):
        try:
            self._stream.stop()
        finally:
            self._stream.close()
