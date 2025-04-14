import logging
import queue
import threading

import aubio
import numpy as np
import samplerate
import sounddevice as sd
import voluptuous as vol

import ledfx.api.websocket
from ledfx.api.websocket import WEB_AUDIO_CLIENTS, WebAudioStream
from ledfx.effects.math import ExpFilter
from ledfx.effects.melbank import FFT_SIZE, MIC_RATE
from ledfx.events import AudioDeviceChangeEvent, Event

_LOGGER = logging.getLogger(__name__)


WEB_AUDIO_API = 'WEB AUDIO'

# https://aubio.org/doc/latest/pitch_8h.html
PITCH_METHODS = [
    "yinfft",
    "yin",
    "yinfast",
    # mcomb and fcomb appears to just explode something deeep in the aubio code, no logs, no errors, it just dies.
    # "mcomb",
    # "fcomb",
    "schmitt",
    "specacf",
]
# https://aubio.org/doc/latest/specdesc_8h.html
ONSET_METHODS = [
    "energy",
    "hfc",
    "complex",
    "phase",
    "wphase",
    "specdiff",
    "kl",
    "mkl",
    "specflux",
]

TEMPO_METHODS = [
    "default",  # only one method so far
]


def AUDIO_CONFIG_SCHEMA(running_config=None):
    return vol.Schema(
        {
            vol.Optional(
                "audio_device",
                default=default_device_index()
            ): device_index_validator,
            # vol.Optional(
            #     "audio_channel",
            #     default=0
            # ): audio_channel_validator(running_config),
            vol.Optional("sample_rate", default=60): int,
            vol.Optional("mic_rate", default=44100): int,
            vol.Optional("fft_size", default=FFT_SIZE): int,
            vol.Optional("min_volume", default=0.2): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=1.0)
            ),
            vol.Optional(
                "delay_ms",
                default=0,
                description="Add a delay to LedFx's output to sync with your audio. Useful for Bluetooth devices which typically have a short audio lag.",
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=5000)),
            vol.Optional(
                "pitch_method",
                default="yinfft",
                description="Method to detect pitch",
            ): vol.In(PITCH_METHODS),
            vol.Optional(
                "tempo_method",
                default="default"
            ): vol.In(TEMPO_METHODS),
            vol.Optional(
                "onset_method",
                default="hfc",
                description="Method used to detect onsets",
            ): vol.In(ONSET_METHODS),
            vol.Optional(
                "pitch_tolerance",
                default=0.8,
                description="Pitch detection tolerance",
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2)),
        },
        extra=vol.ALLOW_EXTRA,
    )


def query_hostapis():
    return sd.query_hostapis() + ({"name": WEB_AUDIO_API},)


def query_devices():
    return sd.query_devices() + tuple(
        {
            "hostapi": WEB_AUDIO_API,
            "name": f"{client}",
            "max_input_channels": 1,
            "client": client,
        }
        for client in WEB_AUDIO_CLIENTS
    )


def available_audio_sources():
    hostapis = query_hostapis()
    devices = query_devices()
    return {
        idx: f"{hostapis[device['hostapi']]['name']}: {device['name']}"
        for idx, device in enumerate(devices)
        if (
                device["max_input_channels"] > 0
                and "asio" not in device["name"].lower()
        )
    }


def device_index_validator(val):
    """
    Validates audio device index in case the saved setting is no longer valid
    """
    if val in valid_device_indexes():
        return val
    return default_device_index()


def valid_device_indexes():
    """
    A list of integers corresponding to valid input devices
    """
    return tuple(available_audio_sources().keys())


def validate_audio_device_index(idx) -> bool:
    """Is this device index valid?"""
    return idx in valid_device_indexes()


def default_device_index():
    """
    Finds the WASAPI loopback device index of the default output device if it exists
    If it does not exist, return the default input device index
    Returns:
        integer: the sounddevice device index to use for audio input
    """
    device_list = sd.query_devices()
    default_output_device_idx = sd.default.device["output"]
    default_input_device_idx = sd.default.device["input"]
    if len(device_list) == 0 or default_output_device_idx == -1:
        _LOGGER.warning("No audio output devices found.")
    else:
        default_output_device_name = device_list[
            default_output_device_idx
        ]["name"]

        # We need to run over the device list looking for the target devices name
        _LOGGER.debug(
            f"Looking for audio loopback device for default output device at index {default_output_device_idx}: {default_output_device_name}"
        )
        for device_index, device in enumerate(device_list):
            # sometimes the audio device name string is truncated, so we need to match what we have and Loopback but otherwise be sloppy
            if (
                    default_output_device_name in device["name"]
                    and "[Loopback]" in device["name"]
            ):
                # Return the loopback device index
                _LOGGER.debug(
                    f"Found audio loopback device for default output device at index {device_index}: {device['name']}"
                )
                return device_index

    # The default input device index is not always valid (i.e no default input devices)
    valid_indexes = valid_device_indexes()
    if len(valid_indexes) == 0:
        _LOGGER.warning(
            "No valid audio input devices found. Unable to use audio reactive effects."
        )
        return None
    else:
        if default_input_device_idx in valid_indexes:
            _LOGGER.debug(
                f"No audio loopback device found for default output device. Using default input device at index {default_input_device_idx}: {device_list[default_input_device_idx]['name']}"
            )
            return default_input_device_idx
        else:
            # Return the first valid input device index if we can't find a valid default input device
            if len(valid_indexes) > 0:
                first_valid_idx = next(iter(valid_indexes))
                _LOGGER.debug(
                    f"No valid default audio input device found. Using first valid input device at index {first_valid_idx}: {device_list[first_valid_idx]['name']}"
                )
                return first_valid_idx


def audio_channel_validator(running_config):
    """Returns a list of the available audio channels for the current active sound device"""
    return [0]


class AudioInputSource:
    _audio_stream_active = False
    _audio = None
    _stream = None
    _callbacks = []
    _audioWindowSize = 4
    _processed_audio_sample = None
    _volume = -90
    _volume_filter = ExpFilter(-90, alpha_decay=0.99, alpha_rise=0.99)
    _subscriber_threshold = 0
    _timer = None

    def __init__(self, ledfx_instance, config):
        self._ledfx = ledfx_instance
        self._config = config
        self._active_device_index = None
        self.lock = threading.Lock()
        # We must not inherit legacy _callbacks from prior instances
        self._callbacks = []
        self.update_config(config)

        def shutdown_event(e):
            # We give the rest of LedFx a second to shutdown before we deactivate the audio subsystem.
            # This is to prevent LedFx hanging on shutdown if the audio subsystem is still running while
            # effects are being unloaded. This is a bit hacky but it works.
            self._timer = threading.Timer(0.5, self.check_and_deactivate)
            self._timer.start()

        self._ledfx.events.add_listener(shutdown_event, Event.LEDFX_SHUTDOWN)

    def active_audio_schema(self):
        """Returns the config schema for the active sound device"""
        return AUDIO_CONFIG_SCHEMA(self._ledfx.config)

    def active_device_index(self):
        """Returns the active audio device index"""
        return self._active_device_index

    def update_config(self, config):
        """Deactivate the audio, update the config, the reactivate"""
        old_input_device = False
        if hasattr(self, "_config"):
            old_input_device = self._config["audio_device"]

        if self._audio_stream_active:
            self.deactivate()
        self._config = self.active_audio_schema()(config)
        if len(self._callbacks) != 0:
            self.activate()
        if (
            old_input_device
            and self._config["audio_device"] is not old_input_device
        ):
            self._ledfx.events.fire_event(
                AudioDeviceChangeEvent(
                    available_audio_sources()[self._config["audio_device"]]
                )
            )
        self._ledfx.config["audio"] = self._config

    def activate(self):
        if self._audio is None:
            try:
                self._audio = sd
            except OSError as Error:
                _LOGGER.critical(f"Sounddevice error: {Error}. Shutting down.")
                self._ledfx.stop()

        # Enumerate all of the input devices and find the one matching the
        # configured host api and device name
        input_devices = query_devices()

        hostapis = query_hostapis()
        default_device = default_device_index()
        if default_device is None:
            # There are no valid audio input devices, so we can't activate the audio source.
            # We should never get here, as we check for devices on start-up.
            # This likely just captures if a device is removed after start-up.
            _LOGGER.warning(
                "Audio input device not found. Unable to activate audio source. Deactivating."
            )
            self.deactivate()
            return
        valid_indexes = valid_device_indexes()
        _LOGGER.debug("********************************************")
        _LOGGER.debug("Valid audio input devices:")
        for index in valid_indexes:
            hostapi_name = hostapis[input_devices[index]["hostapi"]]["name"]
            device_name = input_devices[index]["name"]
            input_channels = input_devices[index]["max_input_channels"]
            _LOGGER.debug(
                f"Audio Device {index}\t{hostapi_name}\t{device_name}\tinput_channels: {input_channels}"
            )
        _LOGGER.debug("********************************************")
        device_idx = self._config["audio_device"]
        _LOGGER.debug(
            f"default_device: {default_device} config_device: {device_idx}"
        )

        if device_idx > max(valid_indexes):
            _LOGGER.warning(
                f"Audio device out of range: {device_idx}. Reverting to default input device: {default_device}"
            )
            device_idx = default_device

        elif device_idx not in valid_indexes:
            _LOGGER.warning(
                f"Audio device {input_devices[device_idx]['name']} not in valid_device_indexes. Reverting to default input device: {default_device}"
            )
            device_idx = default_device

        # Setup a pre-emphasis filter to balance the input volume of lows to highs
        self.pre_emphasis = aubio.digital_filter(3)
        # depending on the coeffs type, we need to use different pre_emphasis values to make em work better. allegedly.
        selected_coeff = self._ledfx.config["melbanks"]["coeffs_type"]
        if selected_coeff == "matt_mel":
            _LOGGER.debug("Using matt_mel settings for pre-emphasis.")
            self.pre_emphasis.set_biquad(
                0.8268, -1.6536, 0.8268, -1.6536, 0.6536
            )
        elif selected_coeff == "scott_mel":
            _LOGGER.debug("Using scott_mel settings for pre-emphasis.")
            self.pre_emphasis.set_biquad(
                1.3662, -1.9256, 0.5621, -1.9256, 0.9283
            )
        else:
            _LOGGER.debug("Using generic settings for pre-emphasis")
            self.pre_emphasis.set_biquad(
                0.85870, -1.71740, 0.85870, -1.71605, 0.71874
            )

        freq_domain_length = (self._config["fft_size"] // 2) + 1

        self._raw_audio_sample = np.zeros(
            MIC_RATE // self._config["sample_rate"],
            dtype=np.float32,
        )

        # Setup the phase vocoder to perform a windowed FFT
        self._phase_vocoder = aubio.pvoc(
            self._config["fft_size"],
            MIC_RATE // self._config["sample_rate"],
        )
        self._frequency_domain_null = aubio.cvec(self._config["fft_size"])
        self._frequency_domain = self._frequency_domain_null
        self._frequency_domain_x = np.linspace(
            0,
            MIC_RATE,
            freq_domain_length,
        )

        samples_to_delay = int(
            0.001 * self._config["delay_ms"] * self._config["sample_rate"]
        )
        if samples_to_delay:
            self.delay_queue = queue.Queue(maxsize=samples_to_delay)
        else:
            self.delay_queue = None

        def open_audio_stream(device_idx):
            """
            Opens an audio stream for the specified input device.
            Parameters:
            device_idx (int): The index of the input device to open the audio stream for.
            Behavior:
            - Detects if the device is a Windows WASAPI Loopback device and logs its name and channel count.
            - If the device is a WEB AUDIO device, initializes a WebAudioStream and sets it as the active audio stream.
            - For other devices, initializes an InputStream with the device's default sample rate and other parameters.
            - Initializes a resampler with the "sinc_fastest" algorithm that downmixes the source to a single-channel.
            - Logs the name of the opened audio source.
            - Starts the audio stream and sets the audio stream active flag to True.
            """

            device = input_devices[device_idx]
            channels = None
            if (
                hostapis[device["hostapi"]]["name"] == "Windows WASAPI"
                and "Loopback" in device["name"]
            ):
                _LOGGER.info(
                    f"Loopback device detected: {device['name']} with {device['max_input_channels']} channels"
                )
            else:
                # if are not a windows loopback device, we will downmix to mono
                # issue seen with poor audio behaviour on Mac and Linux
                # this is similar to the long standing prior implementation
                channels = 1

            if hostapis[device["hostapi"]]["name"] == "WEB AUDIO":
                ledfx.api.websocket.ACTIVE_AUDIO_STREAM = self._stream = (
                    WebAudioStream(
                        device["client"], self._audio_sample_callback
                    )
                )
            else:
                self._stream = self._audio.InputStream(
                    samplerate=int(device["default_samplerate"]),
                    device=device_idx,
                    callback=self._audio_sample_callback,
                    dtype=np.float32,
                    latency="low",
                    blocksize=int(
                        device["default_samplerate"]
                        / self._config["sample_rate"]
                    ),
                    # only pass channels if we set it to something other than None
                    **({"channels": channels} if channels is not None else {}),
                )

            self.resampler = samplerate.Resampler("sinc_fastest", channels=1)

            _LOGGER.info(
                f"Audio source opened: {hostapis[device['hostapi']]['name']}: {device.get('name', device.get('client'))}"
            )

            self._stream.start()
            self._audio_stream_active = True

        try:
            open_audio_stream(device_idx)
            self._active_device_index = device_idx
        except OSError as e:
            _LOGGER.critical(
                f"Unable to open Audio Device: {e} - please retry."
            )
            self.deactivate()
        except sd.PortAudioError as e:
            _LOGGER.error(f"{e}, Reverting to default input device")
            open_audio_stream(default_device)

    def deactivate(self):
        with self.lock:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._audio_stream_active = False
            self._active_device_index = None
        _LOGGER.info("Audio source closed.")

    def subscribe(self, callback):
        """Registers a callback with the input source"""
        self._callbacks.append(callback)
        if len(self._callbacks) > 0 and not self._audio_stream_active:
            self.activate()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def unsubscribe(self, callback):
        """Unregisters a callback with the input source"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
        if (
            len(self._callbacks) <= self._subscriber_threshold
            and self._audio_stream_active
        ):
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(5.0, self.check_and_deactivate)
            self._timer.start()

    def check_and_deactivate(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        if (
            len(self._callbacks) <= self._subscriber_threshold
            and self._audio_stream_active
        ):
            self.deactivate()

    def get_device_index_by_name(self, device_name: str):
        for key, value in available_audio_sources().items():
            if device_name == value:
                return key
        return -1

    def _audio_sample_callback(self, in_data, frame_count, time_info, status):
        """Callback for when a new audio sample is acquired"""
        # time_start = time.time()
        # self._raw_audio_sample = np.frombuffer(in_data, dtype=np.float32)
        raw_sample = np.frombuffer(in_data, dtype=np.float32)

        in_sample_len = len(raw_sample)
        out_sample_len = MIC_RATE // self._config["sample_rate"]

        if in_sample_len != out_sample_len:
            # Simple resampling
            processed_audio_sample = self.resampler.process(
                raw_sample,
                # MIC_RATE / self._stream.samplerate
                out_sample_len / in_sample_len,
                # end_of_input=True
            )
        else:
            processed_audio_sample = raw_sample

        if len(processed_audio_sample) != out_sample_len:
            _LOGGER.debug(
                f"Discarded malformed audio frame - {len(processed_audio_sample)} samples, expected {out_sample_len}"
            )
            return

        # handle delaying the audio with the queue
        if self.delay_queue:
            try:
                self.delay_queue.put_nowait(processed_audio_sample)
            except queue.Full:
                self._raw_audio_sample = self.delay_queue.get_nowait()
                self.delay_queue.put_nowait(processed_audio_sample)
                self.pre_process_audio()
                self._invalidate_caches()
                self._invoke_callbacks()
        else:
            self._raw_audio_sample = processed_audio_sample
            self.pre_process_audio()
            self._invalidate_caches()
            self._invoke_callbacks()

        # print(f"Core Audio Processing Latency {round(time.time()-time_start, 3)} s")
        # return self._raw_audio_sample

    def _invoke_callbacks(self):
        """Notifies all clients of the new data"""
        for callback in self._callbacks:
            callback()

    def _invalidate_caches(self):
        """Invalidates the necessary cache"""
        pass

    def pre_process_audio(self):
        """
        Pre-processing stage that will run on every sample, only
        core functionality that will be used for every audio effect
        should be done here. Everything else should be deferred until
        queried by an effect.
        """
        # clean up nans that have been mysteriously appearing..
        self._raw_audio_sample[np.isnan(self._raw_audio_sample)] = 0

        # Calculate the current volume for silence detection
        self._volume = 1 + aubio.db_spl(self._raw_audio_sample) / 100
        self._volume = max(0, min(1, self._volume))
        self._volume_filter.update(self._volume)

        # Calculate the frequency domain from the filtered data and
        # force all zeros when below the volume threshold
        if self._volume_filter.value > self._config["min_volume"]:
            self._processed_audio_sample = self._raw_audio_sample

            # Perform a pre-emphasis to balance the highs and lows
            if self.pre_emphasis:
                self._processed_audio_sample = self.pre_emphasis(
                    self._raw_audio_sample
                )

            # Pass into the phase vocoder to get a windowed FFT
            self._frequency_domain = self._phase_vocoder(
                self._processed_audio_sample
            )
        else:
            self._frequency_domain = self._frequency_domain_null

    def audio_sample(self, raw=False):
        """Returns the raw audio sample"""

        if raw:
            return self._raw_audio_sample
        return self._processed_audio_sample

    def frequency_domain(self):
        return self._frequency_domain

    def volume(self, filtered=True):
        if filtered:
            return self._volume_filter.value
        return self._volume


