import logging
import queue
import threading
import aubio
import numpy as np
import samplerate
import sounddevice as sd

import ledfx.api.websocket
from ledfx.api.websocket import WebAudioStream
from ledfx.effects.math import ExpFilter
from ledfx.effects.melbank import MIC_RATE
from ledfx.events import AudioDeviceChangeEvent, Event

from .schema import (refresh_audio_schema, available_audio_device_details, default_audio_device_details,
                     WEB_AUDIO_NAME)


_LOGGER = logging.getLogger(__name__)


class AudioInputSource:
    _audio_stream_active = False
    _audio = None
    _stream = None
    _audioWindowSize = 4
    _processed_audio_sample = None
    _volume = -90
    _volume_filter = ExpFilter(-90, alpha_decay=0.99, alpha_rise=0.99)
    _subscriber_threshold = 0
    _timer = None

    def __init__(self, ledfx_instance, config):
        self._ledfx = ledfx_instance
        self._config = config
        self._active_device = None
        self.lock = threading.Lock()
        self._callbacks = []

        self._prepare_filters()
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
        return refresh_audio_schema(self._ledfx.config)

    def active_device_index(self):
        """Returns the active audio device index"""
        return self._active_device['index']

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
                    # TODO: who subscribes to this event? Do they need the device attributes or just the device name?
                    available_audio_device_details()[self._config["audio_device"]]
                )
            )
        self._ledfx.config["audio"] = self._config

    def activate(self):
        """activate the audio source"""
        if self._audio is None:
            try:
                self._audio = sd
            except OSError as error:
                _LOGGER.critical("Sounddevice error: %s. Shutting down.", error)
                self._ledfx.stop()

        # Check the available input devices - the configured device may have been removed
        available_devices = available_audio_device_details()

        if not available_devices:
            _LOGGER.warning(
                "No audio input devices available. Unable to activate audio source. Deactivating."
            )
            self.deactivate()
            return

        _LOGGER.debug("********************************************")
        _LOGGER.debug("Available audio input devices:")
        for index, device in available_devices.items():
            _LOGGER.debug(
                "%s\tchannels: %s",
                index, device["max_input_channels"]
            )
        _LOGGER.debug("********************************************")

        if self._config["audio_device"] in available_devices:
            new_audio_device = available_devices[self._config["audio_device"]]
        else:
            new_audio_device = default_audio_device_details()
            _LOGGER.warning(
                "Requested audio device %s not available, reverting to default input device %s",
                self._config["audio_device"], new_audio_device['name'],
            )

        try:
            self._open_audio_stream(new_audio_device)
            self._active_device = new_audio_device
        except (sd.PortAudioError, OSError) as err:
            _LOGGER.critical(
                "Unable to open Audio Device %s: %s - please retry",
                new_audio_device['name'], err
            )
            self.deactivate()

    def _prepare_filters(self):
        """Configure audio filters"""
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

    def _open_audio_stream(self, device):
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

        if "WASAPI" in device.get('hostapi_name') and "loopback" in device['name'].lower():
            _LOGGER.info("WASAPI Loopback device detected: %s", device['name'])
            mono = False
        else:
            # if not using a Windows loopback device, downmix to mono
            # issue seen with poor audio behaviour on Mac and Linux
            # this is similar to the long standing prior implementation
            mono = True

        if device.get('hostapi_name') == WEB_AUDIO_NAME:
            ledfx.api.websocket.ACTIVE_AUDIO_STREAM = self._stream = (
                WebAudioStream(
                    device["client"], self._audio_sample_callback
                )
            )
        else:
            self._stream = self._audio.InputStream(
                samplerate=int(device["default_samplerate"]),
                device=device["index"],   # the host OS device index, not the audio schema index
                callback=self._audio_sample_callback,
                dtype=np.float32,
                latency="low",
                blocksize=int(
                    device["default_samplerate"]
                    / self._config["sample_rate"]
                ),
                channels=1 if mono else None,
            )

        self.resampler = samplerate.Resampler("sinc_fastest", channels=1)

        _LOGGER.info("Audio source opened: %s", device['name'])

        self._stream.start()
        self._audio_stream_active = True

    def deactivate(self):
        with self.lock:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._audio_stream_active = False
            self._active_device = None
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
