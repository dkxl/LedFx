"""
Configuration shema and constants for audio sources
The configuration schema is refreshed for each call, so the attributes returned can be dependent on
the current active audio source.
For example an audio device with more than one channel allows selection of the channel to monitor
"""

import logging
import voluptuous as vol
import sounddevice as sd

from ledfx.effects.melbank import FFT_SIZE
from ledfx.api.websocket import WEB_AUDIO_CLIENTS


_LOGGER = logging.getLogger(__name__)

# schema attributes that the API may modify
PERMITTED_AUDIO_KEYS = (
    "min_volume",
    "audio_device",
    "audio_channel",
    "delay_ms",
    "pitch_method",
    "onset_method",
    "pitch_tolerance",
)

# Hostapi name used for Web Audio sources
WEB_AUDIO_NAME = 'WEB AUDIO'

# https://aubio.org/doc/latest/pitch_8h.html
PITCH_METHODS = [
    "yinfft",
    "yin",
    "yinfast",
    # TODO retest mcomb and fcomb with the latest aubio release
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
    "default",  # aubio only provides one tempo method so far
]


def audio_device_selector() -> dict:
    """
    Returns a filtered dict of the available audio devices for use by the UI and APIs
    """
    return {idx: device["display_name"] for idx, device in available_audio_devices().items()}
    # return {device['index']: format_device_name(device) for device in available_audio_device_details().values()}


def _format_display_name(device: dict) -> str:
    """
    Formats the name of the audio device for use within the UI and the schema.
    Uses 'hostapi name: device name' to retain compatibility with the front end UI, which groups devices by hostapi
    Appends the channel number for multichannel audio devices.
    """
    display_name = f"{device['hostapi_name']}: {device['name']}"
    if device['max_input_channels'] > 1:
        display_name = f"{display_name} channel {device['channel']}"
    return display_name


def available_audio_devices() -> dict:
    """
    Returns a dict of audio device attributes, keyed by device index.
    Ignores devices with no input channels.
    Ignores ASIO devices (legacy compatibility).
    For multichannel devices, adds an entry for each channel
    """
    _LOGGER.debug('Refreshing available audio devices')
    available_devices = {}
    idx = 0

    # Start with the local devices
    for device in sd.query_devices():
        if device["max_input_channels"] == 0 or "asio" in device["name"].lower():
            continue
        device["hostapi_name"] = sd.query_hostapis(device["hostapi"])["name"]
        for channel in range(device["max_input_channels"]):
            available_devices[idx] = device.copy()
            available_devices[idx]["channel"] = channel
            idx += 1

    # Now add any web audio clients. WEB_AUDIO_CLIENTS is a set, so sort to make the index more deterministic
    for client in sorted(WEB_AUDIO_CLIENTS):
        available_devices[idx] = {
            "hostapi_name": WEB_AUDIO_NAME,
            "name": f"{client}",
            "max_input_channels": 1,
            "client": client,
            "channel": 0
        }
        idx += 1

    # Add the display_name for use by the UI and public APIs
    for device in available_devices.values():
        device["display_name"] = _format_display_name(device)

    return available_devices


def default_audio_device_index() -> int:
    """
    Returns the index of the default device within available_audio_devices() to use for audio input
    In order of preference:
     - if using Windows WASAPI, the first available loopback device
     - the default local sound device, if available
     - the first available device
    """
    available_devices = available_audio_devices()
    default_input_device_idx = sd.default.device["input"]

    if len(available_devices) == 0:
        _LOGGER.warning(
            "No valid audio input devices found. Unable to use audio reactive effects."
        )
        return -1

    # Return the first available device if we can't find a valid local input device
    default_idx = 0

    for idx, device in available_devices.items():
        if "WASAPI" in device["hostapi_name"] and "loopback" in device["name"].lower():
            default_idx = idx
            break
        if device['index'] == default_input_device_idx:
            default_idx = idx
            break

    _LOGGER.debug(
        "Setting %s as default input device",
        available_devices[default_idx]['display_name'],
    )
    return default_idx


AUDIO_CONFIG_SCHEMA = vol.Schema(
    {
        vol.Optional(
            "audio_device",
            default=0
        ): vol.Any(vol.In(audio_device_selector()), vol.SetTo(default_audio_device_index())),
        vol.Optional("sample_rate", default=60): int,
        vol.Optional("mic_rate", default=44100): int,
        vol.Optional("fft_size", default=FFT_SIZE): int,
        vol.Optional("min_volume", default=0.2): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=1.0)
        ),
        vol.Optional(
            "delay_ms",
            default=0,
            description="Add a delay to LedFx's output to sync with your audio."
                        + "Useful for Bluetooth devices which typically have a short audio lag.",
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
