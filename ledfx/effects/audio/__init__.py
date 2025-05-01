"""
Module for handling local and remote audio input sources and audio analysis
Exposes base classes for Audio Reactive effects
"""
from .sources import AudioInputSource
from .analysis import AudioAnalysisSource
from .effect import AudioReactiveEffect
from .schema import AUDIO_CONFIG_SCHEMA, audio_device_selector
from .melbank import Melbank, Melbanks

__version__ = "0.1.0"
__all__ = [
    'AudioInputSource',
    'AudioAnalysisSource',
    'AudioReactiveEffect',
    'AUDIO_CONFIG_SCHEMA',
    'audio_device_selector',
    'Melbank',
    'Melbanks',
]

# Only used by pitchSpectrum effect
MIN_MIDI = 21
MAX_MIDI = 108
