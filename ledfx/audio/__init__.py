"""
Module for handling local and remote audio input sources and audio analysis
Exposes base classes for Audio Reactive effects
"""
from .sources import AudioInputSource
from .analysis import AudioAnalysisSource
from .effects import AudioReactiveEffect
from .schema import refresh_audio_schema, available_audio_devices

__version__ = "0.1.0"
__all__ = [
    'AudioInputSource',
    'AudioAnalysisSource',
    'AudioReactiveEffect',
    'refresh_audio_schema',
    'available_audio_devices',
]

# Only used by pitchSpectrum effect
MIN_MIDI = 21
MAX_MIDI = 108
