"""
Module for handling local and remote audio input sources and audio analysis
Exposes base classes for Audio Reactive effects
"""
from .sources import AudioInputSource, available_audio_sources
from .analysis import AudioAnalysisSource
from .effects import AudioReactiveEffect

__version__ = "0.1.0"
__all__ = ['AudioInputSource', 'AudioAnalysisSource', 'AudioReactiveEffect']

MIN_MIDI = 21
MAX_MIDI = 108
