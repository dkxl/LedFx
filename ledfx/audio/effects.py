"""
Base class for audio effects.
"""
import logging
from functools import cached_property, lru_cache
import numpy as np

from ledfx.effects import Effect
from ledfx.effects.math import ExpFilter
from .analysis import AudioAnalysisSource

_LOGGER = logging.getLogger(__name__)


@Effect.no_registration
class AudioReactiveEffect(Effect):
    """
    Base for audio reactive effects. This really just subscribes
    to the melbank input source and forwards input along to the
    subclasses. This can be expanded to do the common r/g/b filters.
    """

    # this can be used by inheriting classes for power func selection in schema
    # see magnitude or scan effect for examples
    POWER_FUNCS_MAPPING = {
        "Beat": "beat_power",
        "Bass": "bass_power",
        "Lows (beat+bass)": "lows_power",
        "Mids": "mids_power",
        "High": "high_power",
    }

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)
        # protect against possible deactivate race condition
        self.audio = None

    def activate(self, channel):
        _LOGGER.info("Activating AudioReactiveEffect.")
        super().activate(channel)

        if not self._ledfx.audio or id(AudioAnalysisSource) != id(
            self._ledfx.audio.__class__
        ):
            self._ledfx.audio = AudioAnalysisSource(
                self._ledfx, self._ledfx.config.get("audio", {})
            )

        self.audio = self._ledfx.audio
        self._ledfx.audio.subscribe(self._audio_data_updated)

    def deactivate(self):
        _LOGGER.info("Deactivating AudioReactiveEffect.")
        if self.audio:
            self.audio.unsubscribe(self._audio_data_updated)
        super().deactivate()

    def create_filter(self, alpha_decay, alpha_rise):
        # TODO: Since most effects reuse the same general filters it would be
        # nice for all that computation to be shared. This mean that shared
        # filters are needed, or if there is really just a small set of filters
        # that those get added to the Melbank input source instead.
        return ExpFilter(alpha_decay=alpha_decay, alpha_rise=alpha_rise)

    def _audio_data_updated(self):
        self.melbank.cache_clear()
        with self.lock:
            if self.is_active:
                self.audio_data_updated(self.audio)

    def audio_data_updated(self, data):
        """
        Callback for when the audio data is updated. Should
        be implemented by subclasses
        """
        pass

    def clear_melbank_freq_props(self):
        """
        Clears the cached data for selecting and interpolating melbank.
        Almost all the properties used to build the melbank are cached
        to try and ease computational load.
        """

        for prop in [
            "_selected_melbank",
            "_melbank_min_idx",
            "_melbank_max_idx",
            "_input_mel_length",
        ]:
            if hasattr(self, prop):
                delattr(self, prop)

        self._melbank_interp_linspaces.cache_clear()

    @cached_property
    def _selected_melbank(self):
        return next(
            (
                i
                for i, x in enumerate(
                    self.audio.melbanks.melbanks_config["max_frequencies"]
                )
                if x >= self._virtual.frequency_range.max
            ),
            len(self.audio.melbanks.melbanks_config["max_frequencies"]),
        )

    @cached_property
    def _melbank_min_idx(self):
        return next(
            idx
            for idx, freq in enumerate(
                self.audio.melbanks.melbank_processors[
                    self._selected_melbank
                ].melbank_frequencies
            )
            if freq >= self._virtual.frequency_range.min
        )

    @cached_property
    def _melbank_max_idx(self):
        return next(
            (
                idx
                for idx, freq in enumerate(
                    self.audio.melbanks.melbank_processors[
                        self._selected_melbank
                    ].melbank_frequencies
                )
                if freq >= self._virtual.frequency_range.max
            ),
            len(
                self.audio.melbanks.melbank_processors[
                    self._selected_melbank
                ].melbank_frequencies
            ),
        )

    @cached_property
    def _input_mel_length(self):
        return self._melbank_max_idx - self._melbank_min_idx

    @lru_cache(maxsize=16)
    def _melbank_interp_linspaces(self, size):
        old = np.linspace(0, 1, self._input_mel_length)
        new = np.linspace(0, 1, size)
        return (new, old)

    def melbank_no_nan(self, melbank):
        # Check for NaN values in the melbank array, replace with 0 in place
        # Difficult to determine why this happens, but it seems to be related to
        # the audio input device.
        # TODO: Investigate why NaNs are present in the melbank array for some people/devices
        if np.isnan(melbank).any():
            _LOGGER.warning(
                "NaN values detected in the melbank array and replaced with 0."
            )
            # Replace NaN values with 0
            np.nan_to_num(melbank, copy=False)

    @lru_cache(maxsize=None)
    def melbank(self, filtered=False, size=0):
        """
        This little bit of code pulls together information from the effect's
        virtual (which controls the audio frequency range), and uses that
        to deliver the melbank, correctly selected and interpolated, to the effect

        size, int      : interpolate the melbank to the target size. value of 0 is no interpolation
        filtered, bool : melbank with smoothed attack and decay
        """
        if filtered:
            melbank = self.audio.melbanks.melbanks_filtered[
                self._selected_melbank
            ][self._melbank_min_idx : self._melbank_max_idx]
        else:
            melbank = self.audio.melbanks.melbanks[self._selected_melbank][
                self._melbank_min_idx : self._melbank_max_idx
            ]

        self.melbank_no_nan(melbank)

        if size and (self._input_mel_length != size):
            return np.interp(*self._melbank_interp_linspaces(size), melbank)
        else:
            return melbank

    def melbank_thirds(self, **kwargs):
        """
        Returns the melbank split into three sections (unequal length)
        Useful for effects that use lows, mids, and highs
        """
        melbank = self.melbank(**kwargs)
        mel_length = len(melbank)
        splits = tuple(map(lambda i: int(i * mel_length), [0.2, 0.5]))

        return np.split(melbank, splits)
