import logging
import time
from collections import deque
from functools import lru_cache

import aubio
import numpy as np


from ledfx.effects.math import ExpFilter
from ledfx.effects.melbank import MIC_RATE, Melbanks

from .sources import AudioInputSource

_LOGGER = logging.getLogger(__name__)


class AudioAnalysisSource(AudioInputSource):

    # some frequency constants
    # beat, bass, mids, high
    freq_max_mels = [
        100,
        250,
        3000,
        10000,
    ]

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)
        self.initialise_analysis()

        # Subscribe functions to be run on every frame of audio
        self.subscribe(self.melbanks)
        self.subscribe(self.pitch)
        self.subscribe(self.onset)
        self.subscribe(self.bar_oscillator)
        self.subscribe(self.volume_beat_now)
        self.subscribe(self.freq_power)

        # ensure any new analysis callbacks are above this line
        self._subscriber_threshold = len(self._callbacks)

    def initialise_analysis(self):
        # melbanks
        if not hasattr(self, "melbanks"):
            self.melbanks = Melbanks(
                self._ledfx, self, self._ledfx.config.get("melbanks", {})
            )

        fft_params = (
            self._config["fft_size"],
            MIC_RATE // self._config["sample_rate"],
            MIC_RATE,
        )

        # pitch, tempo, onset
        self._tempo = aubio.tempo(self._config["tempo_method"], *fft_params)
        self._onset = aubio.onset(self._config["onset_method"], *fft_params)
        self._pitch = aubio.pitch(self._config["pitch_method"], *fft_params)
        self._pitch.set_unit("midi")
        self._pitch.set_tolerance(self._config["pitch_tolerance"])

        # bar oscillator
        self.beat_counter = 0

        # beat oscillator
        self.beat_timestamp = time.time()
        self.beat_period = 2

        # freq power
        self.freq_power_raw = np.zeros(len(self.freq_max_mels))
        self.freq_power_filter = ExpFilter(
            np.zeros(len(self.freq_max_mels)), alpha_decay=0.2, alpha_rise=0.97
        )
        self.freq_mel_indexes = []

        for freq in self.freq_max_mels:
            assert self.melbanks.melbanks_config["max_frequencies"][2] >= freq

            self.freq_mel_indexes.append(
                next(
                    (
                        i
                        for i, f in enumerate(
                            self.melbanks.melbank_processors[
                                2
                            ].melbank_frequencies
                        )
                        if f > freq
                    ),
                    len(
                        self.melbanks.melbank_processors[2].melbank_frequencies
                    ),
                )
            )

        # volume based beat detection
        self.beat_max_mel_index = next(
            (
                i - 1
                for i, f in enumerate(
                    self.melbanks.melbank_processors[0].melbank_frequencies
                )
                if f > self.freq_max_mels[0]
            ),
            self.melbanks.melbank_processors[0].melbank_frequencies[-1],
        )

        self.beat_min_percent_diff = 0.5
        self.beat_min_time_since = 0.1
        self.beat_min_amplitude = 0.5
        self.beat_power_history_len = int(self._config["sample_rate"] * 0.2)

        self.beat_prev_time = time.time()
        self.beat_power_history = deque(maxlen=self.beat_power_history_len)

    def update_config(self, config):
        super().update_config(config)
        self.initialise_analysis()

    def _invalidate_caches(self):
        """Invalidates the cache for all melbank related data"""
        super()._invalidate_caches()

        self.pitch.cache_clear()
        self.onset.cache_clear()
        self.bpm_beat_now.cache_clear()
        self.volume_beat_now.cache_clear()
        self.bar_oscillator.cache_clear()

    @lru_cache(maxsize=None)
    def pitch(self):
        # If our audio handler is returning null, then we just return 0 for midi_value and wait for the device starts sending audio.
        try:
            return self._pitch(self.audio_sample(raw=True))[0]
        except ValueError as e:
            _LOGGER.warning(e)
            return 0

    @lru_cache(maxsize=None)
    def onset(self):
        try:
            return bool(self._onset(self.audio_sample(raw=True))[0])
        except ValueError as e:
            _LOGGER.warning(e)
            return 0

    @lru_cache(maxsize=None)
    def bpm_beat_now(self):
        """
        Returns True if a beat is expected now based on BPM data
        """
        try:
            return bool(self._tempo(self.audio_sample(raw=True))[0])
        except ValueError as e:
            _LOGGER.warning(e)
            return False

    @lru_cache(maxsize=None)
    def volume_beat_now(self):
        """
        Returns True if a beat is expected now based on volume of the beat freq region
        This algorithm is a bit weird, but works quite nicely.
        I've tried my best to optimise it from the original
        implementation in systematic_leds
        """

        time_now = time.time()
        melbank = self.melbanks.melbanks[0][: self.beat_max_mel_index]
        beat_power = np.sum(melbank)
        melbank_max = np.max(melbank)

        # calculates the % difference of the first value of the channel to the average for the channel
        if sum(self.beat_power_history) > 0:
            difference = (
                beat_power
                * self.beat_power_history_len
                / sum(self.beat_power_history)
                - 1
            )
        else:
            difference = 0

        self.beat_power_history.appendleft(beat_power)

        if (
            difference >= self.beat_min_percent_diff
            and melbank_max >= self.beat_min_amplitude
            and time_now - self.beat_prev_time > self.beat_min_time_since
        ):
            self.beat_prev_time = time_now
            return True
        else:
            return False

    def freq_power(self):
        # hard coded this bc i'm tired and it'll run faster

        melbank = self.melbanks.melbanks[2]

        self.freq_power_raw[0] = np.average(
            melbank[: self.freq_mel_indexes[0]]
        )
        self.freq_power_raw[1] = np.average(
            melbank[self.freq_mel_indexes[0] : self.freq_mel_indexes[1]]
        )
        self.freq_power_raw[2] = np.average(
            melbank[self.freq_mel_indexes[1] : self.freq_mel_indexes[2]]
        )
        self.freq_power_raw[3] = np.average(
            melbank[self.freq_mel_indexes[2] : self.freq_mel_indexes[3]]
        )
        np.minimum(self.freq_power_raw, 1, out=self.freq_power_raw)
        self.freq_power_filter.update(self.freq_power_raw)

    def get_freq_power(self, i, filtered=True):
        if filtered:
            value = self.freq_power_filter.value[i]
        else:
            value = self.freq_power_raw[i]

        return value if not np.isnan(value) else 0.0

    def beat_power(self, filtered=True):
        """
        Returns a float (0<=x<=1) corresponding to the beat power
        """
        return self.get_freq_power(0, filtered)

    def bass_power(self, filtered=True):
        """
        Returns a float (0<=x<=1) corresponding to the bass power
        """
        return self.get_freq_power(1, filtered)

    def lows_power(self, filtered=True):
        """
        Returns a float (0<=x<=1) corresponding to the lows power.
        this is just the sum of bass and beat power.
        """
        return (
            self.get_freq_power(0, filtered) + self.get_freq_power(1, filtered)
        ) * 0.5

    def mids_power(self, filtered=True):
        """
        Returns a float (0<=x<=1) corresponding to the mids power
        """
        return self.get_freq_power(2, filtered)

    def high_power(self, filtered=True):
        """
        Returns a float (0<=x<=1) corresponding to the highs power
        """
        return self.get_freq_power(3, filtered)

    @lru_cache(maxsize=None)
    def bar_oscillator(self):
        """
        Returns a float (0<=x<4) corresponding to the position of the beat
        tracker in the musical bar (4 beats)
        This is synced and quantized to the bpm of whatever is playing.
        While the beat number might not necessarily be accurate, the
        relative position of the tracker between beats will be quite accurate.

        NOTE: currently this makes no attempt to guess which beat is the first
        in the bar. It simple counts to four with each beat that is detected.
        The actual value of the current beat in the bar is completely arbitrary,
        but in time with each beat.

        0           1           2           3
        {----------time for one bar---------}
               ^    -->      -->      -->
            value of
        beat grid pointer
        """
        # update tempo and oscillator
        # print(self._tempo.get_delay_s())
        if self.bpm_beat_now():
            self.beat_counter = (self.beat_counter + 1) % 4
            self.beat_period = self._tempo.get_period_s()
            # print("beat at:", self._tempo.get_delay_s())
            self.beat_timestamp = time.time()
            oscillator = self.beat_counter
        else:
            time_since_beat = time.time() - self.beat_timestamp
            oscillator = (
                1 - (self.beat_period - time_since_beat) / self.beat_period
            ) + self.beat_counter
            # ensure it's between [0 and 4). useful when audio cuts
            oscillator = oscillator % 4
        return oscillator

    def beat_oscillator(self):
        """
        returns a float (0<=x<1) corresponding to the relative position of the
        bar oscillator in the current beat.

        0                0.5                 <1
        {----------time for one beat---------}
               ^    -->      -->      -->
            value of
           oscillator
        """
        return self.bar_oscillator() % 1

