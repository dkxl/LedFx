# import asyncio
import logging
import time
from functools import cached_property

import numpy as np
import voluptuous as vol
import zeroconf

from ledfx.effects import DummyEffect
from ledfx.effects.math import interpolate_pixels
from ledfx.effects.melbank import FrequencyRange
from ledfx.events import (
    EffectClearedEvent,
    EffectSetEvent,
    Event,
    VirtualUpdateEvent,
)

# from ledfx.config import save_config
from ledfx.transitions import Transitions

_LOGGER = logging.getLogger(__name__)


class Virtual:

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Required(
                "name", description="Friendly name for the device"
            ): str,
            vol.Required(
                "mapping",
                description="Span: Effect spans all segments. Copy: Effect copied on each segment",
                default="span",
            ): vol.In(["span", "copy"]),
            vol.Optional(
                "icon_name",
                description="Icon for the device*",
                default="mdi:led-strip-variant",
            ): str,
            vol.Optional(
                "max_brightness",
                description="Max brightness for the device",
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
            vol.Optional(
                "center_offset",
                description="Number of pixels from the perceived center of the device",
                default=0,
            ): int,
            vol.Optional(
                "preview_only",
                description="Preview the pixels without updating the devices",
                default=False,
            ): bool,
            vol.Optional(
                "transition_time",
                description="Length of transition between effects",
                default=0.4,
            ): vol.All(
                vol.Coerce(float),
                vol.Range(min=0, max=5, min_included=True, max_included=True),
            ),
            vol.Optional(
                "transition_mode",
                description="Type of transition between effects",
                default="Add",
            ): vol.In([mode for mode in Transitions]),
            vol.Optional(
                "frequency_min",
                description="Lowest frequency for this virtual's audio reactive effects",
                default=20,  # GET THIS FROM CORE AUDIO
            ): vol.All(
                vol.Coerce(int),
                vol.Range(
                    min=20, max=10000, min_included=True, max_included=True
                ),
            ),  # AND HERE TOO,
            vol.Optional(
                "frequency_max",
                description="Highest frequency for this virtual's audio reactive effects",
                default=10000,  # GET THIS FROM CORE AUDIO
            ): vol.All(
                vol.Coerce(int),
                vol.Range(
                    min=20, max=10000, min_included=True, max_included=True
                ),
            ),  # AND HERE TOO,
        }
    )

    # vol.Required(
    #     "start_pixel",
    #     description="First pixel the virtual will map onto device (inclusive)",
    # ): int,
    # vol.Required(
    #     "end_pixel",
    #     description="Last pixel the virtual will map onto device (inclusive)",
    # ): int,
    # vol.Optional(
    #     "invert",
    #     description="Reverse the virtual mapping onto this device",
    #     default=False,
    # ): bool,

    _paused = False
    _active = False
    _output_thread = None
    _active_effect = None
    _transition_effect = None

    def __init__(self, ledfx, config):
        self._ledfx = ledfx
        self._config = config
        # the multiplier to fade in/out of an effect. -ve values mean fading
        # in, +ve mean fading out
        self.fade_timer = 0
        self._segments = []

        self.frequency_range = FrequencyRange(
            self._config["frequency_min"], self._config["frequency_max"]
        )

        # list of devices in order of their mapping on the virtual
        # [[id, start, end, invert]...]
        # not a very good schema, but vol seems a bit handicapped in terms of lists.
        # this won't necessarily ensure perfectly validated segments, but it at
        # least gives an idea of the format
        self.SEGMENTS_SCHEMA = vol.Schema([self.validate_segment])

    def __del__(self):
        self.active = False

    def _valid_id(self, id):
        device = self._ledfx.devices.get(id)
        if device is not None:
            return id
        else:
            msg = f"Invalid device id: {id}"
            _LOGGER.warning(msg)
            raise ValueError(msg)

    def activate_segments(self, segments):
        for device_id, start_pixel, end_pixel, invert in segments:
            device = self._ledfx.devices.get(device_id)
            try:
                device.add_segment(self.id, start_pixel, end_pixel, force=True)
            except ValueError as e:  # TODO pass this up the chain
                print(e)
            if not device.is_active():
                device.activate()

    def deactivate_segments(self):
        for device in self._devices:
            device.clear_virtual_segments(self.id)

    def validate_segment(self, segment):
        valid = True
        msg = None

        if len(segment) != 4:
            msg = f"Invalid segment format: {segment}, should be [device_id, start, end, invert]"
            valid = False

        device_id, start_pixel, end_pixel, invert = segment
        device = self._ledfx.devices.get(device_id)

        if device is None:
            msg = f"Invalid device id: {device_id}"
            valid = False

        if (
            (start_pixel < 0)
            or (end_pixel < 0)
            or (start_pixel > end_pixel)
            or (end_pixel >= device.pixel_count)
        ):
            msg = f"Invalid segment pixels: ({start_pixel}, {end_pixel}). Device '{self.name}' valid pixels between (0, {self.pixel_count-1})"
            valid = False

        if not valid:
            _LOGGER.error(msg)
            raise ValueError(msg)
        else:
            return segment

    def invalidate_cached_props(self):
        # invalidate cached properties
        for prop in [
            "pixel_count",
            "refresh_rate",
            "_devices",
            "_segments_by_device",
        ]:
            if hasattr(self, prop):
                delattr(self, prop)

    def update_segments(self, segments_config):
        segments_config = [list(item) for item in segments_config]
        _segments = self.SEGMENTS_SCHEMA(segments_config)

        _pixel_count = self.pixel_count

        if _segments != self._segments:
            if self._active:
                self.deactivate_segments()
                # try to register this new set of segments
                # if it fails, restore previous segments and raise the error
                try:
                    self.activate_segments(_segments)
                except ValueError:
                    self.deactivate_segments()
                    self.activate_segments(self._segments)
                    raise

            self._segments = _segments

            self.invalidate_cached_props()

            _LOGGER.info(
                f"Updated virtual {self.name} with {len(self._segments)} segments, totalling {self.pixel_count} pixels"
            )

            # Restart active effect if total pixel count has changed
            # eg. devices might be reordered, but total pixel count is same
            # so no need to restart the effect
            if self.pixel_count != _pixel_count:
                self.transitions = Transitions(self.pixel_count)
                if self._active_effect is not None:
                    self._active_effect.deactivate()
                    if self.pixel_count > 0:
                        self._active_effect.activate(self)

            mode = self._config["transition_mode"]
            self.frame_transitions = self.transitions[mode]

    def set_effect(self, effect):
        self.transition_frame_total = (
            self.refresh_rate * self._config["transition_time"]
        )
        self.transition_frame_counter = 0

        if self._active_effect is None:
            self._transition_effect = DummyEffect(self.pixel_count)
            self._active_effect = effect
            self._active_effect.activate(self)
            self._ledfx.loop.call_later(
                self._config["transition_time"], self.clear_transition_effect
            )
            self._ledfx.events.fire_event(
                EffectSetEvent(self._active_effect.name)
            )
        else:
            self.clear_transition_effect()
            self._transition_effect = self._active_effect
            self._active_effect = effect
            self._active_effect.activate(self)
            self._ledfx.loop.call_later(
                self._config["transition_time"], self.clear_transition_effect
            )
            self._ledfx.events.fire_event(
                EffectSetEvent(self._active_effect.name)
            )

        self.active = True

    def transition_to_active(self):
        self._active_effect = self._transition_effect
        self._transition_effect = None

    def active_to_transition(self):
        self._transition_effect = self._active_effect
        self._active_effect = None

    def clear_effect(self):
        self._ledfx.events.fire_event(EffectClearedEvent())

        self._transition_effect = self._active_effect
        self._active_effect = DummyEffect(self.pixel_count)

        self.transition_frame_total = (
            self.refresh_rate * self._config["transition_time"]
        )
        self.transition_frame_counter = 0

        self._ledfx.loop.call_later(
            self._config["transition_time"], self.clear_frame
        )

    def clear_transition_effect(self):
        if self._transition_effect is not None:
            self._transition_effect.deactivate()
        self._transition_effect = None

    def clear_active_effect(self):
        if self._active_effect is not None:
            self._active_effect.deactivate()
        self._active_effect = None

    def clear_frame(self):
        self.clear_active_effect()
        self.clear_transition_effect()

        if self._active:
            # Clear all the pixel data before deactivating the device
            self.assembled_frame = np.zeros((self.pixel_count, 3))
            self.flush(self.assembled_frame)
            self._ledfx.events.fire_event(
                VirtualUpdateEvent(self.id, self.assembled_frame)
            )

            self._active = False

    @property
    def active_effect(self):
        return self._active_effect

    def process_active_effect(self):
        # Assemble the frame if necessary, if nothing changed just sleep
        self.assembled_frame = self.assemble_frame()
        if self.assembled_frame is not None and not self._paused:
            if not self._config["preview_only"]:
                self.flush(self.assembled_frame)

            def trigger_virtual_update_event():
                self._ledfx.events.fire_event(
                    VirtualUpdateEvent(self.id, self.assembled_frame)
                )

            self._ledfx.loop.call_soon_threadsafe(trigger_virtual_update_event)

    def thread_function(self):
        if self._active:
            start_time = time.time()
            self.process_active_effect()
            render_latency = time.time() - start_time

            self._ledfx.loop.call_later(
                1 / self.refresh_rate - render_latency, self.thread_function
            )
            # print(f"Virtual processing latency: {render_latency}")

    def assemble_frame(self):
        """
        Assembles the frame to be flushed.
        """

        # Get and process active effect frame
        frame = self._active_effect.get_pixels()
        np.clip(frame, 0, 255, frame)

        if self._config["center_offset"]:
            frame = np.roll(frame, self._config["center_offset"], axis=0)

        # This part handles blending two effects together
        if (
            self._transition_effect is not None
            and self._config["transition_mode"] != "None"
            and self._config["transition_time"] > 0
        ):
            # Get and process transition effect frame
            transition_frame = self._transition_effect.get_pixels()
            np.clip(transition_frame, 0, 255, transition_frame)

            if self._config["center_offset"]:
                transition_frame = np.roll(
                    transition_frame,
                    self._config["center_offset"],
                    axis=0,
                )

            # Blend both frames together
            self.transition_frame_counter += 1
            self.transition_frame_counter = min(
                max(self.transition_frame_counter, 0),
                self.transition_frame_total,
            )
            weight = (
                self.transition_frame_counter / self.transition_frame_total
            )
            self.frame_transitions(
                self.transitions, frame, transition_frame, weight
            )

        np.multiply(frame, self._config["max_brightness"], frame)

        return frame

    def activate(self):
        if not self._devices:
            error = f"Cannot activate virtual {self.id}, it has no configured device segments"
            _LOGGER.warning(error)
            raise RuntimeError(error)
        if not self._active_effect:
            error = f"Cannot activate virtual {self.id}, it has no configured effect"
            _LOGGER.warning(error)
            raise RuntimeError(error)

        _LOGGER.info(f"Activating virtual {self.id}")
        if not self._active:
            self._active = True
            self.activate_segments(self._segments)

        self.thread_function()

    def deactivate(self):
        self._active = False
        self.deactivate_segments()

    # @lru_cache(maxsize=32)
    # def _normalized_linspace(self, size):
    #     return np.linspace(0, 1, size)

    # def interp_channel(self, channel, x_new, x_old):
    #     return np.interp(x_new, x_old, channel)

    # # TODO cache this!
    # # need to be able to access pixels but not through args
    # # @lru_cache(maxsize=8)
    # def interpolate(self, pixels, new_length):
    #     """Resizes the array by linearly interpolating the values"""
    #     if len(pixels) == new_length:
    #         return pixels

    #     x_old = self._normalized_linspace(len(pixels))
    #     x_new = self._normalized_linspace(new_length)

    #     z = np.apply_along_axis(self.interp_channel, 0, pixels, x_new, x_old)
    #     return z

    def flush(self, pixels):
        """
        Flushes the provided data to the devices.
        """
        for device_id, segments in self._segments_by_device.items():
            data = []
            for (
                start,
                stop,
                step,
                device_start,
                device_end,
            ) in segments:
                if self._config["mapping"] == "span":
                    data.append(
                        (pixels[start:stop:step], device_start, device_end)
                    )
                elif self._config["mapping"] == "copy":
                    target_len = device_end - device_start + 1
                    data.append(
                        (
                            interpolate_pixels(pixels, target_len)[::step],
                            device_start,
                            device_end,
                        )
                    )
            device = self._ledfx.devices.get(device_id)
            if device is None:
                _LOGGER.warning("No active devices - Deactivating.")
                self.deactivate()
            elif device.is_active():
                device.update_pixels(self.id, data)
        # self.interpolate.cache_clear()

    @property
    def name(self):
        return self._config["name"]

    @property
    def max_brightness(self):
        return self._config["max_brightness"] * 256

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, active):
        active = bool(active)
        if active and not self._active:
            self.activate()
        if not active and self._active:
            self.deactivate()
        self._active = active

    @property
    def id(self) -> str:
        """Returns the id for the virtual"""
        return getattr(self, "_id", None)

    @property
    def segments(self):
        return self._segments

    @cached_property
    def _segments_by_device(self):
        """
        List to help split effect output to the correct pixels of each device
        """
        data_start = 0
        segments_by_device = {}
        for device_id, device_start, device_end, inverse in self._segments:
            segment_width = device_end - device_start + 1
            if not inverse:
                start = data_start
                stop = data_start + segment_width
                step = 1
            else:
                start = data_start + segment_width - 1
                stop = None if data_start == 0 else data_start - 1
                step = -1
            segment_info = (
                start,
                stop,
                step,
                device_start,
                device_end,
            )
            if device_id in segments_by_device.keys():
                segments_by_device[device_id].append(segment_info)
            else:
                segments_by_device[device_id] = [segment_info]
            data_start += segment_width
        return segments_by_device

    @cached_property
    def _devices(self):
        """
        Return an iterable of the device object of each segment of the virtual
        """
        return list(
            self._ledfx.devices.get(device_id)
            for device_id in {segment[0] for segment in self._segments}
        )

    @cached_property
    def refresh_rate(self):
        if not self._devices:
            return False
        return min(device.max_refresh_rate for device in self._devices)

    @cached_property
    def pixel_count(self):
        if self._config["mapping"] == "span":
            total = 0
            for device_id, start_pixel, end_pixel, invert in self._segments:
                total += end_pixel - start_pixel + 1
            return total
        elif self._config["mapping"] == "copy":
            if self._segments:
                return max(
                    end_pixel - start_pixel + 1
                    for _, start_pixel, end_pixel, _ in self._segments
                )
            else:
                return 0

    @property
    def config(self) -> dict:
        """Returns the config for the object"""
        return getattr(self, "_config", None)

    @config.setter
    def config(self, new_config):
        """Updates the config for an object"""
        if self._config is not None:
            _config = self._config | new_config
        else:
            _config = new_config

        _config = self.CONFIG_SCHEMA(_config)
        _only_frequencies_updated = False

        if hasattr(self, "_config"):
            if _config["mapping"] != self._config["mapping"]:
                self.invalidate_cached_props()
            if _config["transition_mode"] != self._config["transition_mode"]:
                mode = _config["transition_mode"]
                self.frame_transitions = self.transitions[mode]
            # special case, where we do not want to reactive effect just
            # because the frequency range has changed. bit sloppy but
            # does the trick!
            # if the only key in new config is "frequency_range":...
            # changed_configs = set(a.items()) ^ set(b.items())
            if all(
                key in ["frequency_min", "frequency_max"]
                for key in _config.keys()
            ) and (
                _config["frequency_min"] != self._config["frequency_min"]
                or _config["frequency_max"] != self._config["frequency_max"]
            ):
                _only_frequencies_updated = True

        setattr(self, "_config", _config)

        self.frequency_range = FrequencyRange(
            self._config["frequency_min"], self._config["frequency_max"]
        )

        if self._active_effect is not None:
            if _only_frequencies_updated:
                self._active_effect.clear_melbank_freq_props()
            else:
                self._active_effect.deactivate()
                if self.pixel_count > 0:
                    self._active_effect.activate(self)


class Virtuals:
    """Thin wrapper around the device registry that manages virtuals"""

    PACKAGE_NAME = "ledfx.virtuals"
    _paused = False

    def __init__(self, ledfx):
        # super().__init__(ledfx, Virtual, self.PACKAGE_NAME)

        def cleanup_effects(e):
            self.clear_all_effects()

        self._ledfx = ledfx
        self._ledfx.events.add_listener(cleanup_effects, Event.LEDFX_SHUTDOWN)
        self._zeroconf = zeroconf.Zeroconf()
        self._virtuals = {}

    def create_from_config(self, config):
        for virtual in config:
            _LOGGER.info(f"Loading virtual from config: {virtual}")
            self._ledfx.virtuals.create(
                id=virtual["id"],
                config=virtual["config"],
                is_device=virtual["is_device"],
                ledfx=self._ledfx,
            )
            if "segments" in virtual:
                self._ledfx.virtuals.get(virtual["id"]).update_segments(
                    virtual["segments"]
                )
            if "effect" in virtual:
                try:
                    effect = self._ledfx.effects.create(
                        ledfx=self._ledfx,
                        type=virtual["effect"]["type"],
                        config=virtual["effect"]["config"],
                    )
                    self._ledfx.virtuals.get(virtual["id"]).set_effect(effect)
                except vol.MultipleInvalid:
                    _LOGGER.warning(
                        "Effect schema changed. Not restoring effect"
                    )

    def schema(self):
        return Virtual.CONFIG_SCHEMA

    def create(self, id=None, *args, **kwargs):
        """Creates a virtual"""

        # Find the first valid id based on what is already in the registry
        dupe_id = id
        dupe_index = 1
        while id in self._virtuals.keys():
            id = f"{dupe_id}-{dupe_index}"
            dupe_index = dupe_index + 1

        # Create the new virtual and validate the schema.
        _config = kwargs.pop("config", None)
        _is_device = kwargs.pop("is_device", False)
        if _config is not None:
            _config = Virtual.CONFIG_SCHEMA(_config)
            obj = Virtual(config=_config, *args, **kwargs)
        else:
            obj = Virtual(*args, **kwargs)

        # Attach some common properties
        setattr(obj, "_id", id)
        setattr(obj, "is_device", _is_device)

        # Store the object into the internal list and return it
        self._virtuals[id] = obj
        return obj

    def destroy(self, id):
        if id not in self._virtuals:
            raise AttributeError(
                ("Object with id '{}' does not exist.").format(id)
            )
        del self._virtuals[id]

    def __iter__(self):
        return iter(self._virtuals)

    def values(self):
        return self._virtuals.values()

    def clear_all_effects(self):
        for virtual in self.values():
            virtual.clear_frame()

    def pause_all(self):
        self._paused = not self._paused
        for virtual in self.values():
            virtual._paused = self._paused

    def get(self, virtual_id):
        for id, virtual in self._virtuals.items():
            if virtual_id == id:
                return virtual
        return None
