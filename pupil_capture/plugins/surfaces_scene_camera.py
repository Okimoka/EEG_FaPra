"""
(*)~----------------------------------------------------------------------------------
 Pupil LSL Relay
 Copyright (C) 2012 Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
"""
from .channel import (
    surface_gaze_channel, #[normx, normy, confidence, on_surf]
    surface_fixations_channel, #[normx, normy, confidence, on_surf, duration, dispersion]
    #fixation_dispersion_channel,
    #fixation_duration_channel,
    #fixation_id_channel,
    #fixation_method_channel,
    #norm_pos_channels,
)
from .outlet import Outlet


class SceneCameraSurfaces(Outlet):
    @property
    def name(self) -> str:
        return "surfaces"

    @property
    def event_key(self) -> str:
        return "surfaces"

    @property
    def lsl_type(self) -> str:
        return "surfaces"

    def setup_channels(self):
        return (
            #fixation_id_channel(),
            *surface_gaze_channel(),
            *surface_fixations_channel(),
            #*norm_pos_channels(),
            #fixation_dispersion_channel(),
            #fixation_duration_channel(),
            #fixation_method_channel(),
        )
