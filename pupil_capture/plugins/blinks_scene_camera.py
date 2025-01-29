"""
(*)~----------------------------------------------------------------------------------
 Pupil LSL Relay
 Copyright (C) 2012 Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
"""
from .channel import (
    blinks_confidence_channel,
    #fixation_dispersion_channel,
    #fixation_duration_channel,
    #fixation_id_channel,
    #fixation_method_channel,
    #norm_pos_channels,
)
from .outlet import Outlet


class SceneCameraBlinks(Outlet):
    @property
    def name(self) -> str:
        return "blinks"

    @property
    def event_key(self) -> str:
        return "blinks"

    @property
    def lsl_type(self) -> str:
        return "blinks"

    def setup_channels(self):
        return (
            #fixation_id_channel(),
            *blinks_confidence_channel(),
            #*norm_pos_channels(),
            #fixation_dispersion_channel(),
            #fixation_duration_channel(),
            #fixation_method_channel(),
        )
