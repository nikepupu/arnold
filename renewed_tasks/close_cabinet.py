import logging
from .open_cabinet import OpenCabinet


class CloseCabinet(OpenCabinet):
    def __init__(self, num_stages, horizon, stage_properties, record=False) -> None:
        super().__init__(num_stages, horizon, stage_properties, record)
        self.task = 'close_cabinet'
        self.grip_open = [True, False, False]
        self.logger = logging.getLogger(__name__)
