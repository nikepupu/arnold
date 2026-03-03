import logging
from .pickup_object import PickupObject


class ReorientObject(PickupObject):
    def __init__(self, num_stages, horizon, stage_properties, record) -> None:
        super().__init__(num_stages, horizon, stage_properties, record)
        self.task = 'reorient_object'
        self.grip_open = [True, False, False]
        self.logger = logging.getLogger(__name__)
