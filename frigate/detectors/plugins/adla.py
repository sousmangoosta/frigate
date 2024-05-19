import logging

import numpy as np
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

from ctypes import *
import os
import time

logger = logging.getLogger(__name__)

DETECTOR_KEY = "adla"


class DetBox(Structure):
    _fields_ = [("ymin", c_float),
                ("xmin", c_float),
                ("ymax", c_float),
                ("xmax", c_float),
                ("score", c_float),
                ("objectClass", c_float)]


class ADLADetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]


class AdlaDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: ADLADetectorConfig):
        print(detector_config)
        self.detector_config = detector_config

        self.height = detector_config.model.height
        self.width = detector_config.model.width

        path = detector_config.model.path
        if path is None:
            raise ValueError("No model path provided")
        adla_path = os.path.splitext(path)[0] + ".adla"
        model_path = adla_path.encode('utf-8')
        if not path.endswith(".adla"):
            raise Exception(f"unknown model format {path}")
        try:
            so_file = "libadla_interface.so"
            self.interface = cdll.LoadLibrary(so_file)
        except OSError as e:
            logger.error(
                "ERROR: failed to load library. %s",
                e,
            )
        try:
            # void* init_network_file(const char *mpath)
            self.interface.init_network_file.argtypes = [c_char_p]
            self.interface.init_network_file.restype = c_void_p
            self.context = self.interface.init_network_file(model_path)
        except OSError as e:
            logger.error("ERROR: failed to initialize NPU with model %s: %s", model_path, e)

    def detect_raw(self, tensor_input):
        # void* init_network_file(const char *mpath)
        start_time = time.time()
        detect_start_time = start_time
        self.interface.set_input.argtypes = [c_void_p, POINTER(c_ubyte)]
        self.interface.set_input.restype = c_int
        ret = self.interface.set_input(self.context, tensor_input.ctypes.data_as(POINTER(c_ubyte)), tensor_input.size)
        stop_time = time.time()
        duration = stop_time - start_time
        logger.debug(f"set input returns {ret} after {duration} seconds")

        # int run_network(void *qcontext)
        start_time = time.time()
        self.interface.run_network.argtypes = [c_void_p, POINTER(c_uint32), POINTER(DetBox)]
        self.interface.run_network.restype = c_int
        box = np.zeros((6, 230), dtype=[('ymin', np.float32), ('xmin', np.float32), ('ymax', np.float32),
                                        ('xmax', np.float32), ('score', np.float32), ('objectClass', np.float32)])
        count = pointer(c_uint32(0))
        ret = self.interface.run_network(self.context, count, box.ctypes.data_as(POINTER(DetBox)))
        stop_time = time.time()
        duration = stop_time - start_time
        logger.debug(f"run network returns {ret} after {duration} seconds")

        detections = np.zeros((20, 6), np.float32)

        # if count[0] > 0:
        #     logger.debug(f"Number of object detected {count[0]}")
        #
        # for i in range(count[0]):
        #     if box['score'][0][i] < 0.4 or i == 20:
        #         break
        #     detections[i] = [
        #         box['objectClass'][0][i],
        #         float(box['score'][0][i]),
        #         box['ymin'][0][i],
        #         box['xmin'][0][i],
        #         box['ymax'][0][i],
        #         box['xmax'][0][i],
        #     ]
        #     #logger.debug(detections[i])
        #
        detect_stop_time = time.time()
        detect_duration = detect_stop_time - detect_start_time
        logger.debug(f"detect raw finished after {detect_duration} seconds")
        return detections
