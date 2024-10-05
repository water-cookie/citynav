import time
from typing import Literal, TypeAlias

import airsim
import numpy as np


from gsamllavanav.space import Pose4D


Perspective: TypeAlias = Literal['survey', 'slanted', 'front']


class AirsimClient:
    ''' an interface to interact with AirSim '''

    def __init__(self, ip: str, port: int, perspective: Perspective):
        '''initializes settings used for all episodes'''
        self.ip = ip  # 192.168.10.185
        self.port = port  # 41451
        self.perspective = perspective
        self.map_name = None
        self.client = airsim.VehicleClient(ip=self.ip, port=self.port)
        self.camera_id = "3" if perspective == 'survey' else "0"

    
    def get_rgbd(self, map_name: str, pose: Pose4D, rgb_size: tuple[int, int], depth_size: tuple[int, int]):

        rgb, depth = self._load_map(map_name)._set_pose(pose)._fetch_rgbd()

        assert rgb.shape[:2] == rgb_size and depth.shape[:2] == depth_size, "AirSim image resolution cannot be changed during runtime. Edit AirSim settings file to adjust the resolution."

        return rgb, depth


    def _load_map(self, map_name: str):

        if self.map_name != map_name:
            self.map_name = map_name
            level_name =  self._map_name_to_level_name(map_name)
            self.client.simLoadLevel(level_name)
            self.client = airsim.VehicleClient(ip=self.ip, port=self.port)  # client stops responding after simloadlevel
            self._reset_camera()
        
            time.sleep(1)
        
        return self


    def _reset_camera(self):

        if self.perspective == 'survey':
            self.client.simSetCameraPose('0', airsim.Pose(airsim.Vector3r(0, 0, -100), airsim.to_quaternion(-np.pi/2, 0, 0)))
        if self.perspective == 'front':
            self.client.simSetCameraPose('0', airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)))
        if self.perspective == 'slanted':
            self.client.simSetCameraPose('0', airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-np.pi/4, 0, 0)))
        
        return self


    def _set_pose(self, pose: Pose4D, pitch=0):

        x, y, z, yaw = pose
        
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(x, -y, -z),  # y & z axes are inverted in UE5 somehow
            airsim.to_quaternion(pitch, 0, -yaw)),
            ignore_collision=True
        )

        return self


    def _fetch_rgbd(self):

        image_requests = [
            airsim.ImageRequest(self.camera_id, airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest(self.camera_id, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
        ]
        
        for attempts in range(10):
            responses = self.client.simGetImages(image_requests)
            
            if responses[0].image_data_uint8:  # response contains image data
                return self._transform_rgb_image(responses[0]), self._transform_depth_image(responses[1])
        
        raise FailedRequestError(self.map_name, attempts)


    def _fetch_rgb(self):

        image_requests = [
            airsim.ImageRequest(self.camera_id, airsim.ImageType.Scene, pixels_as_float=False, compress=False),
        ]
        
        for attempts in range(10):
            responses = self.client.simGetImages(image_requests)
            
            if responses[0].image_data_uint8:  # response contains image data
                return self._transform_rgb_image(responses[0])
        
        raise FailedRequestError(self.map_name, attempts)


    @staticmethod
    def _transform_rgb_image(response: airsim.ImageResponse):
        
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # read 1D uint8 arr from binary string literal
        img_bgr = img1d.reshape(response.height, response.width, 3)
        img_rgb = img_bgr[..., ::-1]
        
        return img_rgb


    @staticmethod
    def _transform_depth_image(response: airsim.ImageResponse):
        
        img_depth = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        img_depth = img_depth[::-1, ]  # flip vertically
        
        return img_depth.astype(np.float32)


    @staticmethod
    def _map_name_to_level_name(map_name: str):
        '''e.g., "birmingham_block_2" -> "b2"'''
        return map_name[0] + map_name.split('_')[-1]


class FailedRequestError(Exception):
    '''raised when an API request has failed'''
    def __init__(self, map_name, attempts) -> None:
        super().__init__(f'request for {map_name} failed after {attempts}')
