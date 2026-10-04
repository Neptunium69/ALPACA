"""Shared depth projection for detections and camera point clouds."""
import cv2
import numpy as np


def to_meters(depth:np.ndarray,encoding:str)->np.ndarray:
    encoding=encoding.lower()
    if encoding=="16uc1":
        return depth.astype(np.float32)/1000.0
    if encoding=="32fc1":
        return depth.astype(np.float32)
    raise ValueError(f"Unsupported depth image format: {encoding}")


def pixel_points(pixels:np.ndarray,depths:np.ndarray,camera_info)->np.ndarray:
    """Project (u,v) pixels and metre depths into camera optical coordinates."""
    pixels=np.asarray(pixels,dtype=np.float64).reshape(-1,1,2)
    depths=np.asarray(depths,dtype=np.float64).reshape(-1)
    if len(pixels)!=len(depths) or not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(depths)) or np.any(depths<=0):
        raise ValueError("Pixels require matching finite, positive depths.")
    camera_matrix=np.asarray(camera_info.K,dtype=np.float64).reshape(3,3)
    fx,fy=camera_matrix[0,0],camera_matrix[1,1]
    if not np.all(np.isfinite(camera_matrix)) or fx<=0 or fy<=0:
        raise ValueError("Camera intrinsics require finite values and positive focal lengths.")
    distortion=np.asarray(camera_info.D,dtype=np.float64)
    if not np.all(np.isfinite(distortion)):
        raise ValueError("Camera distortion coefficients must be finite.")
    if not len(depths):
        return np.empty((0,3),dtype=np.float64)
    if distortion.size>0 and np.any(np.abs(distortion)>1e-12):
        if camera_info.distortion_model in ["plumb_bob","rational_polynomial"]:
            normalized=cv2.undistortPoints(pixels,camera_matrix,distortion).reshape(-1,2)
        elif camera_info.distortion_model=="equidistant":
            normalized=cv2.fisheye.undistortPoints(pixels,camera_matrix,distortion[:4].reshape(4,1)).reshape(-1,2)
        else:
            raise ValueError(f"Unsupported distortion model: {camera_info.distortion_model}")
    else:
        normalized=(pixels.reshape(-1,2)-camera_matrix[:2,2])/np.array([fx,fy])
    return np.column_stack((normalized*depths[:,None],depths))


def depth_points(depth:np.ndarray,encoding:str,camera_info)->np.ndarray:
    """Convert valid depth pixels using the existing detector's 0.1–10 m range."""
    depth=to_meters(depth,encoding)
    if depth.ndim!=2 or depth.shape!=(camera_info.height,camera_info.width):
        raise ValueError("CameraInfo and depth image dimensions do not match.")
    valid=np.isfinite(depth)&(depth>0.1)&(depth<10.0)
    v,u=np.nonzero(valid)
    return pixel_points(np.column_stack((u,v)),depth[valid],camera_info).astype(np.float32)
