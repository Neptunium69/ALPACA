from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from copy import deepcopy
from threading import RLock, Thread
import rospy
from sklearn.cluster import DBSCAN
import numpy as np
import torch
import warnings
from scipy.spatial.transform import Rotation
import message_filters
from .dataclasses import Function
from .wmtools.wmregistry import WMRegistry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import logging
import os
from pathlib import Path
from nav_msgs.msg import Odometry
from alpaca_msgs.msg import ObservationFrame, ObjectDetection,SpatialObservationFrame
from sklearn.neighbors import KNeighborsClassifier
from math import e
import json
import time
@dataclass
class Position:
    x: float
    y: float
    z: float

@dataclass
class Size:
    conservative_width: float | None = None
    conservative_height: float | None = None

@dataclass
class Orientation:
    pitch: float
    yaw: float
    roll: float

@dataclass
class Object:
    id: str
    class_name: str
    size: Size
    position: Position
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    detection_history:list[bool]=field(default_factory=list)

    def __post_init__(self):
        defaults = {
            "existence_rawscore": -2.0,
            "existence_score": 0.1192,
            "positioning_rawscore": -2.0,
            "positioning_score": 0.1192,
        }
        for key, value in defaults.items():
            self.confidence_scores.setdefault(key, value)

@dataclass
class Voxel:
    position: Position
    state: int | None
    rawscore: float = 0
    score: float = 0.5

@dataclass
class ObservationActivity:
    score: float = 0.0
    last_update: float = 0.0

@dataclass
class Model:
    objects: list[Object] = field(default_factory=list)

@dataclass
class VoxelModel:
    voxelsize: float
    voxels: list[Voxel] = field(default_factory=list)
    grid_origin: Position | None = None
    frame_id: str | None = None

class WorldModel():
    def __init__(self) -> None:
        self.submodels:list[Submodel]=[]
        self.registry:dict[str,WMRegistry]={}

    @property
    def modelnames(self)->list[str]:
        return [model.name for model in self.submodels]

    def add(self,models:list[Submodel])->None:
        for model in models:
            if model.name not in self.modelnames:
                self.submodels.append(model)
                self.registry[model.name]=model.registry
            else:
                warnings.warn(f"Submodel {model.name} was not added as another submodel of the same name is already in the World Model.")

    def get(self,name:str)->Submodel:
        for model in self.submodels:
            if model.name==name:
                return model
        raise KeyError(f"No submodel named {name!r} exists.")

    def remove(self,name:str)->None:
        self.submodels.remove(self.get(name))
        del self.registry[name]

    def initwm(self)->None:
        for model in self.submodels:
            model.initwm()

class Submodel(Protocol):
    name: str
    registry: WMRegistry

    def initwm(self)->None: ...

class VoxelWorldModel(Submodel):
    """Sensor-independent voxel storage, spatial queries and observation activity."""
    def __init__(self,name:str="voxels_master"):
        self.name=name
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.registry=WMRegistry([
            Function(name="get_observation_novelty",arguments={"position":("Position",True),"arrival_seconds":("float",False)}),
            Function(name="get_voxel_map_frame",arguments={}),
            Function(name="get_voxel_size",arguments={}),
            Function(name="get_voxel_snapshot",arguments={}),
            Function(name="cluster_voxels",arguments={"indices":("set",True),"voxel_size":("float",True),"gap":("float",True),"min_samples":("int",False)}),
            Function(name="world_position_to_voxel",arguments={"position":("Position",True)}),
            Function(name="voxel_to_position",arguments={"index":("tuple",True)}),
            Function(name="get_voxel_states",arguments={"indices":("list",True)}),
            Function(name="states_scores",arguments={"indices":("list",True)}),
            Function(name="get_vehicle_voxels",arguments={"vehicle_pose":("Odometry",True),"vehicle_min":("ndarray",True),"vehicle_max":("ndarray",True)}),
            Function(name="footprint_at",arguments={"position":("Position",True),"orientation":("Orientation",True),"vehicle_min":("ndarray",True),"vehicle_max":("ndarray",True),"samples":("ndarray",False)}),
            Function(name="get_unknown_score",arguments={"minimum":("Position",True),"maximum":("Position",True)}),
            Function(name="get_filled_score",arguments={"minimum":("Position",True),"maximum":("Position",True)}),
            Function(name="get_empty_score",arguments={"minimum":("Position",True),"maximum":("Position",True)}),
        ])
        self.voxelmap = VoxelModel(voxelsize=float("nan"), voxels=[], grid_origin=None, frame_id=None)
        self._voxel_lookup: dict[tuple[int,int,int],Voxel]={}
        # Separate, coarser voxel layer: observation origins are not occupancy evidence.
        self._observation_field: dict[tuple[int,int,int],ObservationActivity]={}
        self.activity_voxel_size=0.25
        self.activity_radius=0.5
        self.activity_decay_seconds=30.0
        self.activity_increment=0.1
        self.correlation_strength=1.0
        self._observation_listeners=[]
        self._lock=RLock()

    def initwm(self)->None:
        """The base voxel model has no sensor subscriptions."""
        pass

    def get_voxels(self):
        with self._lock:
            return self._voxel_lookup

    def get_voxel_snapshot(self)->tuple[dict[tuple[int,int,int],int | None],float]:
        """Copy voxel states and grid size together for a consistent route assessment."""
        with self._lock:
            return {index:voxel.state for index,voxel in self._voxel_lookup.items()},self.get_voxel_size()

    @staticmethod
    def sigmoid(x):
        return 1/(1+e**-x)

    def _observation_activity(self,position:Position,now:float,deposit:bool=False)->float:
        """Read decayed activity, optionally depositing one scan afterward. Caller holds _lock."""
        point=np.array([position.x,position.y,position.z],dtype=np.float64)
        radius=3.0*self.activity_radius
        lower=np.floor((point-radius)/self.activity_voxel_size).astype(np.int64)
        upper=np.floor((point+radius)/self.activity_voxel_size).astype(np.int64)
        indices=np.indices(tuple(upper-lower+1)).reshape(3,-1).T+lower
        distances_squared=np.sum(((indices+0.5)*self.activity_voxel_size-point)**2,axis=1)
        inside=distances_squared<=radius**2
        indices=indices[inside]
        weights=np.exp(-distances_squared[inside]/(2.0*self.activity_radius**2))
        activity=0.0
        for index,weight in zip(indices,weights):
            key=(int(index[0]),int(index[1]),int(index[2]))
            cell=self._observation_field.get(key)
            decayed=0.0 if cell is None else cell.score*math.exp(-max(0.0,now-cell.last_update)/self.activity_decay_seconds)
            activity+=float(weight)*decayed
            if deposit:
                self._observation_field[key]=ObservationActivity(score=decayed+self.activity_increment*float(weight),last_update=now)
        return activity/float(weights.sum()) if weights.size else 0.0

    def get_observation_novelty(self,position:Position,arrival_seconds:float=0.0)->float:
        """Predict novelty without changing the field or its timestamps."""
        if not all(math.isfinite(value) for value in (position.x,position.y,position.z,arrival_seconds)) or arrival_seconds<0:
            raise ValueError("Observation position and non-negative arrival time must be finite.")
        with self._lock:
            activity=self._observation_activity(position,time.monotonic()+arrival_seconds)
            return math.exp(-self.correlation_strength*activity)

    def position_to_voxel(self,position: np.ndarray,grid_origin: np.ndarray,voxel_size:float) -> tuple[int,int,int]:
        index=np.floor((position-grid_origin)/voxel_size).astype(np.int32)
        return int(index[0]),int(index[1]),int(index[2])

    def get_vehicle_voxels(self,vehicle_pose:Odometry,vehicle_min: np.ndarray,vehicle_max: np.ndarray)->list[tuple[int,int,int]]:
        """Obtain the current occupied voxels of the vehicle"""
        if vehicle_pose is None:
            raise ValueError("Vehicle pose has not been initialized.")
        pose=vehicle_pose.pose.pose
        translation=np.array([pose.position.x,pose.position.y,pose.position.z],dtype=np.float64)
        quaternion=np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w],dtype=np.float64)
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion)==0:
            raise ValueError("Vehicle pose must contain a finite position and a non-zero finite quaternion.")
        rotation=Rotation.from_quat(quaternion).as_matrix()
        return self._footprint(translation,rotation,vehicle_min,vehicle_max)

    def footprint_at(self,position:Position,orientation:Orientation,vehicle_min:np.ndarray,vehicle_max:np.ndarray,samples:np.ndarray | None=None)->list[tuple[int,int,int]]:
        """Obtain voxels at one pose or union axis-aligned path samples."""
        pose_values=np.array([position.x,position.y,position.z,orientation.roll,orientation.pitch,orientation.yaw],dtype=np.float64)
        if not np.all(np.isfinite(pose_values)):
            raise ValueError("Target position and orientation must contain only finite values.")
        translation=pose_values[:3]
        rotation=Rotation.from_euler("xyz",[orientation.roll,orientation.pitch,orientation.yaw]).as_matrix()
        if samples is None:
            return self._footprint(translation,rotation,vehicle_min,vehicle_max)
        if np.any(pose_values[3:]!=0):
            raise ValueError("Batched path footprints require zero orientation.")
        samples=np.asarray(samples,dtype=np.float64)
        vehicle_min=np.asarray(vehicle_min,dtype=np.float64)
        vehicle_max=np.asarray(vehicle_max,dtype=np.float64)
        if samples.ndim!=2 or samples.shape[1]!=3 or not np.all(np.isfinite(samples)):
            raise ValueError("Path samples must contain finite XYZ coordinates.")
        if vehicle_min.shape!=(3,) or vehicle_max.shape!=(3,) or not np.all(np.isfinite(vehicle_min)) or not np.all(np.isfinite(vehicle_max)) or np.any(vehicle_min>=vehicle_max):
            raise ValueError("Invalid vehicle dimensions.")
        with self._lock:
            voxel_size=self.voxelmap.voxelsize
            grid_origin=deepcopy(self.voxelmap.grid_origin)
        if not math.isfinite(voxel_size) or voxel_size<=0 or grid_origin is None:
            raise ValueError("Voxel grid has not been initialized.")
        if not len(samples):
            return []
        translations=torch.as_tensor(samples,dtype=torch.float64,device=self.device)
        minimum=torch.as_tensor(vehicle_min,dtype=torch.float64,device=self.device)
        maximum=torch.as_tensor(vehicle_max,dtype=torch.float64,device=self.device)
        origin=torch.tensor([grid_origin.x,grid_origin.y,grid_origin.z],dtype=torch.float64,device=self.device)
        lower=torch.floor((translations+minimum-origin)/voxel_size).long()
        upper=torch.ceil((translations+maximum-origin)/voxel_size).long()
        # Preserve _footprint's strict overlap comparisons at floating-point boundaries.
        lower+=((origin+(lower.to(torch.float64)+0.5)*voxel_size-translations)+voxel_size/2.0<=minimum).long()
        upper-=((origin+(upper.to(torch.float64)-0.5)*voxel_size-translations)-voxel_size/2.0>=maximum).long()
        boxes=torch.unique(torch.cat((lower,upper),dim=1),dim=0)
        boxes=boxes[torch.all(boxes[:,3:]>boxes[:,:3],dim=1)]
        if not boxes.numel():
            return []
        shape=(boxes[:,3:]-boxes[:,:3]).amax(dim=0).cpu().tolist()
        offsets=torch.cartesian_prod(*[torch.arange(size,device=self.device) for size in shape])
        grids=boxes[:,None,:3]+offsets[None,:,:]
        indices=grids[torch.all(grids<boxes[:,None,3:],dim=2)]
        indices=torch.unique(indices,dim=0)
        return [(x,y,z) for x,y,z in indices.cpu().tolist()]

    def _footprint(self,translation:np.ndarray,rotation:np.ndarray,vehicle_min:np.ndarray,vehicle_max:np.ndarray)->list[tuple[int,int,int]]:
        """Obtain the vehicle's occupied voxels after performing a transformation."""
        vehicle_min=np.asarray(vehicle_min,dtype=np.float64)
        vehicle_max=np.asarray(vehicle_max,dtype=np.float64)
        if vehicle_min.shape!=(3,) or vehicle_max.shape!=(3,) or not np.all(np.isfinite(vehicle_min)) or not np.all(np.isfinite(vehicle_max)) or np.any(vehicle_min>=vehicle_max):
            raise ValueError("Invalid vehicle dimensions.")
        translation=np.asarray(translation,dtype=np.float64)
        rotation=np.asarray(rotation,dtype=np.float64)
        if translation.shape!=(3,) or rotation.shape!=(3,3) or not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
            raise ValueError("Invalid vehicle pose transform.")

        with self._lock:
            voxel_size=self.voxelmap.voxelsize
            grid_origin=self.voxelmap.grid_origin

        if not math.isfinite(voxel_size) or voxel_size<=0 or grid_origin is None:
            raise ValueError("Voxel grid has not been initialized.")

        origin=np.array([grid_origin.x,grid_origin.y,grid_origin.z],dtype=np.float64)

        body_corners=np.array([[x,y,z] for x in (vehicle_min[0],vehicle_max[0]) for y in (vehicle_min[1],vehicle_max[1]) for z in (vehicle_min[2],vehicle_max[2])],dtype=np.float64)
        world_corners=body_corners@rotation.T+translation
        minimum_index=np.floor((world_corners.min(axis=0)-origin)/voxel_size).astype(np.int64)
        maximum_index=np.ceil((world_corners.max(axis=0)-origin)/voxel_size).astype(np.int64)

        voxel_half_extent=np.full(3,voxel_size/2.0,dtype=np.float64)
        body_frame_padding=np.abs(rotation.T)@voxel_half_extent
        axes=[torch.arange(int(minimum_index[i]),int(maximum_index[i]),device=self.device) for i in range(3)]
        grid=torch.cartesian_prod(*axes)

        origin_t=torch.as_tensor(origin,dtype=torch.float64,device=self.device)
        translation_t=torch.as_tensor(translation,dtype=torch.float64,device=self.device)
        rotation_t=torch.as_tensor(rotation,dtype=torch.float64,device=self.device)
        padding_t=torch.as_tensor(body_frame_padding,dtype=torch.float64,device=self.device)
        minimum_t=torch.as_tensor(vehicle_min,dtype=torch.float64,device=self.device)
        maximum_t=torch.as_tensor(vehicle_max,dtype=torch.float64,device=self.device)

        centres_world=origin_t+(grid.to(torch.float64)+0.5)*voxel_size
        centres_body=(centres_world-translation_t)@rotation_t
        overlaps=torch.all(centres_body+padding_t>minimum_t,dim=1)&torch.all(centres_body-padding_t<maximum_t,dim=1)
        indices=[(x,y,z) for x,y,z in grid[overlaps].cpu().tolist()]

        return indices

    def get_voxel_map_frame(self)->str | None:
        with self._lock:
            return self.voxelmap.frame_id

    def get_voxel_size(self)->float:
        with self._lock:
            voxel_size=self.voxelmap.voxelsize
        if not math.isfinite(voxel_size) or voxel_size<=0:
            raise ValueError("Voxel grid has not been initialized.")
        return voxel_size

    def world_position_to_voxel(self,position:Position)->tuple[int,int,int]:
        """Convert a world-frame position to an index in the active voxel map."""
        if not isinstance(position,Position):
            raise TypeError("position must be a Position.")
        coordinates=np.array([position.x,position.y,position.z],dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("World position must contain only finite values.")
        with self._lock:
            voxel_size=self.voxelmap.voxelsize
            grid_origin=deepcopy(self.voxelmap.grid_origin)
        if not math.isfinite(voxel_size) or voxel_size<=0 or grid_origin is None:
            raise ValueError("Voxel map has not been initialized.")
        origin=np.array([grid_origin.x,grid_origin.y,grid_origin.z],dtype=np.float64)
        return self.position_to_voxel(coordinates,origin,voxel_size)

    def voxel_to_position(self,index:tuple[int,int,int])->Position:
        if len(index)!=3 or any(not isinstance(component,int) or isinstance(component,bool) for component in index):
            raise ValueError("Voxel index must contain exactly three integers.")
        with self._lock:
            voxel_size=self.voxelmap.voxelsize
            grid_origin=deepcopy(self.voxelmap.grid_origin)
        if not math.isfinite(voxel_size) or voxel_size<=0 or grid_origin is None:
            raise ValueError("Voxel map has not been initialized.")
        origin=np.array([grid_origin.x,grid_origin.y,grid_origin.z],dtype=np.float64)
        center=origin+(np.asarray(index,dtype=np.float64)+0.5)*voxel_size
        return Position(x=float(center[0]),y=float(center[1]),z=float(center[2]))

    def get_voxel_states(self,indices:list[tuple[int,int,int]])->list[int | None]:
        with self._lock:
            states=[]
            for index in indices:
                voxel=self._voxel_lookup.get(index)
                states.append(None if voxel is None else voxel.state)
            return states

    def states_scores(self,indices:list[tuple[int,int,int]])->list[tuple[int | None,float] | None]:
        """Return atomic occupancy state/confidence snapshots for the requested voxels."""
        with self._lock:
            return [None if (voxel:=self._voxel_lookup.get(index)) is None else (voxel.state,voxel.score) for index in indices]

    def _get_voxel_regions(self,minimum: Position,maximum: Position)->list[int]:
        """Return [filled, empty, unknown] voxel counts within a world-frame bounding box."""
        coordinates=np.array([minimum.x,minimum.y,minimum.z,maximum.x,maximum.y,maximum.z],dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Bounding-box coordinates must be finite.")
        minimum_array=coordinates[:3]
        maximum_array=coordinates[3:]
        if np.any(minimum_array>=maximum_array):
            raise ValueError("Bounding-box minimum must be smaller than its maximum.")

        with self._lock:
            voxel_size=self.voxelmap.voxelsize
            grid_origin=self.voxelmap.grid_origin
            if not math.isfinite(voxel_size) or voxel_size<=0 or grid_origin is None:
                raise ValueError("Voxel-grid size and origin have not been initialized.")

            origin_array=np.array([grid_origin.x,grid_origin.y,grid_origin.z],dtype=np.float64)
            minimum_index=np.floor((minimum_array-origin_array)/voxel_size).astype(np.int64)
            maximum_index=np.ceil((maximum_array-origin_array)/voxel_size).astype(np.int64)
            region_shape=maximum_index-minimum_index
            total_voxels=int(np.prod(region_shape,dtype=np.int64))
            known_voxels=[voxel for index,voxel in self._voxel_lookup.items() if minimum_index[0]<=index[0]<maximum_index[0] and minimum_index[1]<=index[1]<maximum_index[1] and minimum_index[2]<=index[2]<maximum_index[2]]
            filled=sum(voxel.state==1 for voxel in known_voxels)
            empty=sum(voxel.state==0 for voxel in known_voxels)
            unknown=total_voxels-filled-empty
            return [filled,empty,unknown]

    def get_unknown_score(self,minimum:Position,maximum:Position)->float:
        voxel_regions=self._get_voxel_regions(minimum,maximum)
        score=voxel_regions[2]/(voxel_regions[0]+voxel_regions[1]+voxel_regions[2])
        return score

    def get_filled_score(self,minimum:Position,maximum:Position)->float:
        voxel_regions=self._get_voxel_regions(minimum,maximum)
        score=voxel_regions[0]/(voxel_regions[0]+voxel_regions[1]+voxel_regions[2])
        return score

    def get_empty_score(self,minimum:Position,maximum:Position)->float:
        voxel_regions=self._get_voxel_regions(minimum,maximum)
        score=voxel_regions[1]/(voxel_regions[0]+voxel_regions[1]+voxel_regions[2])
        return score
    
    @staticmethod
    def cluster_voxels(indices:set[tuple[int,int,int]],voxel_size:float,gap:float,min_samples:int=3):
        if not indices:
            return [],set()

        keys=list(indices)
        points=np.asarray(keys,dtype=np.float64)*voxel_size
        labels=DBSCAN(eps=gap,min_samples=min_samples).fit_predict(points)

        groups={}
        unresolved=set()
        for index,label in zip(keys,labels):
            if label==-1:
                unresolved.add(index)
            else:
                groups.setdefault(int(label),set()).add(index)

        return list(groups.values()),unresolved

class MasterVoxelWorldModel(VoxelWorldModel):
    """Syncronizes the two world models"""
    def __init__(self,wms:list[VoxelWorldModel],frequency:float=5.0,important_wms:tuple[str,str]=("voxels_camera","voxels_lidar")):
        super().__init__(name="voxels_master")
        self.vwms:dict[str,VoxelWorldModel]={}
        self.period=1/frequency
        self._locker=RLock()
        self.lr=0.1

        self.important_wms=important_wms
        self.thread=Thread(target=self._syncronization_callback,name="voxel-fusion",daemon=True)
        for wm in wms:
            if wm.name not in [wmname for wmname in self.vwms]:
                self.vwms[wm.name]=wm
            else:
                raise RuntimeError(f"World model {wm.name} is already added.")
        self.observation_pair_slop=0.1
        self.observation_pairs=0
        self._pending_observations={name:[] for name in important_wms}
        self._last_activity_stamps={}
        self._activity_last_time=0.0
        for name in important_wms:
            if name in self.vwms:
                self.vwms[name]._observation_listeners.append(lambda message,now,name=name:self.record_observation(name,message,now))

    def record_observation(self,sensor:str,message:SpatialObservationFrame,now:float)->None:
        """Deposit once per accepted camera/LiDAR pair, never per fusion iteration."""
        if sensor not in self._pending_observations or len(self._pending_observations)!=2:
            return
        stamp=message.header.stamp.to_nsec(); position=message.sensor_position
        if stamp<=0 or not message.header.frame_id or not all(math.isfinite(value) for value in (now,position.x,position.y,position.z)):
            return
        with self._lock:
            key=(sensor,message.source_id)
            if stamp<=self._last_activity_stamps.get(key,-1):
                return
            self._last_activity_stamps[key]=stamp
            other=next(name for name in self._pending_observations if name!=sensor); pending=self._pending_observations[other]
            matches=[(abs(item[0]-stamp),index) for index,item in enumerate(pending) if item[1]==message.header.frame_id and abs(item[0]-stamp)<=round(self.observation_pair_slop*1e9)]
            if not matches:
                queue=self._pending_observations[sensor]; queue.append((stamp,message.header.frame_id,Position(position.x,position.y,position.z),now)); del queue[:-100]
                return
            _,index=min(matches); _,_,previous,received=pending.pop(index)
            origin=Position((previous.x+position.x)/2,(previous.y+position.y)/2,(previous.z+position.z)/2)
            now=max(now,received,self._activity_last_time)
            self._observation_activity(origin,now,deposit=True)
            self._activity_last_time=now; self.observation_pairs+=1

    def initwm(self):
        if not self.thread.is_alive():
            self.thread.start()
    def _syncronization_callback(self):
        deadline=time.monotonic()+self.period
        while True:
            with self._locker:
                # 2 world models are used for first prototype. Later versions will generalize the algorithm to any number of voxel world models.
                with self.vwms[self.important_wms[0]]._lock:
                    # Snapshot immutable scores; retain every voxel without copying its objects.
                    camera_scores={index:voxel.score for index,voxel in self.vwms[self.important_wms[0]]._voxel_lookup.items()}
                    grid: VoxelModel=self.vwms[self.important_wms[0]].voxelmap
                    camgrid: tuple[str | None, Position | None, float]=deepcopy((grid.frame_id,grid.grid_origin,grid.voxelsize))
                with self.vwms[self.important_wms[1]]._lock:
                    lidar_scores={index:voxel.score for index,voxel in self.vwms[self.important_wms[1]]._voxel_lookup.items()}
                    grid=self.vwms[self.important_wms[1]].voxelmap
                    lidgrid: tuple[str | None, Position | None, float]=deepcopy((grid.frame_id,grid.grid_origin,grid.voxelsize))

                if any(not frame or origin is None or not math.isfinite(size) or size<=0 for frame,origin,size in (camgrid,lidgrid)):
                    time.sleep(self.period)
                    continue
                if camgrid!=lidgrid:
                    raise ValueError("Camera and LiDAR voxel grids have different frames, origins or resolutions.")
                with self._lock:
                    self.voxelmap.frame_id,self.voxelmap.grid_origin,self.voxelmap.voxelsize=camgrid

                _,grid_origin,voxel_size=camgrid
                # The grid validation above guarantees an initialized origin.
                assert grid_origin is not None
                voxels=camera_scores.keys()|lidar_scores.keys()
                for voxel in voxels:
                    camera_present=voxel in camera_scores
                    lidar_present=voxel in lidar_scores
                    camera_score=camera_scores.get(voxel,0.5)
                    lidar_score=lidar_scores.get(voxel,0.5)
                    master_voxel=self._voxel_lookup.get(voxel)
                    master_score=0.5 if master_voxel is None else master_voxel.score
                    fused_score=master_score+self.lr/2*((1-((camera_score-lidar_score)**2 if camera_present and lidar_present else 0))*((int(camera_present)+int(lidar_present))/2*(camera_score+lidar_score-1)-abs(camera_score+lidar_score-1)*(2*master_score-1))-(1-(1-((camera_score-lidar_score)**2 if camera_present and lidar_present else 0)))*(2*master_score-1))
                    with self._lock:
                        if master_voxel is None:
                            position=Position(x=grid_origin.x+voxel[0]*voxel_size,y=grid_origin.y+voxel[1]*voxel_size,z=grid_origin.z+voxel[2]*voxel_size)
                            master_voxel=Voxel(position=position,state=None)
                            self._voxel_lookup[voxel]=master_voxel
                            self.voxelmap.voxels.append(master_voxel)
                        master_voxel.score=fused_score
                        master_voxel.state=None if abs(fused_score-0.5)<=0.2 else int(fused_score>0.5)

            time.sleep(max(0.0,deadline-time.monotonic()))
            deadline+=self.period
            if deadline<=time.monotonic():
                deadline=time.monotonic()+self.period

class CameraVoxelWorldModel(VoxelWorldModel):
    """Populate a voxel model from the adapter's camera observations."""
    def __init__(self,name:str="voxels_camera",observation_topic:str | None="camera_voxel_observations"):
        super().__init__(name=name)
        self.observation_topic=observation_topic
        self.voxel_subscriber=None
        self.voxel_learning_rate=0.25
        self.maximum_voxel_rawscore=4.0
        self.voxel_unknown_margin=0.2
        self._last_voxel_stamps: dict[str,int]={}

    def initwm(self)->None:
        if self.observation_topic is not None and self.voxel_subscriber is None:
            self.voxel_subscriber=rospy.Subscriber(self.observation_topic,SpatialObservationFrame,callback=self._voxel_callback)

    def _voxel_callback(self, message: SpatialObservationFrame):
        voxel_size=float(message.voxel_size)
        origin_array=np.array([message.grid_origin.x,message.grid_origin.y,message.grid_origin.z],dtype=np.float64)
        frame_id=message.header.frame_id
        if not math.isfinite(voxel_size) or voxel_size<=0 or not np.all(np.isfinite(origin_array)):
            rospy.logerr("Received voxel map with an invalid voxel size or grid origin.")
            return
        if not frame_id:
            rospy.logerr("Received voxel map with an empty frame ID.")
            return
        sensor_position=getattr(message,"sensor_position",None)
        if sensor_position is None or not all(math.isfinite(value) for value in (sensor_position.x,sensor_position.y,sensor_position.z)):
            rospy.logerr("Voxel observations require a finite sensor_position; rebuild alpaca_msgs and restart both nodes.")
            return
        stamp=message.header.stamp.to_nsec()
        if stamp<=0:
            rospy.logwarn("Rejected voxel observations without a valid scan timestamp.")
            return

        with self._lock:
            if stamp<=self._last_voxel_stamps.get(message.source_id,-1):
                return
            if self.voxelmap.grid_origin is None:
                self.voxelmap.voxelsize=voxel_size
                self.voxelmap.grid_origin=Position(x=float(origin_array[0]),y=float(origin_array[1]),z=float(origin_array[2]))
                self.voxelmap.frame_id=frame_id
            else:
                stored_origin=np.array([self.voxelmap.grid_origin.x,self.voxelmap.grid_origin.y,self.voxelmap.grid_origin.z],dtype=np.float64)
                if not math.isclose(self.voxelmap.voxelsize,voxel_size,rel_tol=0.0,abs_tol=1e-9) or not np.allclose(stored_origin,origin_array,rtol=0.0,atol=1e-9) or self.voxelmap.frame_id!=frame_id:
                    rospy.logerr("Received voxel map whose size, origin, or frame differs from the established grid.")
                    return

            if not message.voxels:
                rospy.logwarn("Received empty voxel map message.")
                return

            now=time.monotonic()
            activity=self._observation_activity(sensor_position,now)
            novelty=math.exp(-self.correlation_strength*activity)
            updated=False
            for observation in message.voxels:
                index=(int(observation.x),int(observation.y),int(observation.z))
                occupied=int(observation.occupied_observations)
                free=int(observation.free_observations)
                if occupied<0 or free<0 or occupied+free==0:
                    continue
                evidence=(2.0*occupied-free)/(2.0*occupied+free)
                voxel=self._voxel_lookup.get(index)
                new_voxel=voxel is None
                if voxel is None:
                    position_array=origin_array+np.asarray(index,dtype=np.float64)*voxel_size
                    voxel=Voxel(position=Position(x=float(position_array[0]),y=float(position_array[1]),z=float(position_array[2])),state=None)
                    self._voxel_lookup[index]=voxel
                    self.voxelmap.voxels.append(voxel)
                new_obstacle=evidence>0 and voxel.state==0
                if new_obstacle:
                    # Repeated viewing must not leave newly blocked space classified as free.
                    voxel.rawscore=0.0
                alpha=self.voxel_learning_rate*(1.0 if new_voxel or new_obstacle else novelty)
                voxel.rawscore+=alpha*(self.maximum_voxel_rawscore*evidence-voxel.rawscore)
                voxel.rawscore=max(-self.maximum_voxel_rawscore,min(self.maximum_voxel_rawscore,voxel.rawscore))
                voxel.score=self.sigmoid(voxel.rawscore)
                voxel.state=None if abs(voxel.score-0.5)<=self.voxel_unknown_margin else int(voxel.score>0.5)
                updated=True
            if updated:
                self._observation_activity(sensor_position,now,deposit=True)
                self._last_voxel_stamps[message.source_id]=stamp
                for listener in self._observation_listeners: listener(message,now)

class LiDARVoxelWorldModel(VoxelWorldModel):
    """Populate a voxel model from the adapter's LiDAR observations."""
    def __init__(self,name:str="voxels_lidar",observation_topic:str | None="voxel_observations"):
        super().__init__(name=name)
        self.observation_topic=observation_topic
        self.voxel_subscriber=None
        self.voxel_learning_rate=0.25
        self.maximum_voxel_rawscore=4.0
        self.voxel_unknown_margin=0.2
        self._last_voxel_stamps: dict[str,int]={}

    def initwm(self)->None:
        if self.observation_topic is not None and self.voxel_subscriber is None:
            self.voxel_subscriber=rospy.Subscriber(self.observation_topic,SpatialObservationFrame,callback=self._voxel_callback)

    def _voxel_callback(self, message: SpatialObservationFrame):
        voxel_size=float(message.voxel_size)
        origin_array=np.array([message.grid_origin.x,message.grid_origin.y,message.grid_origin.z],dtype=np.float64)
        frame_id=message.header.frame_id
        if not math.isfinite(voxel_size) or voxel_size<=0 or not np.all(np.isfinite(origin_array)):
            rospy.logerr("Received voxel map with an invalid voxel size or grid origin.")
            return
        if not frame_id:
            rospy.logerr("Received voxel map with an empty frame ID.")
            return
        sensor_position=getattr(message,"sensor_position",None)
        if sensor_position is None or not all(math.isfinite(value) for value in (sensor_position.x,sensor_position.y,sensor_position.z)):
            rospy.logerr("Voxel observations require a finite sensor_position; rebuild alpaca_msgs and restart both nodes.")
            return
        stamp=message.header.stamp.to_nsec()
        if stamp<=0:
            rospy.logwarn("Rejected voxel observations without a valid scan timestamp.")
            return

        with self._lock:
            if stamp<=self._last_voxel_stamps.get(message.source_id,-1):
                return
            if self.voxelmap.grid_origin is None:
                self.voxelmap.voxelsize=voxel_size
                self.voxelmap.grid_origin=Position(x=float(origin_array[0]),y=float(origin_array[1]),z=float(origin_array[2]))
                self.voxelmap.frame_id=frame_id
            else:
                stored_origin=np.array([self.voxelmap.grid_origin.x,self.voxelmap.grid_origin.y,self.voxelmap.grid_origin.z],dtype=np.float64)
                if not math.isclose(self.voxelmap.voxelsize,voxel_size,rel_tol=0.0,abs_tol=1e-9) or not np.allclose(stored_origin,origin_array,rtol=0.0,atol=1e-9) or self.voxelmap.frame_id!=frame_id:
                    rospy.logerr("Received voxel map whose size, origin, or frame differs from the established grid.")
                    return

            if not message.voxels:
                rospy.logwarn("Received empty voxel map message.")
                return

            now=time.monotonic()
            activity=self._observation_activity(sensor_position,now)
            novelty=math.exp(-self.correlation_strength*activity)
            updated=False
            for observation in message.voxels:
                index=(int(observation.x),int(observation.y),int(observation.z))
                occupied=int(observation.occupied_observations)
                free=int(observation.free_observations)
                if occupied<0 or free<0 or occupied+free==0:
                    continue
                evidence=(2.0*occupied-free)/(2.0*occupied+free)
                voxel=self._voxel_lookup.get(index)
                new_voxel=voxel is None
                if voxel is None:
                    position_array=origin_array+np.asarray(index,dtype=np.float64)*voxel_size
                    voxel=Voxel(position=Position(x=float(position_array[0]),y=float(position_array[1]),z=float(position_array[2])),state=None)
                    self._voxel_lookup[index]=voxel
                    self.voxelmap.voxels.append(voxel)
                new_obstacle=evidence>0 and voxel.state==0
                if new_obstacle:
                    # Repeated viewing must not leave newly blocked space classified as free.
                    voxel.rawscore=0.0
                alpha=self.voxel_learning_rate*(1.0 if new_voxel or new_obstacle else novelty)
                voxel.rawscore+=alpha*(self.maximum_voxel_rawscore*evidence-voxel.rawscore)
                voxel.rawscore=max(-self.maximum_voxel_rawscore,min(self.maximum_voxel_rawscore,voxel.rawscore))
                voxel.score=self.sigmoid(voxel.rawscore)
                voxel.state=None if abs(voxel.score-0.5)<=self.voxel_unknown_margin else int(voxel.score>0.5)
                updated=True
            if updated:
                self._observation_activity(sensor_position,now,deposit=True)
                self._last_voxel_stamps[message.source_id]=stamp
                for listener in self._observation_listeners: listener(message,now)

    def _lidar_transform(self,observation: ObservationFrame,flag: int=0)->np.ndarray:
        """Return the body-to-world transform or its inverse for body-frame LiDAR points."""
        pose=observation.vehicle_pose.pose.pose
        q_WB=pose.orientation

        T_WB=np.eye(4,dtype=float)
        T_WB[:3,:3]=Rotation.from_quat([q_WB.x,q_WB.y,q_WB.z,q_WB.w]).as_matrix()
        T_WB[:3,3]=[pose.position.x,pose.position.y,pose.position.z]

        if flag==0:
            return T_WB
        if flag==1:
            return np.linalg.inv(T_WB)

        raise ValueError("flag must be 0 for body-to-world or 1 for world-to-body")

    def w2l(self,frame, position:np.ndarray | list[float] | tuple[float,float,float])->np.ndarray:
        """Convert a world-frame point to the LiDAR cloud's body frame."""
        position=np.append(np.asarray(position,dtype=float),1.0)
        transform=self._lidar_transform(frame,1)
        newpos=transform@position
        return newpos[:3]

    def l2w(self,frame, position:np.ndarray | list[float] | tuple[float,float,float])->np.ndarray:
        """Convert a body-frame LiDAR cloud point to the world frame."""
        position=np.append(np.asarray(position,dtype=float),1.0)
        transform=self._lidar_transform(frame,0)
        newpos=transform@position
        return newpos[:3]

class ObjectWorldModel(Submodel):
    def __init__(self, observations: Model, detection_classes, name:str="objects"):
        self.name=name
        self.registry=WMRegistry([
            Function(name="query_world",arguments={"object_id":("str",False),"class_name":("str",False)}),
            Function(name="write",arguments={"world_object":("Object",True)}),
            Function(name="bounding_box_to_world",arguments={"bounding_box":("list",True)}),
            Function(name="get_detection_classes",arguments={}),
            Function(name="record",arguments={}),
            Function(name="stop_record",arguments={}),
        ])
        self.observations = observations
        self.observationstemp=Model()
        self._last_detection_stamp=None
        self.detection_classes = tuple(detection_classes)
        self.association_distance=0.3
        self._bridge = CvBridge()
        self._latest_frame=None
        self._latest_depth=None
        self._lock = RLock()
        self._startstate=None


    def initwm(self):
        # Automatic world model updater
        log_directory=Path(os.environ.get("ALPACA_LOG_DIR","/tmp/alpaca_runtime"))
        log_directory.mkdir(parents=True,exist_ok=True)
        self.confirmation_logger=logging.getLogger(f"alpaca.worldmodel.confirmations.{id(self)}")
        self.confirmation_logger.setLevel(logging.INFO)
        self.confirmation_logger.propagate=False
        handler=logging.FileHandler(log_directory/"world_model.log",encoding="utf-8")
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        self.confirmation_logger.addHandler(handler)
        self.detection_subscriber=message_filters.Subscriber("detections", ObservationFrame)
        self.depth_subscriber=message_filters.Subscriber("camera_depth", Image)
        self.synchronizer=message_filters.ApproximateTimeSynchronizer([self.depth_subscriber,self.detection_subscriber], queue_size=60, slop=0.05)
        self.synchronizer.registerCallback(self._update_callback)


    @staticmethod
    def sigmoid(x):
        return 1/(1+e**-x)


    def _update_callback(self, depth, detections):
        try:
            detection_list=detections.detections
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            rospy.logerr("Invalid detection message: %s", error)
            return

        with self._lock:
            stamp=detections.header.stamp.to_nsec()
            if self._last_detection_stamp is not None and stamp<=self._last_detection_stamp:
                if detection_list:
                    rospy.loginfo("World-model rejected old frame: stamp_ns=%d last_stamp_ns=%d",stamp,self._last_detection_stamp)
                return
            self._latest_frame=detections
            self._latest_depth=depth
            try:
                working_objects = deepcopy(self.observations.objects)
                working_temp=deepcopy(self.observationstemp.objects)
                self._apply_detections(working_objects, detection_list, detections, depth, working_temp)
            except Exception as error:
                rospy.logerr("World-model update failed: %s", error)
                return

            self.observations.objects[:] = working_objects
            self.observationstemp.objects[:]=working_temp
            self._last_detection_stamp=stamp
            if detection_list or working_temp:
                self.confirmation_logger.info("World-model confirmation: %s",json.dumps({"stamp_ns":stamp,"detections":len(detection_list),"confirmed_ids":[obj.id for obj in working_objects],"temporary":[{"id":obj.id,"position":[obj.position.x,obj.position.y,obj.position.z],"history":obj.detection_history} for obj in working_temp]}))


    def _apply_detections(self, objects: list[Object], detection_list: list[ObjectDetection], frame: ObservationFrame, depthmap: Image, temp_objects:list[Object]) -> None:
        observed_list=[]
        for detection in detection_list:
            updated_object=None
            detected_size=None
            fx=frame.camera_info.K[0]
            fy=frame.camera_info.K[4]
            if detection.has_depth and detection.optical_depth>0 and fx>0 and fy>0:
                detected_width=((detection.x_max-detection.x_min)*detection.optical_depth)/fx
                detected_height=((detection.y_max-detection.y_min)*detection.optical_depth)/fy
                if detected_width>0 and detected_height>0:
                    detected_size=Size(detected_width,detected_height)

            if(detection.has_world_position):
                detection_position=np.array([detection.world_position.x,detection.world_position.y,detection.world_position.z],dtype=np.float64)
                same_class=[candidate for candidate in objects+temp_objects if candidate.class_name==detection.class_name]
                matched_object=min(same_class,key=lambda candidate:float(np.linalg.norm(detection_position-np.array([candidate.position.x,candidate.position.y,candidate.position.z],dtype=np.float64))),default=None)
                association_distance=self.association_distance
                if detection.has_depth and math.isfinite(detection.camera_range):
                    association_distance=max(association_distance,min(0.30,0.05*detection.camera_range))
                if matched_object is not None:
                    matched_distance=float(np.linalg.norm(detection_position-np.array([matched_object.position.x,matched_object.position.y,matched_object.position.z],dtype=np.float64)))
                    if matched_distance>association_distance:
                        matched_object=None

                if matched_object is not None:
                    rawscore=matched_object.confidence_scores.get("existence_rawscore",-2.0)+0.1*detection.confidence
                    matched_object.confidence_scores["existence_rawscore"]=rawscore
                    matched_object.confidence_scores["existence_score"]=self.sigmoid(rawscore)
                    updated_object=matched_object
                    observed_list.append(matched_object.id)
                    if detection.has_depth:
                        distance_scaled=detection.camera_range/10
                        positioning_rawscore=matched_object.confidence_scores.get("positioning_rawscore",-2.0)+0.1*((1-distance_scaled)**2+0.05)
                        positioning_score=self.sigmoid(positioning_rawscore)
                        matched_object.confidence_scores["positioning_rawscore"]=positioning_rawscore
                        matched_object.confidence_scores["positioning_score"]=positioning_score
                        matched_object.position.x=(matched_object.position.x*positioning_score)+((1-positioning_score)*detection.world_position.x)
                        matched_object.position.y=(matched_object.position.y*positioning_score)+((1-positioning_score)*detection.world_position.y)
                        matched_object.position.z=(matched_object.position.z*positioning_score)+((1-positioning_score)*detection.world_position.z)
                        if detected_size is not None:
                            old_width=matched_object.size.conservative_width
                            old_height=matched_object.size.conservative_height
                            new_width=detected_size.conservative_width
                            new_height=detected_size.conservative_height
                            if old_width is None or old_height is None or new_width is None or new_height is None:
                                matched_object.size=detected_size
                            else:
                                matched_object.size.conservative_width=(old_width*(1-detection.confidence))+(detection.confidence*new_width)
                                matched_object.size.conservative_height=(old_height*(1-detection.confidence))+(detection.confidence*new_height)
                else:
                    id=f"{detection.class_name}_{self._get_latest(detection.class_name, objects+temp_objects)+1}"
                    updated_object=Object(id=id,class_name=detection.class_name,size=detected_size if detected_size is not None else Size(),position=Position(detection.world_position.x,detection.world_position.y,detection.world_position.z),confidence_scores={"existence_rawscore":-2.0,"existence_score":self.sigmoid(-2.0)})
                    temp_objects.append(updated_object)
                    observed_list.append(id)

                # Prevents the rawscore form being too large, which may cause world model updates to be slow if the object's position is changed.
                if updated_object is not None and updated_object.confidence_scores["existence_rawscore"]>5.0:
                    updated_object.confidence_scores["existence_rawscore"]=5.0
                    updated_object.confidence_scores["existence_score"]=self.sigmoid(5.0)
                if updated_object is not None and updated_object.confidence_scores["positioning_rawscore"]>5.0:
                    updated_object.confidence_scores["positioning_rawscore"]=5.0
                    updated_object.confidence_scores["positioning_score"]=self.sigmoid(5.0)

        observed_ids=set(observed_list)
        for obj in temp_objects:
            obj.detection_history=(obj.detection_history+[obj.id in observed_ids])[-5:]
            if sum(obj.detection_history)>=3:
                objects.append(obj)
        temp_objects[:]=[obj for obj in temp_objects if sum(obj.detection_history)<3 and obj.detection_history!=[False]*5]

        for object in objects:
            if object.attributes.get("VLMFlag")==1:
                observed_list.append(object.id)

        # First check distance
        unobserved_objects=[]
        campos=self._camera_transform(frame,0)[:3,3]
        for object in objects:
            if object.id not in observed_list:
                if math.sqrt(((object.position.x-campos[0])**2)+((object.position.y-campos[1])**2)+((object.position.z-campos[2])**2))<=10.0:
                    unobserved_objects.append(object)

        caminfo=frame.camera_info

        fx = caminfo.K[0]
        fy = caminfo.K[4]
        cx = caminfo.K[2]
        cy = caminfo.K[5]

        unobserved_objects_new=[]

        for world_object in unobserved_objects:
            objpos_c=self.w2c(frame,[world_object.position.x,world_object.position.y,world_object.position.z])
            z=objpos_c[2]
            if(z<=0):
                continue

            u=fx*(objpos_c[0]/z)+cx
            v=fy*(objpos_c[1]/z)+cy
            center_inside_margin=(20<u<(caminfo.width-20)) and (20<v<(caminfo.height-20))
            if(not center_inside_margin):
                continue

            conservative_width=world_object.size.conservative_width
            conservative_height=world_object.size.conservative_height
            if(conservative_width is None or conservative_height is None or conservative_width<=0 or conservative_height<=0):
                continue

            pred_w_obj=(fx*conservative_width)/z
            pred_h_obj=(fy*conservative_height)/z

            x1=u-(pred_w_obj/2)
            x2=u+(pred_w_obj/2)
            y1=v-(pred_h_obj/2)
            y2=v+(pred_h_obj/2)

            x1_i=max(0, int(math.floor(x1)))
            x2_i=min(caminfo.width, int(math.ceil(x2)))
            y1_i=max(0, int(math.floor(y1)))
            y2_i=min(caminfo.height, int(math.ceil(y2)))

            intersection_w=max(0.0,min(x2_i,caminfo.width)-max(x1_i,0.0))
            intersection_h=max(0.0,min(y2_i,caminfo.height)-max(y1_i,0.0))
            visible_fraction=(intersection_w*intersection_h)/(pred_w_obj*pred_h_obj)

            if(visible_fraction<0.8):
                continue

            depthmap_data=np.asarray(self._bridge.imgmsg_to_cv2(depthmap,desired_encoding="passthrough"),dtype=np.float32)
            if depthmap.encoding=="16UC1":
                depthmap_data=depthmap_data/1000.0
            elif depthmap.encoding!="32FC1":
                raise ValueError(f"Unsupported depth image encoding: {depthmap.encoding}")

            depth_crop=depthmap_data[y1_i:y2_i,x1_i:x2_i]
            depthmap_new=np.full(depth_crop.shape,3,dtype=np.uint8)
            valid_mask=np.isfinite(depth_crop) & (depth_crop>0)
            depthmap_new[valid_mask & (depth_crop<(z-0.2))]=0
            depthmap_new[valid_mask & (depth_crop>=(z-0.2)) & (depth_crop<=(z+0.2))]=1
            depthmap_new[valid_mask & (depth_crop>(z+0.2))]=2
                
            rows,cols=np.indices(depthmap_new.shape)
            known_mask=depthmap_new!=3
            unknown_mask=depthmap_new==3

            if known_mask.any() and unknown_mask.any():
                known_coordinates=np.column_stack((rows[known_mask],cols[known_mask]))
                known_categories=depthmap_new[known_mask]
                unknown_coordinates=np.column_stack((rows[unknown_mask],cols[unknown_mask]))

                neighbour_count=min(5,len(known_categories))
                imputer=KNeighborsClassifier(n_neighbors=neighbour_count,weights="distance")
                imputer.fit(known_coordinates,known_categories)
                depthmap_new[unknown_mask]=imputer.predict(unknown_coordinates)

            if not known_mask.any():
                continue

            closer=0
            samedistance=0
            further=0

            for i in depthmap_new:
                for j in i:
                    if j==0:
                        closer+=1
                    elif j==1:
                        samedistance+=1
                    elif j==2:
                        further+=1

            classified_pixels=closer+samedistance+further
            if classified_pixels==0:
                continue
            fnear=(closer+samedistance)/classified_pixels
            if(fnear>=0.3):
                continue
            
            unobserved_objects_new.append(world_object)

        for object in unobserved_objects_new:
            object.confidence_scores["positioning_rawscore"]-=0.1
            object.confidence_scores["existence_rawscore"]-=0.1
            object.confidence_scores["positioning_score"]=self.sigmoid(object.confidence_scores["positioning_rawscore"])
            object.confidence_scores["existence_score"]=self.sigmoid(object.confidence_scores["existence_rawscore"])

        objects[:]=[object for object in objects if object.confidence_scores["existence_score"]>=0.08]


    def _camera_transform(self,observation: ObservationFrame,flag: int=0)->np.ndarray:
        pose=observation.vehicle_pose.pose.pose
        q_WB=pose.orientation

        T_WB=np.eye(4,dtype=float)
        T_WB[:3,:3]=Rotation.from_quat([q_WB.x,q_WB.y,q_WB.z,q_WB.w]).as_matrix()
        T_WB[:3,3]=[pose.position.x,pose.position.y,pose.position.z]

        camera_transform=observation.camera_pose_in_body.transform
        q_BC=camera_transform.rotation

        T_BC=np.eye(4,dtype=float)
        T_BC[:3,:3]=Rotation.from_quat([q_BC.x,q_BC.y,q_BC.z,q_BC.w]).as_matrix()
        T_BC[:3,3]=[camera_transform.translation.x,camera_transform.translation.y,camera_transform.translation.z]

        T_WC=T_WB@T_BC

        if flag==0:
            return T_WC
        if flag==1:
            return np.linalg.inv(T_WC)

        raise ValueError("flag must be 0 for camera-to-world or 1 for world-to-camera")


    def w2c(self,frame, position:np.ndarray | list[float] | tuple[float,float,float])->np.ndarray:
        position=np.append(np.asarray(position,dtype=float),1.0)
        transform=self._camera_transform(frame,1)
        newpos=transform@position
        return newpos[:3]


    def c2w(self,frame, position:np.ndarray | list[float] | tuple[float,float,float])->np.ndarray:
        position=np.append(np.asarray(position,dtype=float),1.0)
        transform=self._camera_transform(frame,0)
        newpos=transform@position
        return newpos[:3]


    def bounding_box_to_world(self,bounding_box)->Position:
        with self._lock:
            if self._latest_frame is None or self._latest_depth is None:
                raise ValueError("No synchronized camera observation is available")

            box=np.asarray(bounding_box,dtype=float)
            if box.shape!=(4,) or not np.all(np.isfinite(box)):
                raise ValueError("bounding_box must contain four finite pixel coordinates: [x_min,y_min,x_max,y_max]")

            x1,y1,x2,y2=box
            if x2<=x1 or y2<=y1:
                raise ValueError("bounding_box maximum coordinates must be greater than its minimum coordinates")

            depthmap_data=self._bridge.imgmsg_to_cv2(self._latest_depth,desired_encoding="passthrough")
            if self._latest_depth.encoding=="16UC1":
                depthmap_data=depthmap_data.astype(np.float32)/1000.0
            elif self._latest_depth.encoding=="32FC1":
                depthmap_data=depthmap_data.astype(np.float32)
            else:
                raise ValueError(f"Unsupported depth image encoding: {self._latest_depth.encoding}")

            height,width=depthmap_data.shape[:2]
            if x2<=0 or y2<=0 or x1>=width or y1>=height:
                raise ValueError("bounding_box does not overlap the depth image")
            x1_i=max(0,min(width-1,int(math.floor(x1))))
            x2_i=max(0,min(width,int(math.ceil(x2))))
            y1_i=max(0,min(height-1,int(math.floor(y1))))
            y2_i=max(0,min(height,int(math.ceil(y2))))
            if x2_i<=x1_i or y2_i<=y1_i:
                raise ValueError("bounding_box does not overlap the depth image")

            box_width=x2_i-x1_i
            box_height=y2_i-y1_i
            sample_x1=x1_i+box_width//4
            sample_x2=x2_i-box_width//4
            sample_y1=y1_i+box_height//4
            sample_y2=y2_i-box_height//4
            cropped_depth=depthmap_data[sample_y1:sample_y2,sample_x1:sample_x2]
            valid_mask=np.isfinite(cropped_depth) & (cropped_depth>0.1) & (cropped_depth<10.0)
            if not valid_mask.any():
                raise ValueError("No valid depth was found inside the bounding box")

            valid_y,valid_x=np.nonzero(valid_mask)
            valid_depth=cropped_depth[valid_mask]
            median_depth=float(np.median(valid_depth))
            median_index=int(np.argmin(np.abs(valid_depth-median_depth)))
            u=float(sample_x1+valid_x[median_index])
            v=float(sample_y1+valid_y[median_index])

            caminfo=self._latest_frame.camera_info
            fx=caminfo.K[0]
            fy=caminfo.K[4]
            cx=caminfo.K[2]
            cy=caminfo.K[5]
            if fx<=0 or fy<=0:
                raise ValueError("Camera focal lengths must be positive")

            camera_position=[((u-cx)*median_depth)/fx,((v-cy)*median_depth)/fy,median_depth]
            world_position=self.c2w(self._latest_frame,camera_position)
            return Position(float(world_position[0]),float(world_position[1]),float(world_position[2]))


    def record(self)->None:
        """Used by tools to record world model updates."""
        with self._lock:
            self._startstate=deepcopy(self.observations)


    def stop_record(self)->dict:
        """Used by tools to stop recording world model updates."""
        with self._lock:
            if self._startstate is None:
                raise RuntimeError("record() must be called before stop_record()")
            startstate=self._startstate
            endstate=deepcopy(self.observations)
            self._startstate=None
        start_objects={object.id:object for object in startstate.objects}
        end_objects={object.id:object for object in endstate.objects}
        added_objects=[object for object in endstate.objects if object.id not in start_objects]
        removed_objects=[object for object in startstate.objects if object.id not in end_objects]
        changed_objects=[]
        for object in endstate.objects:
            if object.id in start_objects:
                old_object=start_objects[object.id]
                position_change=np.sqrt((object.position.x-old_object.position.x)**2+(object.position.y-old_object.position.y)**2+(object.position.z-old_object.position.z)**2)
                if abs(object.confidence_scores["existence_score"]-old_object.confidence_scores["existence_score"])>0.3 or abs(object.confidence_scores["positioning_score"]-old_object.confidence_scores["positioning_score"])>0.3 or position_change>0.3:
                    changed_objects.append({"New":object,"Old":old_object})

        return {"Added Objects":added_objects,"Removed Objects":removed_objects,"Changed Objects":changed_objects}


    def _get_latest(self,class_name:str, objects: list[Object] | None = None)->int:
        with self._lock:
            largest_number = 0
            source_objects = self.observations.objects if objects is None else objects

            for world_object in source_objects:
                if world_object.class_name == class_name:
                    number = int(world_object.id.rsplit("_", 1)[1])

                    if number > largest_number:
                        largest_number = number

            return largest_number


    def query_world(self,object_id: str | None = None,class_name: str | None = None,) -> list[Object]:
        """Return objects matching an ID, a class, or all objects."""
        with self._lock:
            if object_id is not None:
                result = [
                    world_object
                    for world_object in self.observations.objects
                    if world_object.id == object_id
                ]
            elif class_name is not None:
                result = [
                    world_object
                    for world_object in self.observations.objects
                    if world_object.class_name == class_name
                ]
            else:
                result = self.observations.objects

            return deepcopy(result)


    def get_detection_classes(self)->tuple[str,...]:
        return self.detection_classes


    def getallobjects(self)->str:
        listofobjects=[]
        for object in deepcopy(self.observations.objects):
            listofobjects.append({"ID":object.id, "Class Name":object.class_name})
        stringofobjects=json.dumps(listofobjects)
        return stringofobjects


    def _add_object(self, world_object: Object) -> None:
        with self._lock:
            if any(existing_object.id == world_object.id for existing_object in self.observations.objects):
                raise ValueError(f"An object with ID {world_object.id!r} already exists.")

            self.observations.objects.append(world_object)


    def _remove_object(self, object_id: str) -> Object:
        with self._lock:
            for index, world_object in enumerate(self.observations.objects):
                if world_object.id == object_id:
                    return self.observations.objects.pop(index)

            raise KeyError(f"No object with ID {object_id!r} exists.")


    def write(self,world_object: Object) -> None:
        candidate = deepcopy(world_object)
        for detection_class in self.detection_classes:
            if f"{detection_class}_" in candidate.class_name:
                raise ValueError(f"Class name {candidate.class_name} is invalid because it contains the reserved phrase '{detection_class}_'")
        candidate.attributes["Notes"]="Artificially written by VLM, not added by automatic updater"
        candidate.attributes["VLMFlag"]=1
        candidate.confidence_scores["existence_rawscore"]=0.0
        candidate.confidence_scores["existence_score"]=float(self.sigmoid(candidate.confidence_scores["existence_rawscore"]))

        with self._lock:
            self._add_object(candidate)
