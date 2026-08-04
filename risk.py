from __future__ import annotations

import math
from typing import TYPE_CHECKING,Any,Protocol

import numpy as np
import rospy
from std_msgs.msg import Float64
from dataclasses import dataclass
from . import worldmodel as wm
from .worldmodel import Voxel
from . import tools

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry,Path
    from .vehicleadapter import State
    from .worldmodel import Orientation,Position

class CommandLike(Protocol):
    @property
    def command_type(self)->str: ...

@dataclass
class ActionPrediction:
    cmd: CommandLike
    potential_voxels: set[tuple[int,int,int]]
    distance: float
    terminal_position: Position
    terminal_orientation: Orientation
    frame_id: str

class RiskAdapter(Protocol):
    @property
    def navigation_frame(self)->str: ...
    def get_state(self)->State: ...
    def get_collision_geometry(self)->tuple[Odometry | None,Any,Any]: ...
    def plan_path(self,destination:Position,orientation:Orientation,frame_id:str | None=None,tolerance:float=0.0)->Path: ...

class RiskWorldModel(Protocol):
    def get_voxel_map_frame(self)->str | None: ...
    def get_voxel_size(self)->float: ...
    def get_vehicle_voxels_at_pose(self,position:Position,orientation:Orientation,vehicle_min:np.ndarray,vehicle_max:np.ndarray)->list[tuple[int,int,int]]: ...
    def get_voxel_states(self,indices:list[tuple[int,int,int]])->list[int | None]: ...

class RiskCal():
    def __init__(self,model:RiskWorldModel,adapter:RiskAdapter)->None:
        self.model=model
        self.adapter=adapter
    def initrcm(self,model:RiskWorldModel,adapter:RiskAdapter)->None:
        if not rospy.core.is_initialized():
            rospy.init_node("RiskCal",anonymous=True)
    def checkcmd(self,command:CommandLike,destination:Position | None=None,orientation:Orientation | None=None,frame_id:str | None=None)->dict[str,Any]:
        state=self.adapter.get_state()
        safe,reason=self._hardsafetyfilter(command,state,destination,orientation,frame_id)
        if not safe:
            return {"Safe":False,"Reason":reason}
        if command.command_type=="navigate":
            try:
                prediction=self.predict_navigation(command,destination,orientation,frame_id)
            except (RuntimeError,TypeError,ValueError) as error:
                return {"Safe":False,"Reason":f"Navigation prediction failed: {error}"}
            return {"Safe":True,"Reason":reason,"Prediction":prediction}
        return {"Safe":True,"Reason":reason}

    def predict_navigation(self,command:CommandLike,destination:Position | None,orientation:Orientation | None,frame_id:str | None,coverage_margin:float=0.10)->ActionPrediction:
        """Convert a move_base path into a conservative orientation-independent voxel envelope."""
        if destination is None or orientation is None or frame_id is None:
            raise ValueError("Navigation destination, orientation, and frame are required.")
        if not isinstance(coverage_margin,(int,float)) or isinstance(coverage_margin,bool) or not math.isfinite(coverage_margin) or coverage_margin<0:
            raise ValueError("Coverage margin must be a finite, non-negative number.")
        path=self.adapter.plan_path(destination,orientation,frame_id)
        positions=self._path_positions(path)
        vehicle_pose,vehicle_min,vehicle_max=self.adapter.get_collision_geometry()
        if vehicle_pose is None:
            raise RuntimeError("Vehicle localization is not ready.")
        vehicle_min=np.asarray(vehicle_min,dtype=np.float64)
        vehicle_max=np.asarray(vehicle_max,dtype=np.float64)
        if vehicle_min.shape!=(3,) or vehicle_max.shape!=(3,) or not np.all(np.isfinite(vehicle_min)) or not np.all(np.isfinite(vehicle_max)) or np.any(vehicle_min>=vehicle_max):
            raise ValueError("Vehicle collision geometry is invalid.")

        maximum_x=max(abs(vehicle_min[0]),abs(vehicle_max[0]))
        maximum_y=max(abs(vehicle_min[1]),abs(vehicle_max[1]))
        cover_radius=math.hypot(maximum_x,maximum_y)+float(coverage_margin)
        conservative_min=np.array([-cover_radius,-cover_radius,vehicle_min[2]],dtype=np.float64)
        conservative_max=np.array([cover_radius,cover_radius,vehicle_max[2]],dtype=np.float64)
        voxel_size=self.model.get_voxel_size()
        sample_spacing=voxel_size/2.0
        potential_voxels:set[tuple[int,int,int]]=set()
        zero_orientation=wm.Orientation(pitch=0.0,yaw=0.0,roll=0.0)
        distance=0.0

        if len(positions)==1:
            potential_voxels.update(self.model.get_vehicle_voxels_at_pose(positions[0],zero_orientation,conservative_min,conservative_max))
        for start,end in zip(positions,positions[1:]):
            start_array=np.array([start.x,start.y,start.z],dtype=np.float64)
            end_array=np.array([end.x,end.y,end.z],dtype=np.float64)
            delta=end_array-start_array
            segment_distance=float(np.linalg.norm(delta))
            distance+=segment_distance
            sample_count=max(1,int(math.ceil(segment_distance/sample_spacing)))
            for sample_index in range(sample_count+1):
                fraction=sample_index/sample_count
                sample=start_array+fraction*delta
                sample_position=wm.Position(x=float(sample[0]),y=float(sample[1]),z=float(sample[2]))
                potential_voxels.update(self.model.get_vehicle_voxels_at_pose(sample_position,zero_orientation,conservative_min,conservative_max))

        return ActionPrediction(cmd=command,potential_voxels=potential_voxels,distance=distance,terminal_position=positions[-1],terminal_orientation=orientation,frame_id=frame_id)

    @staticmethod
    def _path_positions(path:Path)->list[Position]:
        if not path.poses:
            raise ValueError("Planned path is empty.")
        positions=[]
        for stamped_pose in path.poses:
            values=np.array([stamped_pose.pose.position.x,stamped_pose.pose.position.y,stamped_pose.pose.position.z],dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("Planned path contains a non-finite position.")
            positions.append(wm.Position(x=float(values[0]),y=float(values[1]),z=float(values[2])))
        return positions
    def _hardsafetyfilter(self,cmd:CommandLike,state:State,destination:Position | None=None,orientation:Orientation | None=None,frame_id:str | None=None)->tuple[bool,str]:
        if not isinstance(state.battery_level,(int,float)) or isinstance(state.battery_level,bool) or not math.isfinite(state.battery_level) or not 0.0<=state.battery_level<=1.0:
            return False,"Battery level is unavailable or invalid."
        if state.battery_level<0.15:
            return False,"Battery level too low, please contact user to recharge or replace battery."
        if(cmd.command_type=="scan"):
            try:
                clutter=rospy.wait_for_message("clutter_level",Float64,timeout=5)
                if clutter is None:
                    return False,"Sensor unresponsive."
                if not math.isfinite(clutter.data):
                    return False,"Clutter sensor returned an invalid value."
                if(clutter.data>=0.7):
                    return False,"Vehicle was unable to scan because the clutter level is too high."
            except rospy.ROSException:
                return False,"Sensor unresponsive."

        if(cmd.command_type=="navigate"):
            if destination is None or orientation is None or frame_id is None:
                return False,"Navigation goal has not been validated."
            try:
                vehicle_pose,vehicle_min,vehicle_max=self.adapter.get_collision_geometry()
            except (AttributeError,NotImplementedError):
                return False,"Vehicle adapter does not provide collision geometry."
            if vehicle_pose is None:
                return False,"Vehicle localization is not ready."
            pose_frame=vehicle_pose.header.frame_id
            voxel_frame=self.model.get_voxel_map_frame()
            if not pose_frame or voxel_frame is None:
                return False,"Vehicle localization or voxel map is not ready."
            if frame_id!=self.adapter.navigation_frame or frame_id!=pose_frame or frame_id!=voxel_frame:
                return False,f"Navigation frame mismatch: goal={frame_id!r}, adapter={self.adapter.navigation_frame!r}, pose={pose_frame!r}, voxel_map={voxel_frame!r}."
            try:
                target_voxels=self.model.get_vehicle_voxels_at_pose(destination,orientation,vehicle_min,vehicle_max)
                target_states=self.model.get_voxel_states(target_voxels)
            except (TypeError,ValueError):
                return False,"Vehicle collision map is not ready or contains invalid data."
            occupied_count=sum(state==1 for state in target_states)
            unknown_count=sum(state is None for state in target_states)
            if occupied_count:
                return False,f"Target footprint intersects {occupied_count} occupied voxel(s)."
            if unknown_count:
                return False,f"Target footprint contains {unknown_count} unknown voxel(s)."
        return True,"Command is safe to execute."

    def _cost_reward(self,cmd: tools.Command):
        reward,cost=0,0
        if cmd.command_type is "finish":
            pass
            # insert something to confirm with the Vision Language Model
        if cmd.command_type is "ask_user":
            pass
            # insert something to determine if ask user is worth it.
        if cmd.command_type is "inspect":
            pass
            # insert something to determine the unknown regions of the tagret.
