from __future__ import annotations

import math
from typing import TYPE_CHECKING,Any,Callable,Protocol

import numpy as np
import rospy
from std_msgs.msg import Float64
from dataclasses import dataclass
from . import worldmodel as wm
from . import vehicleadapter as va
from .worldmodel import Voxel
from . import tools

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry,Path
    from .vehicleadapter import State
    from .worldmodel import Orientation,Position

@dataclass
class ActionPrediction:
    cmd: tools.Command
    potential_voxels: set[tuple[int,int,int]]
    distance: float
    terminal_position: Position
    terminal_orientation: Orientation
    frame_id: str

@dataclass
class PathVoxelAssessment:
    occupied_voxels: set[tuple[int,int,int]]
    free_voxels: set[tuple[int,int,int]]
    unknown_voxels: set[tuple[int,int,int]]

    @property
    def total_count(self)->int:
        return len(self.occupied_voxels)+len(self.free_voxels)+len(self.unknown_voxels)

    @property
    def unknown_score(self)->float:
        return len(self.unknown_voxels)/self.total_count if self.total_count else 1.0

    @property
    def occupied_score(self)->float:
        return len(self.occupied_voxels)/self.total_count if self.total_count else 1.0

    @property
    def free_score(self)->float:
        return len(self.free_voxels)/self.total_count if self.total_count else 1.0

@dataclass(frozen=True)
class VantagePoint:
    position: wm.Position
    orientation: wm.Orientation
    target_position: wm.Position
    observable_voxels: frozenset[tuple[int,int,int]]
    standoff_distance: float
    straight_line_distance: float

class RiskCal():
    def __init__(self,model:wm.WorldModel,adapter:va.VehicleAdapter)->None:
        self.model=model
        self.adapter=adapter
    def checkcmd(self,command:tools.Command,destination:Position | None=None,orientation:Orientation | None=None,frame_id:str | None=None)->dict[str,Any]:
        state=self.adapter.get_state()
        safe,reason=self._hardsafetyfilter(command,state,destination,orientation,frame_id)
        if not safe:
            return {"Safe":False,"Reason":reason}

        if command.command_type=="navigate":
            if destination is None or orientation is None or frame_id is None:
                return {"Safe":False,"Reason":"Missing Arguments"}
            navresult=self.navigate(command=command,destination=destination,orientation=orientation,frame_id=frame_id)
            return navresult

        if command.command_type=="inspect":
            hiddenpoint=self.model.hiddenpoint(command.arguments["target"],self.adapter.state.position,state=self.adapter.get_state(), distance_behind=0.7)
            orient=self.adapter.state.orientation
            if destination is None or orientation is None or frame_id is None:
                return {"Safe":False,"Reason":"Missing Arguments"}
            navresult=self.navigate(command=command, destination=hiddenpoint,orientation=orient,frame_id=frame_id)
            return navresult
        
        return {"Safe":True,"Reason":reason}

    def predict_navigation(self,command:tools.Command,destination:Position | None,orientation:Orientation | None,frame_id:str | None,coverage_margin:float=0.10)->ActionPrediction:
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
    def _hardsafetyfilter(self,cmd:tools.Command,state:State,destination:Position | None=None,orientation:Orientation | None=None,frame_id:str | None=None)->tuple[bool,str]:
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

    def center(self,assessment:PathVoxelAssessment)->tuple[int,int,int]:
        """Find a voxel approximately at the center of all voxels."""
        if not assessment.unknown_voxels:
            raise ValueError("No unknown voxels present.")
        unknown_voxels=sorted(assessment.unknown_voxels)
        coordinates=np.asarray(unknown_voxels,dtype=np.float64)
        centroid=np.median(coordinates,axis=0)
        nearest_voxel=min(unknown_voxels,key=lambda voxel:(float(np.sum((np.asarray(voxel,dtype=np.float64)-centroid)**2)),voxel))
        return int(nearest_voxel[0]),int(nearest_voxel[1]),int(nearest_voxel[2])

    def navigate(self,command:tools.Command,destination:Position,orientation:Orientation,frame_id:str,sensor_range:float=5.0,observation_probability:float=0.7,distance_cost_weight:float=0.2)->dict[str,Any]:
        if not isinstance(sensor_range,(int,float)) or isinstance(sensor_range,bool) or not math.isfinite(sensor_range) or sensor_range<=0:
            raise ValueError("sensor_range must be a finite, positive number.")
        if not isinstance(observation_probability,(int,float)) or isinstance(observation_probability,bool) or not math.isfinite(observation_probability) or not 0.0<=observation_probability<=1.0:
            raise ValueError("observation_probability must be between zero and one.")
        if not isinstance(distance_cost_weight,(int,float)) or isinstance(distance_cost_weight,bool) or not math.isfinite(distance_cost_weight) or distance_cost_weight<0:
            raise ValueError("distance_cost_weight must be a finite, non-negative number.")
        try:
            prediction=self.predict_navigation(command,destination,orientation,frame_id)
        except (RuntimeError,TypeError,ValueError) as error:
            return {"Safe":False,"Reason":f"Navigation prediction failed: {error}"}

        free_voxels:set[tuple[int,int,int]]=set()
        occupied_voxels:set[tuple[int,int,int]]=set()
        unknown_voxels:set[tuple[int,int,int]]=set()
        potential_voxels=list(prediction.potential_voxels)
        voxel_states=self.model.get_voxel_states(potential_voxels)
        for voxel,state in zip(potential_voxels,voxel_states):
            if state==0:
                free_voxels.add(voxel)
            elif state==1:
                occupied_voxels.add(voxel)
            elif state is None:
                unknown_voxels.add(voxel)
            else:
                return {"Safe":False,"Reason":f"Voxel {voxel!r} has invalid state {state!r}."}
        pathvoxel=PathVoxelAssessment(occupied_voxels=occupied_voxels,unknown_voxels=unknown_voxels,free_voxels=free_voxels)
        if pathvoxel.total_count==0:
            return {"Safe":False,"Reason":"Navigation prediction contains no voxels."}
        if pathvoxel.occupied_score>0.2:
            return {"Safe":False,"Reason":"Path contains excessive obstacles."}

        if (len(occupied_voxels)+len(unknown_voxels))/pathvoxel.total_count>0.2 and pathvoxel.occupied_score<0.2:
            vp=self.adapter.vantagepoint(pathvoxel)
            center_position=self.model.voxel_to_position(self.center(pathvoxel))
            total_unknown_value=float(len(unknown_voxels))
            reference_distance=max(prediction.distance,1.0)
            options=[]

            for vantage_voxel in sorted(vp):
                vantage_state=self.model.get_voxel_states([vantage_voxel])[0]
                if vantage_state!=0:
                    continue
                vantage_position=self.model.voxel_to_position(vantage_voxel)
                vantage_yaw=math.atan2(center_position.y-vantage_position.y,center_position.x-vantage_position.x)
                vantage_orientation=wm.Orientation(pitch=0.0,yaw=vantage_yaw,roll=0.0)
                try:
                    vantage_prediction=self.predict_navigation(command,vantage_position,vantage_orientation,frame_id)
                except (RuntimeError,TypeError,ValueError):
                    continue
                vantage_path_states=self.model.get_voxel_states(list(vantage_prediction.potential_voxels))
                if not vantage_path_states or any(state!=0 for state in vantage_path_states):
                    continue

                observable_voxels=set()
                for unknown_voxel in unknown_voxels:
                    unknown_position=self.model.voxel_to_position(unknown_voxel)
                    distance=math.sqrt((unknown_position.x-vantage_position.x)**2+(unknown_position.y-vantage_position.y)**2+(unknown_position.z-vantage_position.z)**2)
                    if distance<=sensor_range:
                        observable_voxels.add(unknown_voxel)

                information_reward=observation_probability*(len(observable_voxels)/total_unknown_value)
                path_cost=distance_cost_weight*(vantage_prediction.distance/reference_distance)
                profit=information_reward-path_cost
                options.append({
                    "Vantage Voxel":vantage_voxel,
                    "Position":vantage_position,
                    "Orientation":vantage_orientation,
                    "Path Distance":vantage_prediction.distance,
                    "Observable Voxels":observable_voxels,
                    "Information Reward":information_reward,
                    "Path Cost":path_cost,
                    "Profit":profit,
                })

            if not options:
                return {"Safe":False,"Reason":"No vantage points availible.","Needs Information":True,"Inform User":True}
            best_option=max(options,key=lambda option:option["Profit"])
            if best_option["Profit"]<=0:
                return {"Safe":False,"Reason":"No vantage points have an expected value greater than cost.","Needs Information":True,"Inform User":True,"Vantage Options":options}
            return {"Safe":False,"Reason":"Route requires more information.","Needs Information":True, "Inform User":False,"Best Vantage Point":best_option,"Vantage Options":options,"Prediction":prediction,"Assessment":pathvoxel}

        return {"Safe":True,"Reason":"Route accepted.","Prediction":prediction,"Assessment":pathvoxel}
