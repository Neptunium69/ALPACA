from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING,Any,Callable,Protocol

import numpy as np
import rospy
from std_msgs.msg import Float64
from dataclasses import dataclass
from . import worldmodel as wm
from . import vehicleadapter as va
from .worldmodel import Voxel

class Command(Protocol):
    @property
    def id(self)->str: ...
    @property
    def command_type(self)->str: ...
    @property
    def arguments(self)->dict[str,Any]: ...

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry,Path
    from .vehicleadapter import State
    from .worldmodel import Orientation,Position

@dataclass
class ActionPrediction:
    cmd: Command
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
    def __init__(self,model:wm.ObjectWorldModel,adapter:va.VehicleAdapter)->None:
        self.model=model
        self.adapter=adapter
        self.occupied_confidence_threshold=0.1
        self.occupied_route_threshold=0.20
        self.information_route_threshold=0.8
        self.minimum_persistent_occupied_voxels=1
        self.maximum_vantage_candidates=32
        self.maximum_vantage_evaluation_seconds=30.0
    def checkcmd(self,command:Command,destination:Position | None=None,orientation:Orientation | None=None,frame_id:str | None=None)->dict[str,Any]:
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
            if frame_id is None:
                return {"Safe":False,"Reason":"Missing navigation frame."}
            hiddenpoint=self.adapter.hiddenpoint(command.arguments["target"],state=state,distance_behind=0.7)
            orient=state.orientation
            navresult=self.navigate(command=command, destination=hiddenpoint,orientation=orient,frame_id=frame_id)
            return navresult
        
        return {"Safe":True,"Reason":reason}

    def predict_navigation(self,command:Command,destination:Position | None,orientation:Orientation | None,frame_id:str | None,coverage_margin:float=0.10)->ActionPrediction:
        """Convert a move_base path into a conservative orientation-independent voxel envelope."""
        if destination is None or orientation is None or frame_id is None:
            raise ValueError("Navigation destination, orientation, and frame are required.")
        if not isinstance(coverage_margin,(int,float)) or isinstance(coverage_margin,bool) or not math.isfinite(coverage_margin) or coverage_margin<0:
            raise ValueError("Coverage margin must be a finite, non-negative number.")
        path=self.adapter.plan_path(destination,orientation,frame_id,tolerance=0.10)
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
        sample_spacing=voxel_size
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
    def _hardsafetyfilter(self,cmd:Command,state:State,destination:Position | None=None,orientation:Orientation | None=None,frame_id:str | None=None)->tuple[bool,str]:
        # Battery enforcement is temporarily disabled because this UGV's MAVROS
        # percentage estimate is intermittent and has rejected otherwise safe commands.
        if cmd.command_type in {"scan","navigate","inspect"}:
            if not state.armed:
                return False,"Vehicle is not armed."
            if state.mode!="OFFBOARD":
                return False,f"Vehicle is in {state.mode or 'an unknown mode'}, not OFFBOARD."
        if(cmd.command_type=="scan" or cmd.command_type=="image"):
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
                target_observations=self.model.get_voxel_states_and_scores(target_voxels)
            except (TypeError,ValueError):
                return False,"Vehicle collision map is not ready or contains invalid data."
            occupied_count=sum(observation is not None and observation[0]==1 and observation[1]>=self.occupied_confidence_threshold for observation in target_observations)
            if occupied_count>=self.minimum_persistent_occupied_voxels:
                return False,f"Target footprint intersects {occupied_count} persistent occupied voxel(s)."
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

    def navigate(self,command:Command,destination:Position,orientation:Orientation,frame_id:str,sensor_range:float=5.0,observation_probability:float=0.7,distance_cost_weight:float=0.2)->dict[str,Any]:
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
        if pathvoxel.occupied_score>self.occupied_route_threshold:
            return {"Safe":False,"Reason":"Path contains excessive obstacles."}

        if (len(occupied_voxels)+len(unknown_voxels))/pathvoxel.total_count>self.information_route_threshold and pathvoxel.occupied_score<self.occupied_route_threshold:
            vp=self.adapter.vantagepoint(pathvoxel)
            center_position=self.model.voxel_to_position(self.center(pathvoxel))
            total_unknown_value=float(len(unknown_voxels))
            reference_distance=max(prediction.distance,1.0)
            options=[]
            candidate_counts={
                "total":len(vp),
                "destination_not_free":0,
                "known_free":0,
                "not_evaluated_due_to_limit":0,
                "not_evaluated_due_to_time":0,
                "evaluated":0,
                "target_footprint_check_failed":0,
                "target_footprint_unsafe":0,
                "planning_failed":0,
                "empty_path":0,
                "path_over_20_percent_occupied":0,
                "accepted":0,
                "time_budget_exhausted":False,
                "evaluation_seconds":0.0,
            }

            robot_position=self.adapter.get_state().position
            known_free_candidates=[]
            for vantage_voxel in vp:
                vantage_state=self.model.get_voxel_states([vantage_voxel])[0]
                if vantage_state!=0:
                    candidate_counts["destination_not_free"]+=1
                    continue
                vantage_position=self.model.voxel_to_position(vantage_voxel)
                distance_squared=(vantage_position.x-robot_position.x)**2+(vantage_position.y-robot_position.y)**2
                known_free_candidates.append((distance_squared,vantage_voxel,vantage_position))
            known_free_candidates.sort(key=lambda candidate:(candidate[0],candidate[1]))
            candidate_counts["known_free"]=len(known_free_candidates)
            candidate_counts["not_evaluated_due_to_limit"]=max(0,len(known_free_candidates)-self.maximum_vantage_candidates)
            candidates_to_evaluate=known_free_candidates[:self.maximum_vantage_candidates]
            vehicle_pose,vehicle_min,vehicle_max=self.adapter.get_collision_geometry()
            if vehicle_pose is None:
                return {"Safe":False,"Reason":"Vehicle localization is not ready.","Needs Information":True,"Inform User":True,"Candidate Counts":candidate_counts}
            evaluation_started=time.monotonic()

            for candidate_index,(_,vantage_voxel,vantage_position) in enumerate(candidates_to_evaluate):
                elapsed=time.monotonic()-evaluation_started
                if elapsed>=self.maximum_vantage_evaluation_seconds:
                    candidate_counts["not_evaluated_due_to_time"]=len(candidates_to_evaluate)-candidate_index
                    candidate_counts["time_budget_exhausted"]=True
                    rospy.logwarn("Vantage evaluation reached its %.1f-second time budget after %d candidate(s); selecting the best valid option found so far.",self.maximum_vantage_evaluation_seconds,candidate_counts["evaluated"])
                    break
                candidate_counts["evaluated"]+=1
                if candidate_counts["evaluated"]==1 or candidate_counts["evaluated"]%10==0 or candidate_counts["evaluated"]==len(candidates_to_evaluate):
                    rospy.loginfo("Vantage evaluation progress: evaluating candidate %d/%d; %d accepted so far.",candidate_counts["evaluated"],len(candidates_to_evaluate),candidate_counts["accepted"])
                vantage_yaw=math.atan2(center_position.y-vantage_position.y,center_position.x-vantage_position.x)
                vantage_orientation=wm.Orientation(pitch=0.0,yaw=vantage_yaw,roll=0.0)
                try:
                    target_voxels=self.model.get_vehicle_voxels_at_pose(vantage_position,vantage_orientation,vehicle_min,vehicle_max)
                    target_observations=self.model.get_voxel_states_and_scores(target_voxels)
                except (TypeError,ValueError):
                    candidate_counts["target_footprint_check_failed"]+=1
                    continue
                persistent_occupied_count=sum(observation is not None and observation[0]==1 and observation[1]>=self.occupied_confidence_threshold for observation in target_observations)
                if persistent_occupied_count>=self.minimum_persistent_occupied_voxels:
                    candidate_counts["target_footprint_unsafe"]+=1
                    continue
                try:
                    vantage_prediction=self.predict_navigation(command,vantage_position,vantage_orientation,frame_id)
                except (RuntimeError,TypeError,ValueError):
                    candidate_counts["planning_failed"]+=1
                    continue
                vantage_path_states=self.model.get_voxel_states(list(vantage_prediction.potential_voxels))
                if not vantage_path_states:
                    candidate_counts["empty_path"]+=1
                    continue
                vantage_occupied_score=sum(state==1 for state in vantage_path_states)/len(vantage_path_states)
                if vantage_occupied_score>self.occupied_route_threshold:
                    candidate_counts["path_over_20_percent_occupied"]+=1
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
                candidate_counts["accepted"]+=1

            candidate_counts["evaluation_seconds"]=round(time.monotonic()-evaluation_started,3)
            rospy.loginfo("Vantage candidates: total=%d, destination_not_free=%d, known_free=%d, evaluated=%d, limited_out=%d, timed_out=%d, target_check_failed=%d, target_unsafe=%d, planning_failed=%d, empty_path=%d, path_over_20_percent_occupied=%d, accepted=%d, time_budget_exhausted=%s, evaluation_seconds=%.3f.",candidate_counts["total"],candidate_counts["destination_not_free"],candidate_counts["known_free"],candidate_counts["evaluated"],candidate_counts["not_evaluated_due_to_limit"],candidate_counts["not_evaluated_due_to_time"],candidate_counts["target_footprint_check_failed"],candidate_counts["target_footprint_unsafe"],candidate_counts["planning_failed"],candidate_counts["empty_path"],candidate_counts["path_over_20_percent_occupied"],candidate_counts["accepted"],candidate_counts["time_budget_exhausted"],candidate_counts["evaluation_seconds"])
            if not options:
                return {"Safe":False,"Reason":"No vantage points available.","Needs Information":True,"Inform User":True,"Candidate Counts":candidate_counts}
            best_option=max(options,key=lambda option:option["Profit"])
            if best_option["Profit"]<=0:
                return {"Safe":False,"Reason":"No vantage points have an expected value greater than cost.","Needs Information":True,"Inform User":True,"Candidate Counts":candidate_counts,"Vantage Options":options}
            return {"Safe":False,"Reason":"Route requires more information.","Needs Information":True,"Inform User":False,"Candidate Counts":candidate_counts,"Best Vantage Point":best_option,"Vantage Options":options,"Prediction":prediction,"Assessment":pathvoxel}

        return {"Safe":True,"Reason":"Route accepted.","Prediction":prediction,"Assessment":pathvoxel}
