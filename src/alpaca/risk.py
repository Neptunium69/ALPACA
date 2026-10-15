from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING,Any,Protocol
from sensor_msgs.msg import CameraInfo
import numpy as np
import rospy
from std_msgs.msg import Float64
from dataclasses import dataclass
from . import worldmodel as wm
from .dataclasses import FCExecute
from .vatools import VAInterface
import torch
from .wmtools import WorldModelInterface

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
    path: Path

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
    def __init__(self,wminterface:WorldModelInterface,vainterface:VAInterface)->None:
        self.wminterface=wminterface
        self.vainterface=vainterface
        self.occupied_confidence_threshold=0.1
        self.occupied_route_threshold=0.20
        self.information_route_threshold=0.6
        self.minimum_persistent_occupied_voxels=10
        self.vantage_prediction_speed=0.3
        self.k=50
        self.turn_cost_wgt=0.2
        self.move_cost_wgt=0.3
        self.frustration=0
        self.frustration_max=6
        self.cam_wgt=2
        self.lid_wgt=2
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim_iterations=81
        self.cs_occ_ratio=0.5 # Fraction of a route cross-section (50%).

    
    def _actually_smart_vantage_reward(self,observable_voxels)->int:
        observations=self.wminterface.execute(FCExecute(function="states_scores",arguments={"indices":list(observable_voxels)}),submodel="voxels_lidar").data
        sum=0
        for observation in observations:
            if observation is not None:
                sum+=max(0,1-2*abs(observation[1]-0.5))
            else:
                sum+=1
        return self.lid_wgt*sum
    
    def _c(self,observable_voxels, position:Position, orientation:Orientation)->int:
        sum=0
        for voxel in observable_voxels:
            voxel_position=self.wminterface.execute(FCExecute(function="voxel_to_position",arguments={"index":voxel}),submodel="voxels_master").data
            observation=self.wminterface.execute(FCExecute(function="states_scores",arguments={"indices":[voxel]}),submodel="voxels_camera").data[0]
            dx,dy=voxel_position.x-position.x,voxel_position.y-position.y
            distance=math.hypot(dx,dy)
            if distance<=1e-9:
                continue
            cos_theta=(dx*math.cos(orientation.yaw)+dy*math.sin(orientation.yaw))/distance
            cos_theta=max(-1.0,min(1.0,cos_theta))
            camera_info: CameraInfo=self.camea_info
            fx=camera_info.K[0]
            fov=2.0*math.atan(camera_info.width/(2.0*fx))
            if observation is not None:
                sum+=max(0,1-2*abs(observation[1]-0.5))/(1+math.e**(-self.k*(cos_theta-math.cos(fov/2.0))))
            else:
                sum+=1
        return self.cam_wgt*sum

    def checkcmd(self,command:Command,destination:Position | None=None,orientation:Orientation | None=None,frame_id:str | None=None,allow_scan:bool=False,completed_views:list[tuple[Position,float]] | None=None)->dict[str,Any]:
        state=self.vainterface.execute(FCExecute(function="get_state",arguments={})).data
        safe,reason=self._hardsafetyfilter(command,state,destination,orientation,frame_id)
        if not safe:
            return {"Safe":False,"Reason":reason}

        if command.command_type=="navigate":
            if destination is None or orientation is None or frame_id is None:
                return {"Safe":False,"Reason":"Missing Arguments"}
            navresult=self.navigate(command=command,destination=destination,orientation=orientation,frame_id=frame_id,allow_scan=allow_scan,completed_views=completed_views)
            return navresult

        if command.command_type=="inspect":
            if frame_id is None:
                return {"Safe":False,"Reason":"Missing navigation frame."}
            hiddenpoint=self.vainterface.execute(FCExecute(function="hiddenpoint",arguments={"object_id":command.arguments["target"],"state":state,"distance_behind":0.7})).data
            orient=state.data["orientation"]
            navresult=self.navigate(command=command, destination=hiddenpoint,orientation=orient,frame_id=frame_id)
            return navresult
        
        return {"Safe":True,"Reason":reason}

    def predict_navigation(self,command:Command,destination:Position | None,orientation:Orientation | None,frame_id:str | None,coverage_margin:float=0.05)->ActionPrediction:
        """Convert a move_base path into a conservative orientation-independent voxel envelope."""
        if destination is None or orientation is None or frame_id is None:
            raise ValueError("Navigation destination, orientation, and frame are required.")
        if not isinstance(coverage_margin,(int,float)) or isinstance(coverage_margin,bool) or not math.isfinite(coverage_margin) or coverage_margin<0:
            raise ValueError("Coverage margin must be a finite, non-negative number.")
        path=self.vainterface.execute(FCExecute(function="plan_path",arguments={"destination":destination,"orientation":orientation,"frame_id":frame_id,"tolerance":0.10})).data
        positions=self._path_positions(path)
        vehicle_pose,vehicle_min,vehicle_max=self.vainterface.execute(FCExecute(function="get_collision_geometry",arguments={})).data
        if vehicle_pose is None:
            raise RuntimeError("Vehicle localization is not ready.")
        vehicle_min=np.asarray(vehicle_min,dtype=np.float64)
        vehicle_max=np.asarray(vehicle_max,dtype=np.float64)
        if vehicle_min.shape!=(3,) or vehicle_max.shape!=(3,) or not np.all(np.isfinite(vehicle_min)) or not np.all(np.isfinite(vehicle_max)) or np.any(vehicle_min>=vehicle_max):
            raise ValueError("Vehicle collision geometry is invalid.")

        maximum_x=max(abs(vehicle_min[0]),abs(vehicle_max[0]))
        maximum_y=max(abs(vehicle_min[1]),abs(vehicle_max[1]))
        cover_radius=0.225
        conservative_min=np.array([-cover_radius,-cover_radius,vehicle_min[2]+0.03],dtype=np.float64)
        conservative_max=np.array([cover_radius,cover_radius,vehicle_max[2]],dtype=np.float64)
        voxel_size=self.wminterface.execute(FCExecute(function="get_voxel_size",arguments={}),submodel="voxels_master").data
        sample_spacing=voxel_size
        path_samples:list[np.ndarray]=[]
        distance=0.0

        if len(positions)==1:
            path_samples.append(np.array([positions[0].x,positions[0].y,positions[0].z],dtype=np.float64))
        for segment_index,(start,end) in enumerate(zip(positions,positions[1:])):
            start_array=np.array([start.x,start.y,start.z],dtype=np.float64)
            end_array=np.array([end.x,end.y,end.z],dtype=np.float64)
            delta=end_array-start_array
            segment_distance=float(np.linalg.norm(delta))
            distance+=segment_distance
            sample_count=max(1,int(math.ceil(segment_distance/sample_spacing)))
            for sample_index in range(0 if segment_index==0 else 1,sample_count+1):
                fraction=sample_index/sample_count
                sample=start_array+fraction*delta
                path_samples.append(sample)

        potential_voxels=set(self.wminterface.execute(FCExecute(function="footprint_at",arguments={"position":positions[0],"orientation":wm.Orientation(pitch=0.0,yaw=0.0,roll=0.0),"vehicle_min":conservative_min,"vehicle_max":conservative_max,"samples":np.asarray(path_samples,dtype=np.float64).reshape(-1,3)}),submodel="voxels_master").data)

        return ActionPrediction(cmd=command,potential_voxels=potential_voxels,distance=distance,terminal_position=positions[-1],terminal_orientation=orientation,frame_id=frame_id,path=path)

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
        # if cmd.command_type in {"scan","navigate","inspect"}:
        #     if not state.data["armed"]:
        #         return False,"Vehicle is not armed."
        #     if state.data["mode"]!="OFFBOARD":
        #         return False,f"Vehicle is in {state.data['mode'] or 'an unknown mode'}, not OFFBOARD."
        # if(cmd.command_type=="scan" or cmd.command_type=="image"):
        #     try:
        #         clutter=rospy.wait_for_message("clutter_level",Float64,timeout=5)
        #         if clutter is None:
        #             return False,"Sensor unresponsive."
        #         if not math.isfinite(clutter.data):
        #             return False,"Clutter sensor returned an invalid value."
        #         if(clutter.data>=0.7):
        #             return False,"Vehicle was unable to scan because the clutter level is too high."
        #     except rospy.ROSException:
        #         return False,"Sensor unresponsive."

        # if(cmd.command_type=="navigate"):
        #     if destination is None or orientation is None or frame_id is None:
        #         return False,"Navigation goal has not been validated."
        #     try:
        #         vehicle_pose,vehicle_min,vehicle_max=self.vainterface.execute(FCExecute(function="get_collision_geometry",arguments={})).data
        #     except (AttributeError,NotImplementedError,ValueError):
        #         return False,"Vehicle adapter does not provide collision geometry."
        #     if vehicle_pose is None:
        #         return False,"Vehicle localization is not ready."
        #     pose_frame=vehicle_pose.header.frame_id
        #     voxel_frame=self.wminterface.execute(FCExecute(function="get_voxel_map_frame",arguments={}),submodel="voxels_master").data
        #     if not pose_frame or voxel_frame is None:
        #         return False,"Vehicle localization or voxel map is not ready."
        #     navigation_frame=self.vainterface.execute(FCExecute(function="get_navigation_frame",arguments={})).data
        #     if frame_id!=navigation_frame or frame_id!=pose_frame or frame_id!=voxel_frame:
        #         return False,f"Navigation frame mismatch: goal={frame_id!r}, adapter={navigation_frame!r}, pose={pose_frame!r}, voxel_map={voxel_frame!r}."
        #     try:
        #         occupancy_min=vehicle_min.copy()
        #         occupancy_min[2]+=0.03
        #         target_voxels=self.wminterface.execute(FCExecute(function="footprint_at",arguments={"position":destination,"orientation":orientation,"vehicle_min":occupancy_min,"vehicle_max":vehicle_max}),submodel="voxels_master").data
        #         target_observations=self.wminterface.execute(FCExecute(function="states_scores",arguments={"indices":target_voxels}),submodel="voxels_master").data
        #     except (TypeError,ValueError):
        #         return False,"Vehicle collision map is not ready or contains invalid data."
        #     occupied_count=sum(observation is not None and observation[0]==1 and observation[1]>=self.occupied_confidence_threshold for observation in target_observations)
        #     if occupied_count>=self.minimum_persistent_occupied_voxels:
        #         return False,f"Target footprint intersects {occupied_count} persistent occupied voxel(s)."
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

    def _route_check(self,potential_voxels:set[tuple[int,int,int]])->tuple[PathVoxelAssessment,list,list,bool]:
        free_voxels:set[tuple[int,int,int]]=set()
        occupied_voxels:set[tuple[int,int,int]]=set()
        unknown_voxels:set[tuple[int,int,int]]=set()
        route_heights={index[2] for index in potential_voxels}
        map_states,voxel_size=self.wminterface.execute(FCExecute(function="get_voxel_snapshot",arguments={}),submodel="voxels_master").data
        vehicle_pose,vehicle_min,vehicle_max=self.vainterface.execute(FCExecute(function="get_collision_geometry",arguments={})).data
        self_voxels=self.wminterface.execute(FCExecute(function="get_vehicle_voxels",arguments={"vehicle_pose":vehicle_pose,"vehicle_min":np.asarray(vehicle_min)-voxel_size,"vehicle_max":np.asarray(vehicle_max)+voxel_size}),submodel="voxels_master").data
        ignored_unknown={index for index in self_voxels if map_states.get(index) is None}
        unknown_check_voxels=potential_voxels-ignored_unknown
        voxel_states=[map_states.get(index) for index in potential_voxels]
        clusters,unclustered=self.wminterface.execute(FCExecute(function="cluster_voxels",arguments={"indices":{index for index,state in map_states.items() if state==1 and index[2] in route_heights},"voxel_size":voxel_size,"gap":0.1}),submodel="voxels_master").data
        for voxel,state in zip(potential_voxels,voxel_states):
            if state==0:
                free_voxels.add(voxel)
            elif state==1:
                occupied_voxels.add(voxel)
            elif state is None:
                if voxel not in ignored_unknown:
                    unknown_voxels.add(voxel)
            else:
                raise ValueError(f"Voxel {voxel!r} has invalid state {state!r}.")
        pathvoxel=PathVoxelAssessment(occupied_voxels=occupied_voxels,unknown_voxels=unknown_voxels,free_voxels=free_voxels)
        if not potential_voxels:
            raise ValueError("Navigation prediction contains no voxels.")
        blocking_clusters=[cluster for cluster in clusters if not cluster.isdisjoint(potential_voxels)]
        if blocking_clusters:
            return pathvoxel,blocking_clusters,[],False

        unknown_clusters,_=self.wminterface.execute(FCExecute(function="cluster_voxels",arguments={"indices":({index for index,state in map_states.items() if state is None and index[2] in route_heights} | unknown_voxels)-ignored_unknown,"voxel_size":voxel_size,"gap":0.1,"min_samples":1}),submodel="voxels_master").data
        unknown_clusters=[cluster for cluster in unknown_clusters if not cluster.isdisjoint(unknown_check_voxels)]
        path_sections={}
        for x,y,z in unknown_check_voxels:
            path_sections.setdefault(x,set()).add((y,z))

        needs_information=False
        for unknown_cluster in unknown_clusters:
            cluster_sections={}
            for x,y,z in unknown_cluster & unknown_check_voxels:
                cluster_sections.setdefault(x,set()).add((y,z))
            area=max(len(cells)/len(path_sections[x]) for x,cells in cluster_sections.items())
            if area>=self.cs_occ_ratio:
                needs_information=True
                break
        return pathvoxel,blocking_clusters,unknown_clusters,needs_information

    def _camera_missing(self,unknown_voxels:set[tuple[int,int,int]],sensor_range:float)->int:
        indices=list(unknown_voxels)
        observations=self.wminterface.execute(FCExecute(function="states_scores",arguments={"indices":indices}),submodel="voxels_camera").data
        position=self.vainterface.execute(FCExecute(function="get_state",arguments={})).data.data["position"]
        missing=0
        for index,observation in zip(indices,observations):
            if observation is not None:
                continue
            point=self.wminterface.execute(FCExecute(function="voxel_to_position",arguments={"index":index}),submodel="voxels_master").data
            if (point.x-position.x)**2+(point.y-position.y)**2+(point.z-position.z)**2<=sensor_range**2:
                missing+=1
        return missing

    def navigate(self,command:Command,destination:Position,orientation:Orientation,frame_id:str,sensor_range:float=5.0,observation_probability:float=0.7,distance_cost_weight:float=0.2,allow_scan:bool=False,completed_views:list[tuple[Position,float]] | None=None)->dict[str,Any]:
        self.camea_info=rospy.wait_for_message("camera_info",CameraInfo,timeout=2.0)
        try:
            prediction=self.predict_navigation(command,destination,orientation,frame_id)
        except (RuntimeError,TypeError,ValueError) as error:
            return {"Safe":False,"Reason":f"Navigation prediction failed: {error}"}

        try:
            pathvoxel,blocking_clusters,unknown_clusters,needs_information=self._route_check(prediction.potential_voxels)
        except ValueError as error:
            return {"Safe":False,"Reason":str(error)}
        if blocking_clusters:
            return {"Safe":False,"Reason":"Objects are present within the planned route","Blocked Clusters":blocking_clusters,"Needs Replan":True}
        unknown_voxels=pathvoxel.unknown_voxels
        if needs_information:
            if self.frustration>self.frustration_max:
                self.frustration=0
                return {"Safe":False,"Reason":"Observation limit reached; avoiding unresolved space.","Needs Replan":True,"Blocked Clusters":unknown_clusters}
            if allow_scan:
                missing_camera=self._camera_missing(unknown_voxels,sensor_range)
                if missing_camera:
                    return {"Safe":False,"Reason":"Unknown route space lacks camera observations.","Needs Information":True,"Needs Scan":True,"Missing Camera Voxels":missing_camera}
            vp=self.vainterface.execute(FCExecute(function="vantagepoint",arguments={"pathvoxel":pathvoxel})).data
            center_position=self.wminterface.execute(FCExecute(function="voxel_to_position",arguments={"index":self.center(pathvoxel)}),submodel="voxels_master").data
            total_unknown_value=float(len(unknown_voxels))
            reference_distance=max(prediction.distance,1.0)
            options=[]
            candidate_counts={
                "total":len(vp),
                "destination_not_free":0,
                "known_free":0,
                "evaluated":0,
                "already_observed":0,
                "target_footprint_check_failed":0,
                "target_footprint_unsafe":0,
                "planning_failed":0,
                "empty_path":0,
                "path_occupied_cluster":0,
                "path_unknown_cluster":0,
                "accepted":0,
                "time_budget_exhausted":False,
                "evaluation_seconds":0.0,
            }

            vehicle_state=self.vainterface.execute(FCExecute(function="get_state",arguments={})).data
            robot_position=vehicle_state.data["position"]
            known_free_candidates=[]
            candidate_indices=list(vp)
            candidate_states=self.wminterface.execute(FCExecute(function="get_voxel_states",arguments={"indices":candidate_indices}),submodel="voxels_master").data
            candidate_size=self.wminterface.execute(FCExecute(function="get_voxel_size",arguments={}),submodel="voxels_master").data
            candidate_center=self.wminterface.execute(FCExecute(function="voxel_to_position",arguments={"index":(0,0,0)}),submodel="voxels_master").data
            candidate_origin=np.array([candidate_center.x,candidate_center.y,candidate_center.z],dtype=float)-candidate_size/2.0
            candidate_positions=candidate_origin+(np.asarray(candidate_indices,dtype=float).reshape(-1,3)+0.5)*candidate_size
            for vantage_voxel,vantage_state,position in zip(candidate_indices,candidate_states,candidate_positions):
                if vantage_state!=0:
                    candidate_counts["destination_not_free"]+=1
                    continue
                vantage_position=wm.Position(x=float(position[0]),y=float(position[1]),z=float(position[2]))
                distance_squared=(vantage_position.x-robot_position.x)**2+(vantage_position.y-robot_position.y)**2
                known_free_candidates.append((distance_squared,vantage_voxel,vantage_position))
            known_free_candidates.sort(key=lambda candidate:(candidate[0],candidate[1]))
            current_voxel=self.wminterface.execute(FCExecute(function="world_position_to_voxel",arguments={"position":robot_position}),submodel="voxels_master").data
            known_free_candidates=[(0.0,current_voxel,robot_position)]+[candidate for candidate in known_free_candidates if candidate[1]!=current_voxel]
            spacing=0.25
            selected={}
            for candidate in known_free_candidates:
                position=candidate[2]
                key=(math.floor(position.x/spacing),math.floor(position.y/spacing))
                selected.setdefault(key,candidate)
            known_free_candidates=list(selected.values())
            candidate_counts["known_free"]=len(known_free_candidates)
            vehicle_pose,vehicle_min,vehicle_max=self.vainterface.execute(FCExecute(function="get_collision_geometry",arguments={})).data
            if vehicle_pose is None:
                return {"Safe":False,"Reason":"Vehicle localization is not ready.","Needs Information":True,"Inform User":True,"Candidate Counts":candidate_counts,"Unknown Clusters":unknown_clusters}
            evaluation_started=time.monotonic()

            indices=list(unknown_voxels)
            positions=[self.wminterface.execute(FCExecute(function="voxel_to_position",arguments={"index":index}),submodel="voxels_master").data for index in indices]
            coordinates=np.array([(p.x,p.y,p.z) for p in positions],dtype=float).reshape(-1,3)
            visibility_states,visibility_size=self.wminterface.execute(FCExecute(function="get_voxel_snapshot",arguments={}),submodel="voxels_master").data
            visibility_occupied={index for index,state in visibility_states.items() if state==1}
            grid_center=self.wminterface.execute(FCExecute(function="voxel_to_position",arguments={"index":(0,0,0)}),submodel="voxels_master").data
            visibility_origin=(grid_center.x-visibility_size/2.0,grid_center.y-visibility_size/2.0,grid_center.z-visibility_size/2.0)
            lidar=self.wminterface.execute(FCExecute(function="states_scores",arguments={"indices":indices}),submodel="voxels_lidar").data
            camera=self.wminterface.execute(FCExecute(function="states_scores",arguments={"indices":indices}),submodel="voxels_camera").data
            lidar_uncertainty=np.array([1.0 if obs is None else max(0.0,1.0-2.0*abs(obs[1]-0.5)) for obs in lidar])
            camera_uncertainty=np.array([1.0 if obs is None else max(0.0,1.0-2.0*abs(obs[1]-0.5)) for obs in camera])
            camera_present=np.array([obs is not None for obs in camera],dtype=bool)
            fov=2.0*math.atan(self.camea_info.width/(2.0*self.camea_info.K[0]))
            fov_boundary=math.cos(fov/2.0)

            candidate_positions=np.array([(p.x,p.y,p.z) for _,_,p in known_free_candidates],dtype=float).reshape(-1,3)
            offsets=coordinates[None,:,:]-candidate_positions[:,None,:]
            within_range=np.sum(offsets*offsets,axis=2)<=sensor_range**2
            candidate_visible=np.zeros(within_range.shape,dtype=bool)
            for candidate_index,position in enumerate(candidate_positions):
                ray_origin=(float(position[0]),float(position[1]),float(position[2]))
                ray_start=tuple(math.floor((ray_origin[i]-visibility_origin[i])/visibility_size) for i in range(3))
                for voxel_index in np.flatnonzero(within_range[candidate_index]):
                    candidate_visible[candidate_index,voxel_index]=self._raycast_voxeels(ray_origin,coordinates[voxel_index],visibility_origin,visibility_size,visibility_occupied,ray_start)

            bearings=torch.as_tensor(np.arctan2(offsets[:,:,1],offsets[:,:,0]),dtype=torch.float32,device=self.device)
            uncertainty=torch.as_tensor(camera_uncertainty,dtype=torch.float32,device=self.device)
            visible=torch.as_tensor(candidate_visible&(np.sum(offsets[:,:,:2]**2,axis=2)>1e-18),dtype=torch.bool,device=self.device)
            yaw=torch.tensor(np.arctan2(center_position.y-candidate_positions[:,1],center_position.x-candidate_positions[:,0]),dtype=torch.float32,device=self.device,requires_grad=True)
            evidence=uncertainty[None,:]*visible
            scale=evidence.sum(dim=1).clamp_min(1.0)
            optimizer=torch.optim.Adam([yaw],lr=0.05)
            best_yaw=yaw.detach().clone()
            best_reward=torch.full_like(yaw,-torch.inf)
            with torch.no_grad():
                for iteration in range(self.optim_iterations):
                    difference=bearings-yaw[:,None]
                    weights=torch.sigmoid(self.k*(torch.cos(difference)-fov_boundary))
                    reward=(evidence*weights).sum(dim=1)
                    improved=reward>best_reward
                    best_reward=torch.where(improved,reward,best_reward)
                    best_yaw=torch.where(improved,yaw,best_yaw)
                    if iteration==self.optim_iterations-1:
                        break
                    gradient=self.k*(evidence*weights*(1.0-weights)*torch.sin(difference)).sum(dim=1)
                    optimizer.zero_grad(set_to_none=True)
                    yaw.grad=-gradient/scale
                    optimizer.step()
            best_yaw=torch.atan2(torch.sin(best_yaw),torch.cos(best_yaw)).cpu().numpy()
            camera_rewards=best_reward.cpu().numpy()

            for candidate_index,(_,vantage_voxel,vantage_position) in enumerate(known_free_candidates):
                candidate_counts["evaluated"]+=1
                if candidate_counts["evaluated"]==1 or candidate_counts["evaluated"]%10==0 or candidate_counts["evaluated"]==len(known_free_candidates):
                    rospy.loginfo("Vantage evaluation progress: evaluating candidate %d/%d; %d accepted so far.",candidate_counts["evaluated"],len(known_free_candidates),candidate_counts["accepted"])
                in_range=candidate_visible[candidate_index]
                observable_voxels={indices[i] for i in np.flatnonzero(in_range)}
                if not observable_voxels:
                    continue
                vantage_yaw=float(best_yaw[candidate_index])
                if any(math.hypot(vantage_position.x-p.x,vantage_position.y-p.y)<0.10 and abs(math.atan2(math.sin(vantage_yaw-yaw),math.cos(vantage_yaw-yaw)))<math.radians(15) for p,yaw in (completed_views or [])):
                    candidate_counts["already_observed"]+=1
                    continue
                vantage_orientation=wm.Orientation(pitch=0.0,yaw=vantage_yaw,roll=0.0)
                try:
                    occupancy_min=vehicle_min.copy()
                    occupancy_min[2]+=0.03
                    target_voxels=self.wminterface.execute(FCExecute(function="footprint_at",arguments={"position":vantage_position,"orientation":vantage_orientation,"vehicle_min":occupancy_min,"vehicle_max":vehicle_max}),submodel="voxels_master").data
                    target_observations=self.wminterface.execute(FCExecute(function="states_scores",arguments={"indices":target_voxels}),submodel="voxels_master").data
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
                if not vantage_prediction.potential_voxels:
                    candidate_counts["empty_path"]+=1
                    continue
                try:
                    _,vantage_blocked,_,vantage_unknown=self._route_check(vantage_prediction.potential_voxels)
                except ValueError:
                    candidate_counts["planning_failed"]+=1
                    continue
                if vantage_blocked:
                    candidate_counts["path_occupied_cluster"]+=1
                    continue
                if vantage_unknown and math.hypot(vantage_position.x-robot_position.x,vantage_position.y-robot_position.y)>=0.10:
                    candidate_counts["path_unknown_cluster"]+=1
                    continue

                lidar_reward=self.lid_wgt*float(lidar_uncertainty[in_range].sum())
                camera_reward=self.cam_wgt*float(camera_rewards[candidate_index])
                information_reward=lidar_reward+camera_reward
                novelty=self.wminterface.execute(FCExecute(function="get_observation_novelty",arguments={"position":vantage_position,"arrival_seconds":vantage_prediction.distance/self.vantage_prediction_speed}),submodel="voxels_master").data
                path_cost=distance_cost_weight*(vantage_prediction.distance/reference_distance)
                normlaiz=0.7/((self.lid_wgt+self.cam_wgt)*total_unknown_value)
                turn_cost=(1.0-math.cos(vantage_orientation.yaw-next((math.atan2(dy,dx) for a,b in zip(vantage_prediction.path.poses[-2::-1],vantage_prediction.path.poses[:0:-1]) for dx,dy in [(b.pose.position.x-a.pose.position.x,b.pose.position.y-a.pose.position.y)] if math.hypot(dx,dy)>1e-9),vehicle_state.data["orientation"].yaw)))/2.0
                profit=normlaiz*novelty*information_reward-(self.move_cost_wgt*path_cost)-(self.turn_cost_wgt*turn_cost)
                options.append({
                    "Vantage Voxel":vantage_voxel,
                    "Position":vantage_position,
                    "Orientation":vantage_orientation,
                    "Path Distance":vantage_prediction.distance,
                    "Observable Voxels":observable_voxels,
                    "Information Reward":information_reward,
                    "Observation Novelty":novelty,
                    "Observation Correlation":1.0-novelty,
                    "Path Cost":path_cost,
                    "Profit":profit,
                })
                candidate_counts["accepted"]+=1

            candidate_counts["evaluation_seconds"]=round(time.monotonic()-evaluation_started,3)
            rospy.loginfo("Vantage candidates: total=%d, destination_not_free=%d, known_free=%d, evaluated=%d, target_check_failed=%d, target_unsafe=%d, planning_failed=%d, empty_path=%d, path_occupied_cluster=%d, path_unknown_cluster=%d, accepted=%d, time_budget_exhausted=%s, evaluation_seconds=%.3f.",candidate_counts["total"],candidate_counts["destination_not_free"],candidate_counts["known_free"],candidate_counts["evaluated"],candidate_counts["target_footprint_check_failed"],candidate_counts["target_footprint_unsafe"],candidate_counts["planning_failed"],candidate_counts["empty_path"],candidate_counts["path_occupied_cluster"],candidate_counts["path_unknown_cluster"],candidate_counts["accepted"],candidate_counts["time_budget_exhausted"],candidate_counts["evaluation_seconds"])
            if not options:
                return {"Safe":False,"Reason":"No unvisited vantage points available." if candidate_counts["already_observed"] else "No vantage points available.","Needs Information":True,"Inform User":True,"Candidate Counts":candidate_counts,"Unknown Clusters":unknown_clusters}
            best_option=max(options,key=lambda option:option["Profit"])
            if best_option["Profit"]<=0:
                return {"Safe":False,"Reason":"No vantage points have an expected value greater than cost.","Needs Information":True,"Inform User":True,"Candidate Counts":candidate_counts,"Vantage Options":options,"Unknown Clusters":unknown_clusters}
            return {"Safe":False,"Reason":"Route requires more information.","Needs Information":True,"Inform User":False,"Candidate Counts":candidate_counts,"Best Vantage Point":best_option,"Vantage Options":options,"Prediction":prediction,"Assessment":pathvoxel,"Unknown Clusters":unknown_clusters}

        return {"Safe":True,"Reason":"Route accepted.","Prediction":prediction,"Assessment":pathvoxel}
        
    def _raycast_voxeels(self,origin_world:tuple[float,float,float],endpoint_world:np.ndarray,grid_origin:tuple[float,float,float],voxel_size:float,occupied:set[tuple[int,int,int]],start_voxel:tuple[int,int,int])->bool:
        """Check a prepared ray, excluding its origin and endpoint cells."""
        ox,oy,oz=origin_world
        px,py,pz=(float(value) for value in endpoint_world)
        gx,gy,gz=grid_origin
        x,y,z=start_voxel
        target=(math.floor((px-gx)/voxel_size),math.floor((py-gy)/voxel_size),math.floor((pz-gz)/voxel_size))
        if start_voxel==target:
            return True
        dx,dy,dz=px-ox,py-oy,pz-oz
        sx,sy,sz=(1 if delta>0 else -1 if delta<0 else 0 for delta in (dx,dy,dz))
        tx=math.inf if dx==0 else (gx+(x+(sx>0))*voxel_size-ox)/dx
        ty=math.inf if dy==0 else (gy+(y+(sy>0))*voxel_size-oy)/dy
        tz=math.inf if dz==0 else (gz+(z+(sz>0))*voxel_size-oz)/dz
        dtx,dty,dtz=(voxel_size/abs(delta) if delta else math.inf for delta in (dx,dy,dz))
        for _ in range(abs(target[0]-x)+abs(target[1]-y)+abs(target[2]-z)+3):
            crossing=min(tx,ty,tz)
            if abs(tx-crossing)<=1e-12:
                x+=sx
                tx+=dtx
            if abs(ty-crossing)<=1e-12:
                y+=sy
                ty+=dty
            if abs(tz-crossing)<=1e-12:
                z+=sz
                tz+=dtz
            index=(x,y,z)
            if index==target:
                return True
            if index in occupied:
                return False
        raise RuntimeError("Voxel ray-casting exceeded its expected step count")

