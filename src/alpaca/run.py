from . import vlminterface
from . import worldmodel
from . import userinterface
from . import vehicleadapter
from . import risk
from . import yolo
from .tools import ToolExecutor,ToolList
from .tools.ask_user import AskUser
from .tools.finish import Finish
from .tools.image import Image
from .tools.inspect import Inspect
from .tools.navigate import Navigate
from .tools.observe import Observe
from .tools.query import Query
from .tools.robot_state import RobotState
from .tools.scan import Scan
from .tools.write import Write
from .vatools import FCExecute,VAInterface
from .wmtools import WorldModelInterface

import rospy
from pathlib import Path
from dataclasses import asdict
import json
import os
import time
from threading import RLock
from std_msgs.msg import Float64
from std_srvs.srv import Trigger,TriggerResponse
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import CommandBool
from alpaca_msgs.msg import ObservationFrame,SpatialObservationFrame

# Prompts and stuff
prompt_path=Path(__file__).resolve().parents[2]/"system_prompt_master.md"
system_prompt_master=prompt_path.read_text(encoding="utf-8").strip()
system_prompt_think="\nEnter PLANNING MODE. Think of a plan of what you should do next but don't write commands yet."
system_prompt_act="\nEnter ACTION MODE. Return exactly one raw JSON object following the command format for the next step. Your first character must be { and your last character must be }. Never wrap the JSON in Markdown, backticks, or code fences; never add a json language label, explanation, or other text outside the object. ALPACA parses the entire response as JSON and rejects wrappers. Check the response boundaries before sending."
# user_prompt="Find the apple behind the board."
user_prompt="Survey the accessible forest with the UGV and inspect standing trees for basal and root damage. Use the tree class to locate trees, including trees with no visible damage, and inspect their trunk bases and visible roots from safe viewpoints. Look for basal_damage and root_damage near each tree; a tree may have neither, either, or both damage classes. Associate each damage observation with a tree using visual evidence and world-model positions, and flag uncertain associations. Keep clear of trunks, roots, and other obstacles, and avoid repeatedly inspecting the same tree. Send me the IDs and locations of damaged trees, the observed damage classes, and supporting images. Do not treat missing detections as proof that a tree is healthy. When the accessible survey is complete or further inspection is blocked, summarize the findings, identify uninspected or uncertain areas, and finish the mission."
project_root=Path(__file__).resolve().parents[3]
model_path=str(project_root/"Models"/"Qwen3-VL-4B-Instruct")
yolo_path="./ALPACA/yolomodel/best.engine"

def start_world_model_logging(model:worldmodel.ObjectWorldModel,log_dir:Path):
    log_path=log_dir/"world_model_objects.jsonl"
    def log_objects(_event):
        try:
            objects=model.query_world()
            record={"monotonic_seconds":time.monotonic(),"object_count":len(objects),"objects":[asdict(world_object) for world_object in objects]}
            with log_path.open("a",encoding="utf-8") as log_file:
                log_file.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")
        except Exception as error:
            rospy.logerr("World-model logging failed: %s",error)
    return rospy.Timer(rospy.Duration(1),log_objects)

def wait_for_runtime_inputs(timeout_seconds:float=30.0)->None:
    required_topics=[("voxel_observations",SpatialObservationFrame,timeout_seconds),("clutter_level",Float64,timeout_seconds),("detections",ObservationFrame,max(timeout_seconds,90.0))]
    for topic,message_type,topic_timeout in required_topics:
        rospy.loginfo("Waiting for ALPACA runtime data on /%s...",topic)
        try:
            rospy.wait_for_message(topic,message_type,timeout=topic_timeout)
        except rospy.ROSException as error:
            message=f"No ALPACA runtime data on /{topic} within {topic_timeout:.0f} seconds."
            raise RuntimeError(message) from error
    rospy.loginfo("Voxel, clutter, and detection pipelines are live.")

def main():
    # Initialize stuff
    rospy.init_node("alpaca", anonymous=True)
    log_dir=Path(os.environ.get("ALPACA_LOG_DIR","/tmp/alpaca_runtime"))
    log_dir.mkdir(parents=True,exist_ok=True)
    observations=worldmodel.Model()
    model=worldmodel.WorldModel()
    object_model=worldmodel.ObjectWorldModel(observations,["apple","board"])
    voxel_model=worldmodel.LiDARVoxelWorldModel(name="voxels_lidar")
    camera_model=worldmodel.CameraVoxelWorldModel()
    master_voxel_model=worldmodel.MasterVoxelWorldModel(
        [camera_model,voxel_model],
        important_wms=(camera_model.name,voxel_model.name),
    )
    model.add([object_model,voxel_model,camera_model,master_voxel_model])
    wminterface=WorldModelInterface(model,"primary")
    adapter=vehicleadapter.UGVAdapter(model,None)
    vainterface=VAInterface(adapter,"primary")
    rcm=risk.RiskCal(wminterface,vainterface)
    adapter.riskcal=rcm
    detector=yolo.YOLOAdapter(model_path=yolo_path,sitl=True)
    vlm_interface=vlminterface.VLMInterface(system_prompt_think=system_prompt_think, system_prompt_master=system_prompt_master,system_prompt_act=system_prompt_act,model_path=model_path,toolexe=None)
    vlm_interface.init_vlm()
    observe_tool=Observe()
    navigate_tool=Navigate(vainterface,rcm)
    scan_tool=Scan(wminterface,vainterface,rcm)
    tool_list=ToolList()
    for tool in (
        Query(wminterface),
        RobotState(vainterface),
        Write(wminterface),
        observe_tool,
        scan_tool,
        navigate_tool,
        Inspect(wminterface,vainterface,navigate_tool,scan_tool),
        Image(vainterface,rcm,observe_tool),
        AskUser(vlm_interface.output_text_publisher,vlm_interface.output_image_publisher),
        Finish(vlm_interface.output_text_publisher,vlm_interface.output_image_publisher,completion_publisher=getattr(vlm_interface,"completion_publisher",None)),
    ):
        tool_list.add(tool)
    toolexecutor=ToolExecutor(tool_list)
    vlm_interface.toolexe=toolexecutor
    ui=userinterface.UI()

    # Initialize the stuff that just got initialized
    model.initwm()
    world_model_log_timer=start_world_model_logging(object_model,log_dir)
    detector.init_yolo()
    vainterface.execute(FCExecute(function="init_sensordata",arguments={}))
    shutdown_lock=RLock()
    adapter.stopping=False
    adapter.stopped=False
    def shutdown_hold(_request=None):
        adapter.stopping=True
        return TriggerResponse(success=True,message="Adapter switched to idle; navigation disabled until restart.")
    def stop_adapter():
        with shutdown_lock:
            if adapter.stopped: return
            shutdown_hold()
            while not rospy.is_shutdown():
                try:
                    state=rospy.wait_for_message("/mavros/state",MavrosState,timeout=2.0)
                    if not state.connected:
                        raise RuntimeError("MAVROS is disconnected; disarm cannot be confirmed.")
                    if state.armed:
                        rospy.wait_for_service("/mavros/cmd/arming",timeout=2.0)
                        response=rospy.ServiceProxy("/mavros/cmd/arming",CommandBool)(value=False)
                        if not response.success:
                            raise RuntimeError("PX4 rejected the disarm request.")
                        state=rospy.wait_for_message("/mavros/state",MavrosState,timeout=2.0)
                    if state.connected and not state.armed:
                        adapter.timer.shutdown()
                        adapter.stopped=True
                        return
                except Exception as error:
                    rospy.logerr("Disarm confirmation failed; retaining adapter stream: %s",error)
                time.sleep(1.0)
    hold_service=rospy.Service("/alpaca/px4_shutdown_hold",Trigger,shutdown_hold)
    rospy.on_shutdown(stop_adapter)
    try:
        ui.initui()
        wait_for_runtime_inputs()
        rospy.set_param("/alpaca/runtime_ready",True)
        if rospy.get_param("~wait_for_start",False):
            rospy.loginfo("Adapter streaming; waiting for launcher authorization after armed OFFBOARD.")
            while not rospy.is_shutdown() and not adapter.stopping and not rospy.get_param("/alpaca/start_mission",False):
                rospy.sleep(0.1)
            if rospy.is_shutdown() or adapter.stopping: return 0
            vainterface.execute(FCExecute(function="require_navigation_ready",arguments={"action_name":"start the mission"}))
        vlm_interface.run_mission(user_prompt=user_prompt)
        rospy.spin()
    except RuntimeError as error:
        rospy.logerr("ALPACA stopped: %s",error)
        return 1
    finally:
        stop_adapter()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
