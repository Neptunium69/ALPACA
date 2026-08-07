from . import vlminterface
from . import worldmodel
from . import tools
from . import userinterface
from . import vehicleadapter
from . import risk
from . import yolo

import rospy
from pathlib import Path

# Prompts and stuff
prompt_path=Path(__file__).resolve().with_name("system_prompt_master.txt")
system_prompt_master=prompt_path.read_text(encoding="utf-8").strip()
system_prompt_think="\nEnter PLANNING MODE. Think of a plan of what you should do next but don't write commands yet."
system_prompt_act="\nOutput a set of JSON commands following the command format for the next step."
project_root=Path(__file__).resolve().parent.parent
model_path=str(project_root/"Models"/"Qwen3-VL-4B-Instruct")
yolo_path="./ALPACA/yolomodel/best.pt"

def main():
    # Initialize stuff
    rospy.init_node("alpaca", anonymous=True)
    observations=worldmodel.Model()
    model=worldmodel.WorldModel(observations,["apple","board"])
    initial_state=vehicleadapter.State(battery_level=float("nan"),position=worldmodel.Position(0.0,0.0,0.0),orientation=worldmodel.Orientation(0.0,0.0,0.0))
    adapter=vehicleadapter.UGVAdapter(initial_state,model,None)
    rcm=risk.RiskCal(model,adapter)
    adapter.riskcal=rcm
    detector=yolo.YOLOAdapter(model_path=yolo_path)
    vlm_interface=vlminterface.VLMInterface(system_prompt_think=system_prompt_think, system_prompt_master=system_prompt_master,system_prompt_act=system_prompt_act,model_path=model_path,toolexe=None)
    vlm_interface.init_vlm()
    toolexecutor=tools.ToolExecutor(model,adapter, rcm,vlm_interface.output_text_publisher,vlm_interface.output_image_publisher,vlm_interface.input_subscriber, detector)
    vlm_interface.toolexe=toolexecutor
    ui=userinterface.UI()

    # Initialize the stuff that just got initialized
    model.initwm()
    detector.init_yolo()
    adapter.init_sensordata()
    ui.initui()

    vlm_interface.run_mission(user_prompt="Find the apple behind a board. After you are done, send me a message that you found it and an image of the apple.")

    rospy.spin()

if __name__ == "__main__":
    main()
