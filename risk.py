from typing import Any

import rospy
from std_msgs.msg import Float64
from . import vehicleadapter as va
from . import worldmodel as wm
from . import tools

class RiskCal():
    def __init__(self, model: wm.WorldModel, adapter: va.VehicleAdapter) -> None:
        self.model=model
        self.adapter=adapter
    def initrcm(self, model: wm.WorldModel, adapter: va.VehicleAdapter):
        rospy.init_node("RiskCal",anonymous=True)
    def checkcmd(self, command: tools.Command, adapter: va.VehicleAdapter) -> dict[str, Any]:
        state=adapter.get_state()
        filter_rst=self._hardsafetyfilter(command, state)
        if filter_rst[0]==False:
            return {"Safe":False,"Reason":filter_rst[1]}
        result=self._cost_reward()
        return result
    def _hardsafetyfilter(self,cmd: tools.Command, state: va.State) -> list[bool,str]:
        if(state.battery_level<0.15):
            return [False,"Battery level too low, please contact user to recharge or replace battery."]
        if(cmd.command_type=="scan"):
            try:
                clutter=rospy.wait_for_message("clutter_level",Float64,timeout=5)
                if(clutter.data>0.7):
                    return [False,"Vehicle was unable to scan becasue clutter level too high."]
            except rospy.ROSException as error:
                return [False,"Sensor unresponsive."]
        
        return [True,"Command is safe to execute."]
    def _cost_reward(self):
        pass