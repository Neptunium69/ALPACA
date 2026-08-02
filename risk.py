from __future__ import annotations

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
        if not rospy.core.is_initialized():
            rospy.init_node("RiskCal",anonymous=True)
    def checkcmd(self, command: tools.Command, adapter: va.VehicleAdapter) -> dict[str, Any]:
        state=adapter.get_state()
        safe,reason=self._hardsafetyfilter(command,state)
        if not safe:
            return {"Safe":False,"Reason":reason}
        return {"Safe":True,"Reason":reason}
    def _hardsafetyfilter(self,cmd: tools.Command, state: va.State) -> tuple[bool,str]:
        if(state.battery_level<0.15):
            return False,"Battery level too low, please contact user to recharge or replace battery."
        if(cmd.command_type=="scan"):
            try:
                clutter=rospy.wait_for_message("clutter_level",Float64,timeout=5)
                if clutter is None:
                    return False,"Sensor unresponsive."
                if(clutter.data>=0.7):
                    return False,"Vehicle was unable to scan because the clutter level is too high."
            except rospy.ROSException:
                return False,"Sensor unresponsive."
        
        return True,"Command is safe to execute."
    def _cost_reward(self):
        pass
