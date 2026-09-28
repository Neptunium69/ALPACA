import math
from importlib.resources import files

import numpy as np

from ...dataclasses import FCExecute
from ...risk import RiskCal
from ...vatools import VAInterface
from ..toolshared import Command,Return,Tool,ToolArgument,ToolSpecification


class Image(Tool):
    def __init__(self,vainterface:VAInterface,risk:RiskCal,observe_tool:Tool):
        self.vainterface=vainterface
        self.risk=risk
        self.observe_tool=observe_tool
        self.specification=ToolSpecification(name="Image",description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),arguments=[ToolArgument(name="target",required=True,type="str")])

    def run(self,command:Command)->Return:
        try:
            state=self.vainterface.execute(FCExecute(function="get_state",arguments={})).data
            target_obj=command.arguments.get("target")
            if target_obj is not None and state is not None:
                target=self.vainterface.execute(FCExecute(function="hiddenpoint",arguments={"object_id":target_obj,"state":state})).data
                positionvector=np.array([state.position.x,state.position.y,state.position.z])
                if target is not None:
                    directionvector=np.array([target.x,target.y,target.z])-positionvector
                else:
                    raise ValueError("Target does not exist.")
                if math.hypot(directionvector[0], directionvector[1]) < 1e-6:
                    raise ValueError("Vehicle position is equal to target position. Thus, position is undefined.")
                yaw=math.atan2(directionvector[1],directionvector[0])
                yaw_error=(yaw-state.orientation.yaw+math.pi)%(2*math.pi)-math.pi
                if abs(yaw_error)>0.1:
                    try:
                        risk_result=self.risk.checkcmd(command)
                        if risk_result.get("Safe") is False:
                            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":risk_result.get("Reason","Command failed the safety check.")})
                        self.vainterface.execute(FCExecute(function="image",arguments={"targpos":target}))
                    except Exception as error:
                        return Return(id=command.id, command_type=command.command_type, status="Failed",results={"error":str(error)})
                result=self.observe_tool.run(Command(id=f"{command.id}",command_type=command.command_type,arguments={}))
                if result.status=="Failed":
                    return Return(id=command.id, command_type=command.command_type, status="Failed",results=result.results)
                else:
                    return Return(id=command.id,command_type=command.command_type,status="Success",results=result.results)
            else:
                raise ValueError(f"Object {target_obj} is not found.")

        except(KeyError,ValueError,TypeError) as error:
            return Return(id=command.id,command_type=command.command_type,status="Failed", results={"Error":str(error)})
