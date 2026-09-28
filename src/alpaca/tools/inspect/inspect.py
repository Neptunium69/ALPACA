import math
from importlib.resources import files

from ... import vehicleadapter as va
from ...dataclasses import FCExecute
from ...vatools import VAInterface
from ...wmtools.wminterface import WorldModelInterface
from ..toolshared import Command,Return,Tool,ToolArgument,ToolSpecification


class Inspect(Tool):
    def __init__(self,wminterface:WorldModelInterface,vainterface:VAInterface,navigate_tool:Tool,scan_tool:Tool):
        self.wminterface=wminterface
        self.vainterface=vainterface
        self.adapter=vainterface.vehicle_adapter
        self.navigate_tool=navigate_tool
        self.scan_tool=scan_tool
        self.specification=ToolSpecification(
            name="Inspect",
            description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[
                ToolArgument(name="target",required=True,type="str"),
            ],
        )

    def run(self,command:Command)->Return:        
        try:
            state=self.adapter.get_state()
            targets=self.adapter.hiddenpoints(object_id=command.arguments["target"],state=state,distance_behind=1)
            targetobjects=self.wminterface.execute(FCExecute(function="query_world",arguments={"object_id":command.arguments["target"],"class_name":None})).data
            if len(targetobjects)!=1:
                raise KeyError(f"Object {command.arguments['target']!r} does not uniquely exist in the current world model.")
            targetobject=targetobjects[0]
            cmd=None
            result=None
            failures=[]
            for target_index,target in enumerate(targets,1):
                targetpose=va.Orientation(pitch=0.0,yaw=math.atan2(targetobject.position.y-target.y,targetobject.position.x-target.x),roll=0.0)
                cmd=Command(id=f"{command.id}_navigate_{target_index}",command_type="navigate", arguments={"destination":{"x":target.x,"y":target.y,"z":target.z},
                    "orientation":{"pitch":targetpose.pitch,"yaw":targetpose.yaw,"roll":targetpose.roll},
                    "frame_id":self.adapter.navigation_frame,})
                result=self.navigate_tool.run(cmd)
                if result is not None and result.status=="Success":
                    break
                if result is not None:
                    failures.append({"Candidate":target_index,"Destination":{"x":target.x,"y":target.y,"z":target.z},"Error":result.results})
                    if result.results.get("Retryable") is not True:
                        return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Notes":"Navigation related error","Error":result.results,"Candidate Failures":failures})
            if result is not None:
                if result.status!="Success":
                    return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Notes":"Navigation related error","Error":result.results,"Candidate Failures":failures})
            else:
                return Return(id=command.id, command_type=command.command_type,status="Failed",results={"Error":"All of the navigation commands failed for some reason."})
            cmd=Command(id=f"{command.id}_scan", command_type="scan", arguments={})
            result=self.scan_tool.run(cmd)
            if result.status!="Success":
                return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Notes": "Scan related error.", "Error":result.results})
            return Return(id=command.id, command_type=command.command_type, status="Success", results={"Inspection":"Completed","Major Changes":result.results.get("Major Changes"),"Candidate Failures":failures})
        except(KeyError,ValueError,TypeError) as error:
            return Return(id=command.id, command_type=command.command_type, status="Failed", results={"Error": str(error)})
