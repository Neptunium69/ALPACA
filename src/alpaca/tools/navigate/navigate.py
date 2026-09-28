import math
from importlib.resources import files

from ... import vehicleadapter as va
from ...risk import RiskCal
from ...vatools import VAInterface
from ..toolshared import Command,Return,Tool,ToolArgument,ToolSpecification


class Navigate(Tool):
    def __init__(self,vainterface:VAInterface,risk:RiskCal):
        self.vainterface=vainterface
        self.adapter=vainterface.vehicle_adapter
        self.risk=risk
        self.redirects_left=3
        self.specification=ToolSpecification(
            name="Navigate",
            description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[
                ToolArgument(name="destination",required=True,type="dict"),
                ToolArgument(name="orientation",required=True,type="dict"),
                ToolArgument(name="frame_id",required=False,type="str"),
            ],
        )

    def run(self,command:Command)->Return:
        retryable_markers=("move_base returned an empty path","target footprint intersects","path contains excessive obstacles","route requires more information","no vantage points","expected value greater than cost","navigation failed with action status 4","navigation timed out")
        try:
            destination_data=command.arguments.get("destination")
            orientation_data=command.arguments.get("orientation")
            if not isinstance(destination_data,dict) or not isinstance(orientation_data,dict):
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"Destination and orientation must be dictionaries."})
            if set(destination_data)!={"x","y","z"} or set(orientation_data)!={"pitch","yaw","roll"}:
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"Destination must contain exactly x, y, and z; orientation must contain exactly pitch, yaw, and roll."})
            goal_values=[destination_data["x"],destination_data["y"],destination_data["z"],orientation_data["roll"],orientation_data["pitch"],orientation_data["yaw"]]
            if any(not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) for value in goal_values):
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"Destination and orientation values must be finite numbers."})
            frame_id=command.arguments.get("frame_id",self.adapter.navigation_frame)
            if not isinstance(frame_id,str) or not frame_id.strip():
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"frame_id must be a non-empty string."})
            destination=va.Position(x=float(destination_data["x"]),y=float(destination_data["y"]),z=float(destination_data["z"]))
            orientation=va.Orientation(pitch=float(orientation_data["pitch"]),yaw=float(orientation_data["yaw"]),roll=float(orientation_data["roll"]))
            self.adapter.require_navigation_ready("navigate")
            risk_result=self.risk.checkcmd(command,destination,orientation,frame_id)
            if risk_result.get("Safe") is not True:
                reason=risk_result.get("Reason","Command failed the safety check.")
                if "move_base returned an empty path" in reason:
                    reason+=" The exact destination is not traversable. If it was copied from a known object's position, use inspect with that object's ID instead; object coordinates describe occupied geometry, not a safe waypoint."
                best_vantage=risk_result.get("Best Vantage Point")
                if risk_result.get("Needs Information") is not True or best_vantage is None:
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":reason,"Inform User":risk_result.get("Inform User",False),"Retryable":any(marker in reason.lower() for marker in retryable_markers)})
                if self.redirects_left<=0:
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":"Maximum number of vantage-point redirects reached.","Original Reason":reason,"Retryable":True})
                if not isinstance(best_vantage,dict):
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"The risk model returned an invalid vantage-point record.","Retryable":False})
                destination_new=best_vantage.get("Position")
                orientation_new=best_vantage.get("Orientation")
                if not isinstance(destination_new,va.Position) or not isinstance(orientation_new,va.Orientation):
                    return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":"The risk model returned invalid vantage-point coordinates.","Retryable":False})
                cmd_new=Command(f"{command.id}_vantage","navigate",{
                    "destination":{"x":destination_new.x,"y":destination_new.y,"z":destination_new.z},
                    "orientation":{"pitch":orientation_new.pitch,"yaw":orientation_new.yaw,"roll":orientation_new.roll},
                    "frame_id":frame_id,
                })
                self.redirects_left-=1
                try:
                    result=self.run(cmd_new)
                    if result.status!="Success":
                        return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":"Unable to navigate to a safe vantage point.","Original Reason":reason,"Vantage Result":result.results,"Retryable":result.results.get("Retryable") is True})
                    return self.run(command)
                finally:
                    self.redirects_left+=1
            self.adapter.navigate(destination,orientation,frame_id)
        except va.NavigationReadinessError as error:
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":str(error),"Category":"navigation_not_ready","Retryable":False})
        except Exception as error:
            reason=str(error)
            return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Error":reason,"Retryable":isinstance(error,TimeoutError) or any(marker in reason.lower() for marker in retryable_markers)})
        return Return(id=command.id,command_type=command.command_type,status="Success",results={"Navigation":"Completed"})
