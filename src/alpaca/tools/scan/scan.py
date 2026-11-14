try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import numpy as np
import rospy

from ...dataclasses import FCExecute
from ...risk import RiskCal
from ...vatools import VAInterface
from ...wmtools.wminterface import WorldModelInterface
from ..toolshared import Command,Return,Tool,ToolSpecification

class Scan(Tool):
    def __init__(self,wminterface:WorldModelInterface,vainterface:VAInterface,risk:RiskCal):
        self.wminterface=wminterface
        self.vainterface=vainterface
        self.risk=risk
        self.specification=ToolSpecification(
            name="Scan",
            description=files(__name__.rsplit(".",1)[0]).joinpath("specification.md").read_text(encoding="utf-8"),
            arguments=[],
            tag="action",
        )

    def run(self, command: Command) -> Return:
        recording_started=False
        try:
            risk_result=self.risk.checkcmd(command)
            if risk_result.get("Safe") is False:
                return Return(id=command.id,command_type=command.command_type,status="Failed",results={"Reason":risk_result.get("Reason","Command failed the safety check.")})
            self.wminterface.execute(FCExecute(function="record",arguments={}),submodel="objects")
            recording_started=True
            _=self.vainterface.execute(FCExecute(function="scan",arguments={"yaw_margin":float(np.deg2rad(45))}))
            changes=self.wminterface.execute(FCExecute(function="stop_record",arguments={}),submodel="objects").data
            recording_started=False
        except rospy.ROSInterruptException as error:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Interrupted",
                results={"Reason": str(error)},
            )
        except Exception as error:
            return Return(
                id=command.id,
                command_type=command.command_type,
                status="Failed",
                results={"Error": str(error)},
            )
        finally:
            if recording_started:
                try:
                    self.wminterface.execute(FCExecute(function="stop_record",arguments={}),submodel="objects")
                except RuntimeError:
                    pass

        return Return(
            id=command.id,
            command_type=command.command_type,
            status="Success",
            results={"Scan": "Completed",
                        "Major Changes": changes,
        })
