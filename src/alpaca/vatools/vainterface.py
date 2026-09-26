from __future__ import annotations

from typing import TYPE_CHECKING

from .dataclasses import VAReturn,VAExecute

if TYPE_CHECKING:
    from ..vehicleadapter import VehicleAdapter

class VAInterface():
    def __init__(self,vehicle_adapter:VehicleAdapter,vaname:str):
        self.vehicle_adapter=vehicle_adapter
        self.vaname=vaname

    @staticmethod
    def _matches_type(value:object,expected_type:str)->bool:
        actual_type=type(value)
        actual_type_names={
            actual_type.__name__,
            actual_type.__qualname__,
            f"{actual_type.__module__}.{actual_type.__qualname__}",
        }
        return expected_type in actual_type_names

    def execute(self,cmd:VAExecute)->VAReturn:
        if not isinstance(cmd.arguments,dict):
            raise ValueError("Vehicle adapter command arguments must be a dictionary.")

        functions=self.vehicle_adapter.registry.tools
        if cmd.function not in [fcn.name for fcn in functions]:
            raise ValueError(f"{cmd.function} is not valid for vehicle adapter {self.vaname}.")
        functionschema=next(fcn for fcn in functions if fcn.name==cmd.function)

        for name,specification in functionschema.arguments.items():
            if not isinstance(specification,tuple) or len(specification)!=2:
                raise ValueError(f"Invalid registry specification for argument {name!r} in {functionschema.name}.")
            expected_type,required=specification
            if not isinstance(expected_type,str) or not isinstance(required,bool):
                raise ValueError(f"Invalid registry specification for argument {name!r} in {functionschema.name}.")
            if required and (name not in cmd.arguments or cmd.arguments[name] is None):
                raise ValueError(f"Missing required argument {name!r} for {functionschema.name}.")

        for name,data in cmd.arguments.items():
            if name not in functionschema.arguments:
                raise ValueError(f"{name} is not valid argument for {functionschema.name}.")
            expected_type,required=functionschema.arguments[name]
            if data is None and not required:
                continue
            if not self._matches_type(data,expected_type):
                raise ValueError(f"{name} must be of type {expected_type} for {functionschema.name}.")

        function=getattr(self.vehicle_adapter,cmd.function,None)
        if not callable(function):
            raise ValueError(f"Registered function {cmd.function!r} is not callable on vehicle adapter {self.vaname}.")

        returned_data=function(**cmd.arguments)
        return VAReturn(function=cmd.function,data=returned_data)
