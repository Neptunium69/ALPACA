from __future__ import annotations

from dataclasses import dataclass
from typing import Any,Protocol
import warnings
import json

@dataclass(frozen=True)
class Command:
    id: str
    command_type: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class Return:
    id: str
    command_type: str
    status: str
    results: dict[str, Any]

@dataclass(frozen=True)
class ToolArgument:
    name: str
    required: bool
    type: str

@dataclass(frozen=True)
class ToolSpecification:
    name:str
    description:str
    arguments: list[ToolArgument]
    tag:str

@dataclass(frozen=True)
class ExecutionRule:
    tag:str
    maximum:int

class Tool(Protocol):
    specification: ToolSpecification

    def run(self,cmd:Command)->Return:
        """Method to run this tool."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class ToolList():
    def __init__(self)->None:
        self._tools:dict[str,Tool]={}
    def add(self,tool:Tool)->None:
        try:
            if(self._tools.get(tool.specification.name) is None):
                self._tools[tool.specification.name]=tool
            else:
                raise ValueError(f"Tool with name {tool.specification.name} already exists. No changes were made.")
        except Exception as error:
            warnings.warn(f"Tool {tool.specification.name} not added due to error: {error}")
    def remove(self,toolname:str)->None:
        if toolname in self._getallnames():
            del self._tools[toolname]
        else:
            warnings.warn(f"Tool {toolname} is not present, no changes were made.")
    def get(self,toolname:str)->Tool|None:
        val=self._tools.get(toolname)
        if val is None:
            warnings.warn(f"Tool {toolname} not present.")
            return None
        else:
            return val
    def getall(self)->dict[str,Tool]:
        return self._tools
    
    def getspec(self,toolname:str)->ToolSpecification|None:
        tool=self._tools.get(toolname)
        return tool.specification if tool is not None else None
    def getallspec(self)->list[ToolSpecification]:
        spec=[]
        for tool in self._tools.values():
            spec.append(tool.specification)
        return spec
    def _getallnames(self)->list[str]:
        return [spec.name for spec in self.getallspec()]

class ToolExecutor():
    def __init__(self,toolist:ToolList,rules:list[ExecutionRule]|None=None):
        self.toolist=toolist
        self.rules=rules if rules is not None else [ExecutionRule(tag="action",maximum=1)]

    def read(self,cmds:str)->list[Command]:
        envelope=json.loads(cmds)
        if not isinstance(envelope,dict):
            raise ValueError("Tool input must be a JSON object.")

        command_data=envelope.get("commands")
        if not isinstance(command_data,list):
            raise ValueError("commands must be a list.")

        commands=[]
        for index,item in enumerate(command_data):
            if not isinstance(item,dict):
                raise ValueError(f"Command {index} must be an object.")
            if set(item)!={"id","command_type","arguments"}:
                raise ValueError(
                    f"Command {index} must contain exactly id, command_type, and arguments."
                )

            commands.append(Command(
                id=item["id"],
                command_type=item["command_type"],
                arguments=item["arguments"],
            ))

        return commands

    @staticmethod
    def _matches_type(value:Any,expected_type:str)->bool:
        type_checks={
            "str": lambda item:isinstance(item,str),
            "int": lambda item:isinstance(item,int) and not isinstance(item,bool),
            "float": lambda item:isinstance(item,(int,float)) and not isinstance(item,bool),
            "number": lambda item:isinstance(item,(int,float)) and not isinstance(item,bool),
            "bool": lambda item:isinstance(item,bool),
            "dict": lambda item:isinstance(item,dict),
            "list": lambda item:isinstance(item,list),
            "Any": lambda item:True,
            "any": lambda item:True,
        }
        check=type_checks.get(expected_type)
        return check(value) if check is not None else False

    def _validate_command(self,command:Command,tool:Tool)->str|None:
        if not isinstance(command.id,str) or not command.id.strip():
            return "Command id must be a non-empty string."
        if not isinstance(command.command_type,str) or not command.command_type.strip():
            return "Command type must be a non-empty string."
        if not isinstance(command.arguments,dict):
            return "Command arguments must be an object."

        specification=tool.specification
        invalid_required_values=[
            argument.name
            for argument in specification.arguments
            if not isinstance(argument.required,bool)
        ]
        if invalid_required_values:
            return (
                "Tool arguments must use required=True or required=False: "
                f"{', '.join(invalid_required_values)}."
            )

        argument_specifications={argument.name:argument for argument in specification.arguments}
        if len(argument_specifications)!=len(specification.arguments):
            return f"Tool {specification.name} has duplicate argument specifications."

        unexpected_arguments=sorted(set(command.arguments)-set(argument_specifications))
        if unexpected_arguments:
            return f"Unexpected arguments: {', '.join(unexpected_arguments)}."

        missing_arguments=[
            argument.name
            for argument in specification.arguments
            if argument.required
            and (argument.name not in command.arguments or command.arguments[argument.name] is None)
        ]
        if missing_arguments:
            return f"Missing required arguments: {', '.join(missing_arguments)}."

        for name,value in command.arguments.items():
            if value is None and not argument_specifications[name].required:
                continue
            expected_type=argument_specifications[name].type
            if not self._matches_type(value,expected_type):
                return f"Argument {name} must be of type {expected_type}."

        return None

    def run(self, cmds:str)->list[Return]:
        """Runs tools"""
        try:
            commands=self.read(cmds)
        except (json.JSONDecodeError,TypeError,ValueError) as error:
            return [Return(
                id="N/A",
                command_type="N/A",
                status="Failed",
                results={"Error":str(error)},
            )]

        if not commands:
            return [Return(
                id="N/A",
                command_type="N/A",
                status="Failed",
                results={"Error":"No commands found."},
            )]

        returns=[]
        command_ids=set()
        tag_counts:dict[str,int]={}

        for command in commands:
            if not isinstance(command.id,str) or not command.id.strip():
                returns.append(Return(
                    id="N/A",
                    command_type=command.command_type if isinstance(command.command_type,str) else "N/A",
                    status="Failed",
                    results={"Error":"Command id must be a non-empty string."},
                ))
                continue
            if not isinstance(command.command_type,str) or not command.command_type.strip():
                returns.append(Return(
                    id=command.id,
                    command_type="N/A",
                    status="Failed",
                    results={"Error":"Command type must be a non-empty string."},
                ))
                continue
            if not isinstance(command.arguments,dict):
                returns.append(Return(
                    id=command.id,
                    command_type=command.command_type,
                    status="Failed",
                    results={"Error":"Command arguments must be an object."},
                ))
                continue

            if command.id in command_ids:
                returns.append(Return(
                    id=command.id,
                    command_type=command.command_type,
                    status="Failed",
                    results={"Error":f"Duplicate command id: {command.id}"},
                ))
                continue
            command_ids.add(command.id)

            tool=next(
                (
                    registered_tool
                    for name,registered_tool in self.toolist.getall().items()
                    if name.casefold()==command.command_type.casefold()
                ),
                None,
            )

            if tool is None:
                returns.append(Return(
                    id=command.id,
                    command_type=command.command_type,
                    status="Failed",
                    results={"Error":f"Unknown tool: {command.command_type}"},
                ))
                continue

            try:
                validation_error=self._validate_command(command,tool)
            except Exception as error:
                validation_error=f"Could not validate command: {error}"
            if validation_error is not None:
                returns.append(Return(
                    id=command.id,
                    command_type=command.command_type,
                    status="Failed",
                    results={"Error":validation_error},
                ))
                continue

            violated_rule=next(
                (
                    rule
                    for rule in self.rules
                    if rule.tag==tool.specification.tag
                    and tag_counts.get(rule.tag,0)>=rule.maximum
                ),
                None,
            )
            if violated_rule is not None:
                returns.append(Return(
                    id=command.id,
                    command_type=command.command_type,
                    status="Failed",
                    results={"Error":(
                        f"Execution rule permits at most {violated_rule.maximum} "
                        f"tool(s) tagged {violated_rule.tag}."
                    )},
                ))
                continue

            tag_counts[tool.specification.tag]=tag_counts.get(tool.specification.tag,0)+1

            try:
                result=tool.run(command)
                if not isinstance(result,Return):
                    raise TypeError(
                        f"Tool {tool.specification.name} returned {type(result).__name__}, not Return."
                    )
                returns.append(result)
            except Exception as error:
                returns.append(Return(
                    id=command.id,
                    command_type=command.command_type,
                    status="Failed",
                    results={"Error":str(error)},
                ))
                
        return returns
