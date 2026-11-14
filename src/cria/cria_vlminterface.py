"""Vehicle mission interface using CRIA transport and ALPACA's existing loop."""
from __future__ import annotations

from typing import Any

if __package__ == "cria":
    from alpaca.vlminterface import VLMInterface
    from alpaca.tools import ToolExecutor
else:
    from ..alpaca.vlminterface import VLMInterface
    from ..alpaca.tools import ToolExecutor
from .localrillm import DEFAULT_MODEL,VLMReciever


class CRIAVLMInterface(VLMInterface):
    def __init__(self,system_prompt_think:str,system_prompt_act:str,model_path:str,host:str,toolexe:ToolExecutor|None=None,system_prompt_master:str|None=None,max_tok:int=512,model_name:str=DEFAULT_MODEL,request_timeout:float=600.0,token:str|None=None) -> None:
        self.receiver=VLMReciever(host,model_name,request_timeout,token)
        super().__init__(system_prompt_think=system_prompt_think,system_prompt_act=system_prompt_act,model_path=model_path,toolexe=toolexe,system_prompt_master=system_prompt_master,max_tok=max_tok)
        self.model_name=self.receiver.model_name
        self.ollama_url=self.receiver.ollama_url
        self.request_timeout=self.receiver.request_timeout

    def _request_json(self,endpoint:str,payload:dict[str,Any]|None=None,method:str="POST",timeout:float|None=None)->dict[str,Any]:
        return self.receiver._request_json(endpoint,payload=payload,method=method,timeout=timeout)
