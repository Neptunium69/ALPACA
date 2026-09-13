from __future__ import annotations

# This is where the VLM actually runs
import rospy
from pathlib import Path
import base64
from . import tools
from std_msgs.msg import String
from threading import Event
import json
import os
import time
from dataclasses import asdict
from urllib import error as urllib_error
from urllib import request as urllib_request


class VLMInterface():
    def __init__(self,system_prompt_think:str,system_prompt_act:str,model_path:str,toolexe:tools.ToolExecutor | None=None, system_prompt_master:str|None=None,max_tok:int=512) -> None:
        self.system_prompt_act=system_prompt_act
        self.system_prompt_think=system_prompt_think
        self.model_path=Path(model_path)
        self.model_name="qwen3-vl:4b-instruct-q4_K_M"
        self.ollama_url="http://127.0.0.1:11434"
        self.request_timeout=600.0
        self.system_prompt_master=system_prompt_master
        self.toolexe=toolexe
        self.user_reply=None
        self.reply_event=Event()
        self.active=0
        self.awt=1
        self.idle=1
        self.steps=0
        self.max_tok=max_tok
        self.log_dir=Path(os.environ.get("ALPACA_LOG_DIR","/tmp/alpaca_runtime"))
        self.dialogue_log_path=self.log_dir/"qwen_dialogue.jsonl"

    def init_vlm(self):
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self._load_model()
        self.output_text_publisher=rospy.Publisher("vlm_text_output",String,queue_size=1)
        self.output_image_publisher=rospy.Publisher("vlm_image_output",String,queue_size=1)
        self.completion_publisher=rospy.Publisher("vlm_completion_output",String,queue_size=1,latch=True)
        self.step_result_publisher=rospy.Publisher("vlm_step_result",String,queue_size=10)
        self.input_subscriber=rospy.Subscriber("user_input",String,queue_size=1)

    def _load_model(self):
        version=self._request_json("/api/version",method="GET",timeout=5.0)
        self._request_json("/api/show",payload={"model":self.model_name},timeout=30.0)
        rospy.loginfo("Connected to Ollama %s using model %s",version.get("version","unknown"),self.model_name)

    def step(self):
        pass

    def _request_json(self,endpoint:str,payload:dict | None=None,method:str="POST",timeout:float | None=None)->dict:
        data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode("utf-8")
        request=urllib_request.Request(f"{self.ollama_url}{endpoint}",data=data,method=method,headers={"Content-Type":"application/json","Accept":"application/json"})
        try:
            with urllib_request.urlopen(request,timeout=self.request_timeout if timeout is None else timeout) as response:
                body=response.read().decode("utf-8")
        except urllib_error.HTTPError as error:
            details=error.read().decode("utf-8",errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {details}") from error
        except (urllib_error.URLError,TimeoutError) as error:
            raise RuntimeError(f"Could not connect to Ollama at {self.ollama_url}: {error}") from error
        try:
            result=json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Ollama returned invalid JSON: {body[:500]}") from error
        if not isinstance(result,dict):
            raise RuntimeError(f"Ollama returned an unexpected response: {result!r}")
        if result.get("error"):
            raise RuntimeError(f"Ollama error: {result['error']}")
        return result

    @staticmethod
    def _encode_image(image_path:str)->str:
        path=Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _convert_messages(self,messages:list[dict])->list[dict]:
        converted=[]
        for message in messages:
            content=message.get("content")
            text_parts=[]
            images=[]
            if isinstance(content,str):
                text_parts.append(content)
            elif isinstance(content,list):
                for part in content:
                    if not isinstance(part,dict):
                        raise ValueError("Message content parts must be dictionaries")
                    if part.get("type")=="text":
                        text=part.get("text")
                        if not isinstance(text,str):
                            raise ValueError("Text message content must be a string")
                        text_parts.append(text)
                    elif part.get("type")=="image":
                        image_path=part.get("path")
                        if not isinstance(image_path,str):
                            raise ValueError("Image message content must contain a file path")
                        images.append(self._encode_image(image_path))
                    else:
                        raise ValueError(f"Unsupported message content type: {part.get('type')!r}")
            else:
                raise ValueError("Message content must be a string or a list")
            converted_message={"role":message.get("role"),"content":"\n".join(text_parts)}
            if images:
                converted_message["images"]=images
            converted.append(converted_message)
        return converted

    def _log_dialogue(self,event:str,**payload)->None:
        record={"monotonic_seconds":time.monotonic(),"mission_step":self.steps,"event":event,**payload}
        with self.dialogue_log_path.open("a",encoding="utf-8") as log_file:
            log_file.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")

    def _predict(self,msg:list[dict],max_tok:int,stage:str):
        started=time.monotonic()
        rospy.loginfo("Qwen step %d: starting %s inference...",self.steps,stage)
        self._log_dialogue("qwen_request",stage=stage,max_tokens=max_tok,messages=msg)
        payload={
            "model":self.model_name,
            "messages":self._convert_messages(msg),
            "stream":False,
            "keep_alive":"30m",
            "options":{
                "num_ctx":8192,
                "num_predict":max_tok,
                "temperature":0.0,
                "seed":0,
            },
        }
        response=self._request_json("/api/chat",payload=payload)
        message=response.get("message")
        if not isinstance(message,dict) or not isinstance(message.get("content"),str):
            raise RuntimeError(f"Ollama returned an invalid chat response: {response}")
        output=message["content"].strip()
        self._log_dialogue("qwen_response",stage=stage,response=output,ollama_metadata={name:value for name,value in response.items() if name!="message"})
        rospy.loginfo("Qwen step %d: %s inference completed in %.1f seconds.",self.steps,stage,time.monotonic()-started)
        return output

    def run_mission(self,user_prompt:str) -> int:
        self.state=1
        self.steps=0
        if self.toolexe is None: raise ValueError("Tool executor is None.")
        self.history=[{"role":"system","content":[{"type":"text","text":self.system_prompt_master}]},
            {"role":"user","content":[{"type":"text","text":user_prompt}],},
            {"role":"user","content":[{"type":"text","text":f"World-model objects already detected at mission startup (IDs and classes, JSON):\n{self.toolexe.model.getallobjects()}"}]},
        ]
        self._log_dialogue("mission_start",user_prompt=user_prompt,history=self.history)

        while not rospy.is_shutdown():
            self.steps+=1
            if self.steps>=50:
                raise RuntimeError("Mission exceeded the maximum number of steps.")
            nukeflag=0
            think_messages=[
                {"role":"system",
                    "content":[{"type":"text","text":self.system_prompt_think}],
                },
                *self.history,
            ]
            plan=self._predict(think_messages,max_tok=256,stage="planning")

            act_messages=[{"role":"system",
                    "content":[{"type":"text","text":self.system_prompt_act}],
                },
                *self.history,
                {"role":"user",
                "content":[{"type":"text","text":f"Current plan:\n{plan}\n\nReturn the next structured commands."}],
                },
            ]
            cmdjson=self._predict(act_messages,max_tok=self.max_tok,stage="action")
            if self.toolexe is not None:
                results=[]
                try:
                    commands=self.toolexe.stringtocmd(cmdjson)
                    self._log_dialogue("commands_parsed",commands=[asdict(command) for command in commands])
                    rospy.loginfo("Qwen step %d chose: %s",self.steps,", ".join(f"{command.command_type} ({command.id})" for command in commands))
                    results=self.toolexe.execute(commands)
                    self._log_dialogue("tool_results",results=[asdict(result) for result in results])
                    rospy.loginfo("Qwen step %d results: %s",self.steps,", ".join(f"{result.command_type}={result.status}" for result in results))
                    step_payload={"step":self.steps,"results":[asdict(result) for result in results]}
                    self.step_result_publisher.publish(String(data=json.dumps(step_payload,ensure_ascii=False,default=str)))
                except ValueError as error:
                    self._log_dialogue("command_rejected",raw_response=cmdjson,error=str(error))
                    self.history.append({"role":"assistant","content":[{"type":"text","text":cmdjson}]})
                    self.history.append({"role":"user","content":[{"type":"text","text":f"Command rejected: {error}. Return corrected raw JSON only."}]})
                    continue
            else:
                raise ValueError("Tool executor is None.")

            self.history.append({"role":"assistant",
            "content":[{"type":"text","text":cmdjson}],
            })
            imagflag=0
            imag=None
            for i in range(len(results)):
                if results[i].command_type in ("image","observe") and results[i].status=="Success":
                    imag=results[i].results.get("image_path")
                    imagflag=1
            if imagflag==1:
                self.history.append({"role":"user",
                "content":[{"type":"text","text":f"Command results:\n{self._dump(results)}"},{"type":"image","path":imag}],
                })
            else:
                self.history.append({"role":"user",
                    "content":[{"type":"text","text":f"Command results:\n{self._dump(results)}"}],
                    })

            for result in results:
                if result.command_type=="finish" and result.status=="Success":
                    nukeflag=1
                    break

            if nukeflag==1:
                break
        self._log_dialogue("mission_end",history=self.history)
        return 0

    @staticmethod
    def _dump(results:list[tools.Return])->str:
        return json.dumps([asdict(result) for result in results], ensure_ascii=False)
