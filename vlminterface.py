from __future__ import annotations

# This is where the VLM actually runs
import rospy
from pathlib import Path
import torch
from typing import Any, cast
from . import tools
from std_msgs.msg import String
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from threading import Event
import json
from dataclasses import asdict


class VLMInterface():
    def __init__(self,system_prompt_think:str,system_prompt_act:str,model_path:str,toolexe:tools.ToolExecutor | None=None, system_prompt_master:str|None=None,max_tok:int=512) -> None:
        self.system_prompt_act=system_prompt_act
        self.system_prompt_think=system_prompt_think
        self.model_path=Path(model_path)
        self.model:Qwen3VLForConditionalGeneration | None=None
        self.system_prompt_master=system_prompt_master
        self.toolexe=toolexe
        self.processor=None
        self.user_reply=None
        self.reply_event=Event()
        self.active=0
        self.awt=1
        self.idle=1
        self.steps=0
        self.max_tok=max_tok

    def init_vlm(self):
        self._load_model()
        self.output_text_publisher=rospy.Publisher("vlm_text_output",String)
        self.output_image_publisher=rospy.Publisher("vlm_image_output", String)
        self.input_subscriber=rospy.Subscriber("user_input",String)

    def _load_model(self):
        device="cuda" if torch.cuda.is_available() else "cpu"
        dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device=="cuda" else torch.float32

        self.model=Qwen3VLForConditionalGeneration.from_pretrained(self.model_path,dtype=dtype,device_map="auto" if device=="cuda" else None).eval()
        self.processor=AutoProcessor.from_pretrained(self.model_path)

    def step(self):
        pass

    def _predict(self,msg:list[dict],max_tok:int):
        if self.processor is None or self.model is None:
            raise RuntimeError("Processor or model not initialized")
        model=cast(Any, self.model)
        processor=self.processor
        inputs=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids=model.generate(**inputs,max_new_tokens=max_tok,do_sample=False)
        trimmed_ids=[output_ids[len(input_ids):] for input_ids,output_ids in zip(inputs.input_ids,generated_ids)]
        return processor.batch_decode(trimmed_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()

    def run_mission(self,user_prompt:str) -> int:
        self.state=1
        self.steps=0
        self.history=[{"role":"system","content":[{"type":"text","text":self.system_prompt_master}]},
            {"role":"user","content":[{"type":"text","text":user_prompt}],},
        ]

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
            plan=self._predict(think_messages,max_tok=256)

            act_messages=[{"role":"system",
                    "content":[{"type":"text","text":self.system_prompt_act}],
                },
                *self.history,
                {"role":"user",
                "content":[{"type":"text","text":f"Current plan:\n{plan}\n\nReturn the next structured commands."}],
                },
            ]
            cmdjson=self._predict(act_messages,max_tok=self.max_tok)
            if self.toolexe is not None:
                results=[]
                try:
                    commands=self.toolexe.stringtocmd(cmdjson)
                    results=self.toolexe.execute(commands)
                except ValueError as error:
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
                if results[i].command_type=="image" and results[i].status=="Success":
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
        return 0

    @staticmethod
    def _dump(results:list[tools.Return])->str:
        return json.dumps([asdict(result) for result in results], ensure_ascii=False)
