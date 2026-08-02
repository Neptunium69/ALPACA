# This is where the VLM actually runs
import rospy
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration



class VLMInterface():
    def __init__(self,system_prompt_think:str,system_prompt_act:str,model_path:str) -> None:
        self.system_prompt_act=system_prompt_act
        self.system_prompt_think=system_prompt_think
        self.model_path=Path(model_path)
        self.model=None
        self.state=0

    def init_vlm(self):
        rospy.init_node("vlm_interface_node",anonymous=True)
        self._load_model()

    def _load_model(self):
        device="cuda" if torch.cuda.is_available() else "cpu"
        dtype=torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device=="cuda" else torch.float32

        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path,torch_dtype=dtype,device_map="auto" if device=="cuda" else "cpu").eval()
        self.processor=AutoProcessor.from_pretrained(self.model_path,min_pixels=256*28*28,max_pixels=1280*28*28)
        

    def _predict(self):
        pass

    def _callback(self, msg):
        pass
