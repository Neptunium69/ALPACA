import rospy
import message_filters
import json
from std_msgs.msg import String

class UI:
    def __init__(self) -> None:
        pass
    def initui(self):
        self.text_subscriber_vlm=message_filters.Subscriber("vlm_text_output",String)
        self.image_subscriber_vlm=message_filters.Subscriber("vlm_image_output",String)
        self.text_output_publisher=rospy.Publisher("user_input",String,queue_size=1)
        self.completion_subscriber=rospy.Subscriber("vlm_completion_output",String,self._completion_callback,queue_size=1)
        self.step_result_subscriber=rospy.Subscriber("vlm_step_result",String,self._step_result_callback,queue_size=10)
        self.syncronizer=message_filters.ApproximateTimeSynchronizer([self.text_subscriber_vlm,self.image_subscriber_vlm], queue_size=10, slop=0.05, allow_headerless=True)
        self.syncronizer.registerCallback(self._callback)

    def _callback(self, textmsg:String,imagemsg:String):
        img=imagemsg.data
        txt=textmsg.data

        response=input(f"Qwen: \n Text:{txt} \n Image{img}")
        self.text_output_publisher.publish(String(data=response))

    @staticmethod
    def _completion_callback(message:String)->None:
        try:
            payload=json.loads(message.data)
            text=payload.get("text","") if isinstance(payload,dict) else ""
            image=payload.get("image","") if isinstance(payload,dict) else ""
        except json.JSONDecodeError:
            text=message.data
            image=""
        print(f"Qwen mission complete:\n Text:{text}\n Image{image}",flush=True)

    @staticmethod
    def _step_result_callback(message:String)->None:
        try:
            payload=json.loads(message.data)
            step=payload.get("step","?") if isinstance(payload,dict) else "?"
            results=payload.get("results",[]) if isinstance(payload,dict) else []
            print(f"\nQwen step {step} results:\n{json.dumps(results,ensure_ascii=False,indent=2)}",flush=True)
        except json.JSONDecodeError:
            print(f"\nQwen step results:\n{message.data}",flush=True)
    
