import rospy
import message_filters
from std_msgs.msg import String

class UI:
    def __init__(self) -> None:
        pass
    def initui(self):
        self.text_subscriber_vlm=message_filters.Subscriber("vlm_text_output",String)
        self.image_subscriber_vlm=message_filters.Subscriber("vlm_image_output",String)
        self.text_output_publisher=rospy.Publisher("user_input",String)
        self.syncronizer=message_filters.ApproximateTimeSynchronizer([self.text_subscriber_vlm,self.image_subscriber_vlm], queue_size=10, slop=0.05, allow_headerless=True)
        self.syncronizer.registerCallback(self._callback)

    def _callback(self, textmsg:String,imagemsg:String):
        recieveflag=0
        imgflag=0
        img=None
        txt=None
        if textmsg is not None and imagemsg is not None:
            imgflag=1
            img=imagemsg.data
            txt=textmsg.data

        response=input(f"Qwen: \n Text:{txt} \n Image{img}")
        self.text_output_publisher.publish(String(data=response))
    
