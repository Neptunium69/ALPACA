"""Compact remote inference for ALPACA."""

from .localrillm import VLMHost,VLMReciever

# Import the vehicle adapter explicitly; the laptop transport does not need ROS.
__all__=["VLMHost","VLMReciever"]
