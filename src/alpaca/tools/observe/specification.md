# Observe Tool
Observe tool captures a recent image from the vehicle's color camera and saves it as a PNG file. This command has no arguments. Use it when a new camera image is needed for visual inspection or planning.

On success, the tool returns the saved image path, image width, image height, camera timestamp in nanoseconds, and camera frame id. The tool fails if it cannot receive, convert, or save a camera image.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "observe",
      "arguments": {}
    }
  ]
}

### Examples:
##### Capture one camera image:
{
  "decision": "execute",
  "commands": [
    {
      "id": "observe_camera_1",
      "command_type": "observe",
      "arguments": {}
    }
  ]
}

##### Capture images for two separate observations:
{
  "decision": "execute",
  "commands": [
    {
      "id": "observe_work_area",
      "command_type": "observe",
      "arguments": {}
    },
    {
      "id": "observe_entry_point",
      "command_type": "observe",
      "arguments": {}
    }
  ]
}

##### Incorrect example—unexpected argument:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_observe_1",
      "command_type": "observe",
      "arguments": {
        "camera": "front"
      }
    }
  ]
}
