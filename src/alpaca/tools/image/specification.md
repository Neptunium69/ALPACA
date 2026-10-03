# Image Tool
Image tool captures a camera image directed toward a known object. It uses the vehicle adapter's existing target-facing behavior when needed, checks risk before that movement, and then uses the Observe tool to capture and save the image.

The `target` argument is the unique object id stored in the world model, not a class name or coordinates. This tool may physically move the vehicle. Capturing an image does not independently verify that the target is visible in it.

On success, the tool returns the Observe tool's image path, image width, image height, camera timestamp in nanoseconds, and camera frame id. Target lookup, risk rejection, movement errors, and capture failures are reported through the existing image-tool behavior.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "image",
      "arguments": {
        "target": "object-id"
      }
    }
  ]
}

### Examples:
##### Capture an image of one known object:
{
  "decision": "execute",
  "commands": [
    {
      "id": "image_apple_1",
      "command_type": "image",
      "arguments": {
        "target": "apple_1"
      }
    }
  ]
}

##### Incorrect example: missing target:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_image_1",
      "command_type": "image",
      "arguments": {}
    }
  ]
}
