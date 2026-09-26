# Finish Tool
Finish tool publishes the mission completion message and returns `Mission Completed: true`. Use it only after the mission requirements have been verified. It does not wait for a user reply or independently verify mission completion.

Provide at least one of `completion_text_message` or `completion_image_message`. The image argument must use the exact path of an existing image returned by Observe or Image.

When a completion publisher is supplied, the tool publishes one JSON message with `text` and `image` fields. Otherwise, it publishes separate text and image messages using the original fallback behavior. Both output publishers must be initialized.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "finish",
      "arguments": {
        "completion_text_message": "Mission completion message",
        "completion_image_message": "/path/returned/by/image/tool.png"
      }
    }
  ]
}

### Examples:
##### Finish with a text message:
{
  "decision": "execute",
  "commands": [
    {
      "id": "finish_mission",
      "command_type": "finish",
      "arguments": {
        "completion_text_message": "The requested inspection is complete."
      }
    }
  ]
}

##### Incorrect example: no completion message:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_finish_1",
      "command_type": "finish",
      "arguments": {}
    }
  ]
}
