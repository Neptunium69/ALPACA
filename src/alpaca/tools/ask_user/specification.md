# Ask User Tool
Ask User tool publishes a question and an optional image path to the user, then waits for a reply on the `user_input` topic. This is a blocking action. On success, it returns the reply in `User Response`.

The `text_message` argument is required. The optional `image_message` argument is the exact path of an existing image returned by Observe or Image. When no image is supplied, the tool publishes an empty image message alongside the text.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "ask_user",
      "arguments": {
        "text_message": "Question for the user",
        "image_message": "/path/returned/by/image/tool.png"
      }
    }
  ]
}

### Examples:
##### Ask a question without an image:
{
  "decision": "execute",
  "commands": [
    {
      "id": "ask_target_confirmation",
      "command_type": "ask_user",
      "arguments": {
        "text_message": "I found two boards. Which one should I inspect?"
      }
    }
  ]
}

##### Incorrect example: missing required text:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_ask_user_1",
      "command_type": "ask_user",
      "arguments": {}
    }
  ]
}
