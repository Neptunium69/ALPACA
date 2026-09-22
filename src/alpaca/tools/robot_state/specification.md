# Robot Tool
Robot State tool returns infomration related to the robot's current state. This command has no arguments.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "robot_state",
      "arguments": {}
    }
  ]
}

### Examples:
##### Correct Example:
{
  "commands": [
    {
      "id": "robot_state_1",
      "command_type": "robot_state",
      "arguments": {}
    }
  ]
}
### Incorrect Example:
{
  "commands": [
    {
      "id": "robot_state_1",
      "command_type": "robot_state",
      "arguments": {
        "query": "battery"
      }
    }
  ]
}