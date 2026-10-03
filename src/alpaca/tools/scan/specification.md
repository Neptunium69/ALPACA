# Scan Tool
Scan tool surveys the environment around the vehicle and records major changes detected by the world model. The vehicle may rotate or otherwise move its sensors while scanning. This is a physical action and must pass the safety check before it is executed.

This command has no arguments. On success, the tool reports that the scan completed and returns the major world-model changes observed during the scan. The command may fail if the safety check rejects it, the vehicle does not support scanning, or the scan is interrupted.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "scan",
      "arguments": {}
    }
  ]
}

### Examples:
##### Scan the surrounding environment:
{
  "decision": "execute",
  "commands": [
    {
      "id": "scan_surroundings_1",
      "command_type": "scan",
      "arguments": {}
    }
  ]
}

##### Incorrect example—unexpected argument:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_scan_1",
      "command_type": "scan",
      "arguments": {
        "yaw_margin": 0.785398
      }
    }
  ]
}
