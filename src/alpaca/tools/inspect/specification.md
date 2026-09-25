# Inspect Tool
Inspect tool examines a known object from a safe observation position. It finds candidate inspection positions around the target, attempts to navigate to them in order, and scans the environment after reaching the first successful position.

The `target` argument is the unique object id stored in the world model. The target must resolve to exactly one object. Inspection may physically move the vehicle and must not be used with an object class name or coordinates in place of the object id.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "inspect",
      "arguments": {
        "target": "object-id"
      }
    }
  ]
}

### Examples:
##### Inspect one known object:
{
  "decision": "execute",
  "commands": [
    {
      "id": "inspect_generator_1",
      "command_type": "inspect",
      "arguments": {
        "target": "generator_1"
      }
    }
  ]
}

##### Incorrect example—missing target:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_inspect_1",
      "command_type": "inspect",
      "arguments": {}
    }
  ]
}

##### Incorrect example—coordinates instead of an object id:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_inspect_2",
      "command_type": "inspect",
      "arguments": {
        "target": {
          "x": 1.0,
          "y": 2.0,
          "z": 0.0
        }
      }
    }
  ]
}
