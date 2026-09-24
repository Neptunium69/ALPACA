# Navigate Tool

Navigate moves the vehicle to a destination pose after checking vehicle readiness and route safety. The destination is a position with `x`, `y`, and `z` coordinates. The orientation uses Euler angles in radians and must contain `pitch`, `yaw`, and `roll`. Every coordinate and angle must be a finite number.

`frame_id` is optional and defaults to the vehicle adapter's navigation frame. If supplied, it must be a non-empty string and must agree with the adapter, localization, and voxel-map frames used by the safety system.

Navigation can fail when the vehicle is not ready, the target footprint is occupied, no path is available, or the route needs more information. When the risk model supplies a safe vantage point, the tool may navigate there, gather the newly available route information, and retry the original destination. A position returned by Query describes an object's occupied geometry and should not be used directly as a safe navigation waypoint; use Inspect when the goal is to approach a known object.

### Format:

```json
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "navigate",
      "arguments": {
        "destination": {"x": 1.5, "y": -0.4, "z": 0.0},
        "orientation": {"pitch": 0.0, "yaw": 1.57, "roll": 0.0},
        "frame_id": "map"
      }
    }
  ]
}
```

### Example using the adapter's default frame:

```json
{
  "decision": "execute",
  "commands": [
    {
      "id": "navigate_to_clear_area_1",
      "command_type": "navigate",
      "arguments": {
        "destination": {"x": 2.0, "y": 0.75, "z": 0.0},
        "orientation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
      }
    }
  ]
}
```

### Incorrect example—missing orientation:

```json
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_navigate_1",
      "command_type": "navigate",
      "arguments": {
        "destination": {"x": 2.0, "y": 0.75, "z": 0.0}
      }
    }
  ]
}
```
