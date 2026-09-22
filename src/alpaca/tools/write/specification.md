# Write Tool
Write tool adds an object observed in the latest camera image to the world model. The object must contain a unique id, a class name, and a bounding box. The bounding box is converted into a world position using the latest synchronized depth image.

The `bounding_box` must contain four pixel coordinates in the order `[x_min, y_min, x_max, y_max]`. The maximum coordinates must be greater than the minimum coordinates. `attributes` is optional and must be an object with string keys. The attribute names `Notes` and `VLMFlag` are reserved and must not be provided.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "write",
      "arguments": {
        "object": {
          "id": "unique-object-id",
          "class_name": "object-class",
          "bounding_box": [x_min, y_min, x_max, y_max],
          "attributes": {}
        }
      }
    }
  ]
}

### Examples:
##### Write one object:
{
  "decision": "execute",
  "commands": [
    {
      "id": "write_charging_station_1",
      "command_type": "write",
      "arguments": {
        "object": {
          "id": "charging_station_1",
          "class_name": "charging_station",
          "bounding_box": [214.0, 126.0, 438.0, 392.0]
        }
      }
    }
  ]
}

##### Write one object with attributes:
{
  "decision": "execute",
  "commands": [
    {
      "id": "write_resource_pile_1",
      "command_type": "write",
      "arguments": {
        "object": {
          "id": "resource_pile_1",
          "class_name": "resource_pile",
          "bounding_box": [91.0, 205.0, 307.0, 461.0],
          "attributes": {
            "material": "wood",
            "condition": "dry",
            "estimated_quantity": 12
          }
        }
      }
    }
  ]
}

##### Multiple writes:
{
  "decision": "execute",
  "commands": [
    {
      "id": "write_entry_point_1",
      "command_type": "write",
      "arguments": {
        "object": {
          "id": "entry_point_1",
          "class_name": "entry_point",
          "bounding_box": [52.0, 73.0, 188.0, 414.0]
        }
      }
    },
    {
      "id": "write_work_area_1",
      "command_type": "write",
      "arguments": {
        "object": {
          "id": "work_area_1",
          "class_name": "work_area",
          "bounding_box": [246.0, 181.0, 603.0, 472.0],
          "attributes": {
            "accessible": true
          }
        }
      }
    }
  ]
}

##### Incorrect example—missing bounding box:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_write_1",
      "command_type": "write",
      "arguments": {
        "object": {
          "id": "unknown_structure_1",
          "class_name": "unknown_structure"
        }
      }
    }
  ]
}

##### Incorrect example—reserved attribute:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_write_2",
      "command_type": "write",
      "arguments": {
        "object": {
          "id": "storage_area_1",
          "class_name": "storage_area",
          "bounding_box": [125.0, 94.0, 522.0, 407.0],
          "attributes": {
            "Notes": "Possible storage location"
          }
        }
      }
    }
  ]
}
