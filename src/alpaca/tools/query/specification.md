# Query Tool
Query tool accesses objects within the world model. Query by either class, object, or none. Query by class and object at the same time is not allowed. Query with no arguments returns a list of all objects within the world model.

### Format:
{
  "commands": [
    {
      "id": "unique-command-id",
      "command_type": "query",
      "arguments": {}
    }
  ]
}

### Examples:
##### Query one object:
{
  "decision": "execute",
  "commands": [
    {
      "id": "query_apple_1",
      "command_type": "query",
      "arguments": {
        "object_id": "apple_1"
      }
    }
  ]
}

##### Query by class:
{
  "decision": "execute",
  "commands": [
    {
      "id": "query_all_chairs",
      "command_type": "query",
      "arguments": {
        "class_name": "chair"
      }
    }
  ]
}

##### Query all objects:
{
  "decision": "execute",
  "commands": [
    {
      "id": "query_all_objects",
      "command_type": "query",
      "arguments": {}
    }
  ]
}

##### Multiple Queries:
{
  "decision": "execute",
  "commands": [
    {
      "id": "query_target_box",
      "command_type": "query",
      "arguments": {
        "object_id": "box_7"
      }
    },
    {
      "id": "query_all_people",
      "command_type": "query",
      "arguments": {
        "class_name": "person"
      }
    }
  ]
}

##### Incorrect Example:
{
  "decision": "execute",
  "commands": [
    {
      "id": "invalid_query",
      "command_type": "query",
      "arguments": {
        "object_id": "chair_2",
        "class_name": "chair"
      }
    }
  ]
}