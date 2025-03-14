import json


def adjust_json(owebui_json):
    # Preparing the new structure
    new_json = {
        "model": owebui_json["model"],
        "stream": owebui_json["stream"],
        "user": owebui_json["user"],
        "messages": []
    }
    if "options" in owebui_json:
        new_json["options"] = owebui_json["options"]

    # Function to recursively process messages
    def process_messages(messages):
        for message in messages:
            if message["role"] == "user":
                # Handle content as a list of structured items
                if isinstance(message["content"], list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            text_message = {
                                "role": "user",
                                "content": item["text"]
                            }
                            new_json["messages"].append(text_message)
                        elif item["type"] == "image_url":
                            image_message = {
                                "role": "user",
                                "images": [item["image_url"]["url"]]
                            }
                            new_json["messages"].append(image_message)
                else:
                    # If it's a plain string
                    user_message = {
                        "role": "user",
                        "content": message["content"]
                    }
                    new_json["messages"].append(user_message)
            else:
                # Append other roles (system, assistant) as is
                new_json["messages"].append(message)

    # Process top-level messages
    if "messages" in owebui_json and isinstance(owebui_json["messages"], list):
        # If top-level messages are provided, process them
        for top_message in owebui_json["messages"]:
            if isinstance(top_message, dict) and "messages" in top_message:
                process_messages(top_message["messages"])
            else:
                process_messages([top_message])  # Handle messages that are not nested

    return new_json


# Test the function with the provided owebui_json structure
owebui_json = {
    "model": "llava:34b",
    "stream": False,
    "messages": [
        {
            "model": "llava:34b",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "Define this prompt from user is ask prediction or ask knowledge or ask code, answer must only prediction or knowledge or code just it"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG0AAABmCAYAAADBPx+VAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAA3VSURBVHgB7Z27r0zdG8fX743i1bi1ikMoFMQloXRpKFFIqI7LH4BEQ+...=="
                            }
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": "This is a drawing of an animal that resembles a pig with cute facial features..."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG0AAABmCAYAAADBPx+VAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAA3VSURBVHgB7Z27r0zdG8fX743i1bi1ikMoFMQloXRpKFFIqI7LH4BEQ+...=="
                            }
                        }
                    ]
                }
            ]
        }
    ],
    "user": {
        "name": "admin",
        "id": "47e8b931-acaf-41fd-9d0f-22e0ba598950",
        "email": "admin@admin.com",
        "role": "admin"
    },
    "options": {
        "temperature": 0.1,
        "context_length": 8192
    }
}

# owebui_json = {'stream': False, 'model': 'qwen2.5:14b-instruct-q4_K_M', 'messages': [{'role': 'system', 'content': 'Define this prompt from user is ask prediction or ask knowledge or ask code, answer must only "prediction" or "knowledge" or "code" just it'}, {'role': 'user', 'content': 'test halo'}], 'user': {'name': 'admin', 'id': '47e8b931-acaf-41fd-9d0f-22e0ba598950', 'email': 'admin@admin.com', 'role': 'admin'}, 'options': {'temperature': 0.1, 'context_length': 8192}}

# Prepare the new JSON structure
new_json = adjust_json(owebui_json)

# Converting the output to JSON string for display
output_json = json.dumps(new_json, indent=4)
print(output_json)