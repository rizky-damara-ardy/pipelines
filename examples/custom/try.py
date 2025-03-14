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

    # Extract and flatten user messages
    for message in owebui_json["messages"][0]["messages"]:
        if message["role"] == "user":
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
                        "images": [item["image_url"]["url"].split(",")[1]]
                    }
                    new_json["messages"].append(image_message)
        else:
            new_json["messages"].append(message)

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

# Prepare the new JSON structure
new_json = prepare_owebui_json(owebui_json)

# Converting the output to JSON string for display
output_json = json.dumps(new_json, indent=4)
print(output_json)