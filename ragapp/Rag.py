

import time
import requests

def post_chat_request(user_input, chat_history):
    url = "https://2a3b-129-80-222-238.ngrok-free.app/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "user_input": user_input,
        "chat_history": chat_history
    }
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()  # or response.text if you expect a plain text response
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

# Example usage:
# result = post_chat_request("who is Sarvesh", [])
# print(result['response'])


def resp(input):
    # return "ok"
    result = post_chat_request(input, [])
    print(result)
    return result['response']