
import json
import textwrap
import requests
import time


def deepseekv3(question, secret_key):
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {secret_key}",
        "Content-Type": "application/json",
        'Connection': 'close'
    }

    proxies = None


    sys_prompts = 'You are a helpful watermark discriminator.'

    user_prompts = textwrap.dedent(question)

    data = {
        "model": "deepseek-v3",
        "messages": [
            {
                "role": "system",
                "content": sys_prompts,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{user_prompts}"}
                ]
            }
        ]
    }

    while True:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), proxies=proxies)
            result = response.json()
            text = result['choices'][0]['message']['content']
            break

        except Exception as e:
            time.sleep(0.2)
            print(f"With error: {e}，try again...")

    return text
