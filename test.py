import requests

API_URL = "https://7b-lx-chat.taide.z12.tw/generate"

prompt = "我在大安森林公園騎著腳踏車，看到宋伯伯、王伯伯正在下棋。"
repetition_penalty = 1.3
grammar = {
    "type": "json",
    "value": {
        "properties": {
            "地點": {"type": "string"},
            "我的行為": {"type": "string"},
            "多少人": {"type": "integer", "minimum": 1, "maximum": 5},
            "他們正在做什麼": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["地點", "我的行為", "多少人", "他們正在做什麼"],
    },
}

payload = {
    "inputs": prompt,
    "parameters": {
        "repetition_penalty": repetition_penalty,
        "grammar": grammar,
    },
}

headers = {"Content-Type": "application/json"}

response = requests.post(API_URL, json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print(f"Error: {response.status_code} - {response.text}")