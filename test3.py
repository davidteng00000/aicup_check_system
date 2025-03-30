import threading
import websockets
import asyncio
import json
import sys
import gradio as gr
from huggingface_hub import InferenceClient
import threading
from flask import Flask,session,render_template



def handle_user_request(prompt):
    client = InferenceClient(model="https://7b-lx-chat.taide.z12.tw")
    partial_message = ''
    for token in client.text_generation(prompt, 
                                        max_new_tokens=500, 
                                        stream=False, 
                                        repetition_penalty=1.1, 
                                        temperature=0.12, 
                                        do_sample=False):  # 沿用您的參數設定
        partial_message += token
    print(partial_message)  # 回傳給使用者

def main():
    while True:
        prompt = input("請輸入您的問題：")
        thread = threading.Thread(target=handle_user_request, args=(prompt,))
        thread.start()

if __name__ == "__main__":
    main()
