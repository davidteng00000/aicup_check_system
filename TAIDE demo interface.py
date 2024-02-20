from threading import Thread
from typing import Iterator
import gradio as gr
import os
import asyncio
import json
import sys


try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")


DESCRIPTION = """\
# TAIDE Demo site
"""

LICENSE = """
Not publish yet
"""

HOST = 'latest.model.taide.z12.tw'
URI = f'ws://{HOST}/api/v1/stream'
INFERENCE_ADDRESS = URI

MAX_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 250
MAX_INPUT_TOKEN_LENGTH = 4096

conversations = []

async def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 250,
    temperature: float = 0.7,
    top_p: float = 0.1,
    top_k: int = 40,
    repetition_penalty: float = 1.18,
) -> Iterator[str]:
    global conversations
    URI = INFERENCE_ADDRESS
    # Note: the selected defaults change from time to time.
    # Note: before we publish the service, we need to limit the parameters that will been accepts.
    if len(chat_history) == 0:
        conversations = []

    conversations.append({
        'role': 'user',
        'content': message,
    })
    
    # prompt = [f'<s>[INST] {c["content"]} [/INST]' if c['role'] == 'user' else c['content'] for c in conversations]
    prompt = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n {c['content']} [/INST]\n" if system_prompt and i == 0 else f'<s>[INST] {c["content"]} [/INST]' if c['role'] == 'user' else c['content'] for i, c in enumerate(conversations)]
    context = '\n'.join(prompt)
    request = {
        'prompt': context,
        'max_new_tokens': max_new_tokens,# 250,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': False,
        'temperature': temperature,#0.7,
        'top_p': top_p,#0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': repetition_penalty,#1.18,
        'repetition_penalty_range': 0,
        'top_k': top_k, #40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'grammar_string': '',
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'custom_token_bans': '',
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    async with websockets.connect(URI, ping_interval=None) as websocket:
        await websocket.send(json.dumps(request))

        #yield context  # Remove this if you just want to see the reply
        outputs = []
        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    # print(outputs)
                    outputs.append(incoming_data['text'])
                    yield "".join(outputs)
                case 'stream_end':
                    conversations.append({
                        'role': 'assistant',             'content': "".join(outputs),         })
                    print(conversations)
                    return

chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS,),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.7,),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.1,),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=40,),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.18,),
    ],
    stop_btn=None,
    examples=[
      ["請介紹一下台灣的風景"],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()