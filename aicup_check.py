import gradio as gr
import requests
from typing import Iterator
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.")


# For local streaming, the websockets are hosted without ssl - ws://
HOST = '203.145.216.157:60000'
URI = f'ws://{HOST}/api/v1/stream'


# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'

async def run(context):

    # Note: the selected defaults change from time to time.
    request = {
        'prompt': context,
        'max_new_tokens': 250,# 250,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,
 
        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': False,
        'temperature':0.7,#0.7,
        'top_p': 0.1,#0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,#1.18,
        'repetition_penalty_range': 0,
        'top_k': 40, #40,
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

        while True:
            incoming_data = await websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    yield incoming_data['text']
                case 'stream_end':
                    return
async def print_response_stream(prompt):
    answer=""
    async for response in run(prompt):   
        answer+=response
    return answer

#紀錄該段落是否已經通過檢測的dict，key 為段落；value 為是否通過(1為通過，0為未通過)
Passed = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 0}

#檢查用的function，input_text為要檢核的段落內容；part為第幾段
#關鍵字: 檢核通過  (如果AI回覆為通過，則務必請AI在reply中加入繁體 "檢核成功" 以方便判定)
def check(input_text, part):
    condition = [" ", 
               "內容規定：在此段落中，需要詳細說明你的作業系統、程式語言、使用的套件(函式庫)、採用的預訓練模型，以及任何额外的資料集等相關資訊。如果使用了預訓練模型或额外資料集，務必逐一列出來源。這部分提供了展示你所用工具和資源的空間。",
               "內容規定：此處需要提供你所選用的演算法設計、模型架構和模型參數等相關細節，同時也包括了可能採用的特殊處理方式。這是解釋你如何設計模型、選擇演算法以及設定參數的機會。",
               "內容規定：這一部分要強調你的演算法所帶來的創新性或者對外部資源做出的具體修改。這也包括解釋哪些部分是你所作的新嘗試，或對已有資源所做的變動。",
               "內容規定：這段需要清楚地描述你對資料的處理或擴增方式。這可能包括資料的預處理、清理、特徵工程等方面的處理方式，請提供充分的細節描述。",
               "內容規定：這部分將關注於模型的訓練方式和訓練過程，請詳細說明你採用的訓練方法、優化技術以及模型的訓練歷程。",
               "內容規定：此區域需要對所使用的模型進行深入的分析與討論，包括模型的成效評估、分析結果的解讀、並簡要提出未來可能的改進方向。同時也可透過成功和失敗的案例來進行具體說明。",
               "",
               "內容規定：在這裡需要提供參考文獻清單，需符合APA格式。這是你使用的所有參考資料的列表，請確保格式準確無誤。"
               ]
    success_template = """
    檢核成功 !\n 原因如下: \n\n1. ... \n2. ... ......"""
    fail_template = """
    檢核失敗 !\n 原因如下: \n\n1. ... \n2. ... ......"""
    system_prompt = \
        "等一下將會輸入一些條件與一個要檢核的段落，請你幫我檢測要檢核的段落有沒有符合條件。 \
        輸入的格式為  「輸入開始: 條件 + 要檢核的段落（條件與要檢核的段落中間以“+”分隔）輸入結束」。 \
        請嚴格檢查(保留一點點彈性)要檢核的段落有沒有符合條件。如果有，請馬上回覆\"檢核成功Passed\"並且提供理由；如果沒有通過，請馬上回覆\"檢核失敗\"並且提供理由。\
        如果要檢核的段落太短，馬上回覆\"檢核失敗\"即可 \
        輸入開始:  條件: " + condition[part] + " + "

    
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n 要檢核的段落: {input_text} 輸入結束。一定要馬上回覆\"檢核成功Passed\"或\"檢核失敗\"。[/INST]\n"


    
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='General debug things')
    parser.add_argument('--msg', type=str)
    args = parser.parse_args()

    response=asyncio.run(print_response_stream(prompt))

    #更新通過紀錄Passed
    if "檢核成功Passed" in response:
        Passed[part] = 1
        gr.Info("檢核通過 !")
    else:
        Passed[part] = 0
        gr.Warning("檢核未通過 !")
    # 全部通過
    if checkPassed():
        passed = True
    else:
        passed = False
    
    return str(response)  #回傳AI的回覆

#取得Passed dict裡面的值
def getPassed(part):
    return Passed[part]

#取得總進度
def getProgress():
    progress = str(int((getPassed(1) + getPassed(2) + getPassed(3) + getPassed(4) + getPassed(5) + getPassed(6) + getPassed(8)) / 7 * 100))
    return progress

#更新所有進度
def updateValue():
    
    dict = {lTop(): 1, 
                "壹、環境": getPassed(1), "貳、演算方法與模型架構": getPassed(2), "參、創新性": getPassed(3), 
                "肆、資料處理": getPassed(4), "伍、訓練方式": getPassed(5), "陸、分析與結論": getPassed(6), "捌、使用的外部資源與參考文獻": getPassed(8)}
    return dict

def lTop():
    if(checkPassed()):
        return "[繳交頁面](https://www.aicup.tw/)"
    else:
        return "完成度" + getProgress() + "%"

def submitURL():
    if(passed):
        return "\n[繳交頁面](https://www.aicup.tw/)"
    
    else:
        return ""

passed = False
# 檢查是否全部通過
def checkPassed():
    for i in Passed.values():
        if i != 1:
            return False
    return True

#主頁面
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # 報告檢核系統
    請依序輸入要檢核的段落
    """)
    with gr.Row():
        with gr.Column(scale=1):
            labels = gr.Label(
                    label="完成度",
                    value=updateValue(),
                    container=True,
                    show_label=False,
                    visible=True,
                    
                    )
            labels.change(fn=updateValue,inputs=None,outputs=labels,every=0.1,)
            # if(passed):
            #     gr.Markdown("[繳交頁面](https://www.aicup.tw/)", visible=False)
            # gr.Markdown.change("[繳交頁面](https://www.aicup.tw/)", visible=True)
                
        with gr.Column(scale=3):
            
            with gr.Tab(label="壹、環境"):
                input = gr.Textbox(max_lines=7, lines=7, label="請說明使用的作業系統、語言、套件(函式庫)、預訓練模型、額外資料集等。如使用預訓練模型及額外資料集，請逐一列出來源。(200~600字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=1, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
                
            with gr.Tab(label="貳、演算方法與模型架構"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法設計、模型架構與模型參數，包括可能使用的特殊處理方式。(400~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=2, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="參、創新性"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法之創新性或者修改外部資源的哪一部分。(300~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=3, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="肆、資料處理"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明對資料的處理或擴增的方式，例如對資料可能的刪減、更正或增補。(300~1500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=4, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="伍、訓練方式"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明模型的訓練方法與過程。(400~1000字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=5, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="陸、分析與結論"):
                input = gr.Textbox(max_lines=7, lines=7, label="分析所使用的模型及其成效，簡述未來可能改進的方向。分析必須附圖，可將幾個成功的和失敗的例子附上並說明之。(400~2500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=6, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
                
            with gr.Tab(label="捌、使用的外部資源與參考文獻"):
                input = gr.Textbox(max_lines=7, lines=7, label="參考文獻請以APA格式為主。")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=8, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part], outputs=output)
        
    
        
if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
