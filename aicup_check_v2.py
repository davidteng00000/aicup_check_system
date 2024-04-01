import websockets
import asyncio
import json
import sys
import gradio as gr

# Constants (replace with your actual values)

HOST = 'latest.model.taide.z12.tw'
URI = f'ws://{HOST}/api/v1/stream'

def create_request(context):
    return {
        'prompt': context,
        'max_new_tokens': 250,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
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

async def connect_websocket(uri, request):
    results = ""  # Store the results

    async with websockets.connect(uri) as websocket: 
        await websocket.send(json.dumps(request))

        async for message in websocket:
            incoming_data = json.loads(message)
            evt = incoming_data['event']
            if evt == 'text_stream':
                results += (incoming_data['text'])  # Append to the results
            elif evt == 'stream_end':
                break
            else:
                print('Unknown event:', incoming_data.get('event')) 

    return results  # Return the collected results
                
condition = [" ", 
            "內容規定：請說明使用的作業系統、語言、套件(函式庫)、預訓練模型、額外資料集等。如使用預訓練模型及額外資料集，請逐一列出來源。",
            "內容規定：說明演算法設計、模型架構與模型參數，包括可能使用的特殊處理方式。",
            "內容規定：說明演算法之創新性或者修改外部資源的哪一部分。",
            "內容規定：說明對資料的處理或擴增的方式，例如對資料可能的刪減、更正或增補。",
            "內容規定：說明模型的訓練方法與過程。",
            "內容規定：分析所使用的模型及其成效，簡述未來可能改進的方向，可將幾個成功的和失敗的例子附上並說明之。",
            "",
            "內容規定：參考文獻請以APA格式為主。"
            ]

async def run(input_text: str, part: int):    # 原名main
    system_prompt = """你將收到內容規定和一個段落，請檢查該段落是否符合規定。請根據以下格式回答：如果段落符合規定，請回答：「檢查通過！原因如下：」並以列表的形式列出支持的理由。若段落不符合規定，請回答：「檢查未通過！原因如下：」並以列表的形式列出支持的理由。請盡量簡化和明確你的回答。
    """
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{condition[part]} + {input_text}。[/INST]"
    request = create_request(prompt)
    result = await connect_websocket(URI, request)
    # print('Result', result)
    return result

def check(input_text: str, part: int, Passed: list[int]):
    response = asyncio.run(run(input_text, part))
    #更新通過紀錄Passed
    if "檢查通過" in response:
        Passed[part] = 1
        gr.Info("檢核通過 !")
        Pass = True
        for i in Passed:
            if i == 0:
                Pass = False
                break
    else:
        Passed[part] = 0
        gr.Warning("檢核未通過 !")
    return str(response)

def getPassed(Passed: list[int]):
    PassList = [1,1,1,1,1,1,1,1,1]
    for i in range(10):
        if Passed[i] == 0:
            PassList[i] = 0
    return PassList

def update_label(Passed: list[int]):
    # PassList = getPassed(Passed)
    PassList = Passed
    progress = int((PassList[1] + PassList[2] + PassList[3] + PassList[4] + PassList[5] + PassList[6] + PassList[8])/7*100)
    dict = {
        "完成度" + str(progress) + "%": 1,
        "壹、環境": PassList[1],
        "貳、演算方法與模型架構": PassList[2],
        "參、創新性": PassList[3],
        "肆、資料處理": PassList[4], 
        "伍、訓練方式": PassList[5],
        "陸、分析與結論": PassList[6],
        "捌、使用的外部資源與參考文獻": PassList[8],
    }
    return dict
    

#主頁面
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # 報告檢核系統
    請依序輸入要檢核的段落
    """)
    Passed = gr.State(value = [1,0,0,0,0,0,0,1,0])
    with gr.Row():
        with gr.Column(scale=1):
            labels = gr.Label(
                    label="完成度",
                    value={
        "完成度0%": 1,
        "壹、環境": 0,
        "貳、演算方法與模型架構": 0,
        "參、創新性": 0,
        "肆、資料處理": 0, 
        "伍、訓練方式": 0,
        "陸、分析與結論": 0,
        "捌、使用的外部資源與參考文獻": 0,
                    },
                    container=True,
                    show_label=False,
                    visible=True,
                    
                    )
            labels.change(fn=update_label,inputs=Passed,outputs=labels,every=0.1,)
                
        with gr.Column(scale=3):
            
            with gr.Tab(label="壹、環境"):
                input = gr.Textbox(max_lines=7, lines=7, label="請說明使用的作業系統、語言、套件(函式庫)、預訓練模型、額外資料集等。如使用預訓練模型及額外資料集，請逐一列出來源。(200~600字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=1, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part,Passed], outputs=output)
                
                
            with gr.Tab(label="貳、演算方法與模型架構"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法設計、模型架構與模型參數，包括可能使用的特殊處理方式。(400~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=2, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="參、創新性"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法之創新性或者修改外部資源的哪一部分。(300~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=3, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="肆、資料處理"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明對資料的處理或擴增的方式，例如對資料可能的刪減、更正或增補。(300~1500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=4, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="伍、訓練方式"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明模型的訓練方法與過程。(400~1000字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=5, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="陸、分析與結論"):
                input = gr.Textbox(max_lines=7, lines=7, label="分析所使用的模型及其成效，簡述未來可能改進的方向。分析必須附圖，可將幾個成功的和失敗的例子附上並說明之。(400~2500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=6, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="捌、使用的外部資源與參考文獻"):
                input = gr.Textbox(max_lines=7, lines=7, label="參考文獻請以APA格式為主。")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=8, visible = False)
                output = gr.Textbox(label="段落檢核結果")
                check_btn.click(fn=check, inputs=[input,part,Passed], outputs=output)
        
    
        
if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=False)