import websockets
import asyncio
import json
import sys
import gradio as gr
from huggingface_hub import InferenceClient
import threading
from queue import Queue
from flask import Flask,session,render_template
from APIs.Openai.sync_api import *

client = InferenceClient(model="https://7b-lx-chat.taide.z12.tw")

def handleInput(input_text: str, part: int, Passed: list[int]):
    result_queue = Queue()
    thread = threading.Thread(target=inference, args=(input_text, part, Passed, result_queue))
    thread.start()
    partial_message = ""
    while True:  # 持續從 Queue 中取值
        if '<END>' in result_queue.get():
            break
        else:
            # result_queue.get() != "</s>":  # 自定義結束標記
            partial_message += result_queue.get()
        yield partial_message  # 將值傳回給呼叫者
    

def inference(input_text: str, part: int, Passed: list[int], result_queue: Queue):
    if len(input_text) < minLength[part]:
        partial_message = "字數不足，檢核未通過 !"
        yield partial_message
        # result_queue.put(partial_message)
        check(partial_message, part, Passed)
        return
    
    if part == 6:
        system_prompt = """
        將收到內容規定和報告段落，請替參賽者檢查該段落是否符合規定，請嚴格檢查是否相關。
        請根據以下格式回答：如果段落符合規定，回覆開頭為「檢查通過！原因如下：」，並以列點的形式列出支持的理由，請勿覆述段落內容。
        如果段落不符合規定，回覆開頭為「檢查未通過！原因如下：」，並列出支持的理由，請勿覆述段落內容。
        回答時請勿複述段落內容。請盡量簡化和明確你的回答。
        """
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt} {condition[part]}\n<</SYS>>\n\n 段落: {input_text}。\n請嚴格檢查段落是否符合以下條件: {condition[part]}。\n回覆開頭必須為「檢查通過！原因如下：」或是「檢查未通過！原因如下：」或是「請提供正確的報告段落!檢查未通過！原因如下：」，隨後列出理由。[/INST]"

    elif part == 3:
        system_prompt = """
        將收到一個AI比賽報告裡面的其中一個段落，請替參賽者檢查該段落是否符合以下規定: 提到創新性，不必嚴格檢查，不需要細節。
        請根據以下格式回答：如果段落符合規定，回覆開頭為「檢查通過！原因如下：」，並以列點的形式列出支持的理由。
        如果段落不符合規定，回覆開頭為「檢查未通過！原因如下：」，並列出支持的理由。
        如果段落內容與主題明顯無關，則回覆開頭為「請提供正確的報告段落！ 檢查未通過！原因如下：」，並列出支持的理由，例如段落缺少了甚麼。
        回答時請勿複述段落內容。請盡量簡化和明確你的回答。
        """
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt} \n<</SYS>>\n\n 段落: {input_text}。\n回覆開頭必須為「檢查通過！原因如下：」或是「檢查未通過！原因如下：」或是「請提供正確的報告段落!檢查未通過！原因如下：」，隨後列出理由。[/INST]"
    
    else:
        system_prompt = """
        將收到內容規定和報告段落，請替參賽者檢查該段落是否符合規定，請嚴格檢查是否相關。
        請根據以下格式回答：如果段落符合規定，回覆開頭為「檢查通過！原因如下：」，並以列點的形式列出支持的理由，請勿覆述段落內容。
        如果段落不符合規定，回覆開頭為「檢查未通過！原因如下：」，並列出支持的理由，請勿覆述段落內容。
        如果段落內容與規定明顯無關，則回覆開頭為「請提供正確的報告段落！ 檢查未通過！原因如下：」，並列出支持的理由，例如段落缺少了甚麼。
        回答時請勿複述段落內容。請盡量簡化和明確你的回答。
        """
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt} {condition[part]}\n<</SYS>>\n\n 段落: {input_text}。\n請嚴格檢查段落是否符合以下條件: {condition[part]}。\n回覆開頭必須為「檢查通過！原因如下：」或是「檢查未通過！原因如下：」或是「請提供正確的報告段落!檢查未通過！原因如下：」，隨後列出理由。[/INST]"

    # system_prompt = """
    # 請替報告段落做出內容相關性檢查。
    # 你將收到該段落內容規定與段落內文，請檢查段落內文是否與段落內容規定相關，不須嚴格檢查。
    # 回覆開頭必須為「檢查通過！原因如下：」或「檢查未通過！原因如下：」或「請提供正確的報告段落!」，並且列出你判斷的依據。
    # 在回覆中，禁止提到"AI CUP 表現優異的參賽者"。
    # """
    # prompt = f"<s>[INST] <<SYS>>\n{system_prompt} \n{condition[part]} <<SYS>> \n\n段落內文: {input_text} 回覆開頭必須為「檢查通過！原因如下：」或「檢查未通過！原因如下：」或是「請提供正確的報告段落!」，並且列出你判斷的依據。[INST]"
    
    
    partial_message = ''
    # partial_message += prompt
    for token in client.text_generation(prompt, 
                                        max_new_tokens=500, 
                                        stream=True, 
                                        repetition_penalty=1.1, 
                                        temperature=0.12, 
                                        do_sample=False
                                        ):
        partial_message += token
        if(token != '</s>'):
            yield partial_message
            # result_queue.put(token)
    # result_queue.put("<END>")
    check(partial_message, part, Passed)
    # return partial_message
                
condition = ["",
            "內容規定：說明使用的作業系統、語言、套件(函式庫)、預訓練模型、額外資料集等。",
            "內容規定：必須說明演算法設計、模型架構與模型參數，包括可能使用的特殊處理方式。",
            "內容規定：說明做法不同之處即可，不必嚴格檢查，不需要細節。",
            "內容規定：必須說明對資料的處理或擴增的方式，例如對資料可能的刪減、更正或增補。",
            "內容規定：必須說明模型的訓練方法與過程，若無關則檢查不通過。",
            "內容規定：說明分析並總結。",
            "",
            "內容規定：以APA/IEEE格式提供參考文獻，或是直接註明沒有參考文獻，不需要理由"
            ]

minLength = [0,100,250,200,200,250,250, 0, 0]

def check(response, part: int, Passed: list[int]):
    #更新通過紀錄Passed
    if "檢查通過" in response:
        Passed[part] = 1
        gr.Info("檢核通過 !")
        
    else:
        Passed[part] = 0
        gr.Warning("檢核未通過 !")


def update_label(Passed: list[int]):
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
    # if 0 in Passed:
    
    
    return dict
    

#主頁面
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # AI CUP 賽後報告檢核系統
    請依序輸入要檢核的段落\n
    本檢核系統基於國科會TAIDE模型開發，目前為demo版，僅為服務參賽者自我檢核，並輔助完成報告。實際敘獎、評分仍100%依競賽評審委員判定為準，系統所提供的完成度、評語或建議，均不涉及評審結果。如有任何改善建議，歡迎您來信moe.ai.ncu@gmail.com。您的寶貴意見將使台灣本土大型語言模型應用愈加蓬勃：）
    """)
    Passed = gr.State(value = [1,0,0,0,0,0,0,1,0])
    # finish = gr.Markdown(
    #     "您已完成自我檢核，請將完整報告書及程式碼寄送至主辦單位信箱! ",
    #     visible=False
    # )
    # finish.change(fin, Passed, finish, every=0.5)
    
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
                    # visible=True,
                    
                    )
            labels.change(fn=update_label,inputs=Passed,outputs=labels,every=0.1,)

                
        with gr.Column(scale=3):
            
            with gr.Tab(label="壹、環境"):
                input = gr.Textbox(max_lines=7, lines=7, label="請說明使用的作業系統、語言、套件(函式庫)、預訓練模型、額外資料集等。如使用預訓練模型及額外資料集，請逐一列出來源。(200~600字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=1, visible = False)
                output = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                check_btn.click(fn=inference, inputs=[input,part,Passed], outputs=output)
                
                
            with gr.Tab(label="貳、演算方法與模型架構"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法設計、模型架構與模型參數，包括可能使用的特殊處理方式。(400~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=2, visible = False)
                output = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                check_btn.click(fn=inference, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="參、創新性"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明演算法之創新性或者修改外部資源的哪一部分。(300~1200字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=3, visible = False)
                output = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                check_btn.click(fn=inference, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="肆、資料處理"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明對資料的處理或擴增的方式，例如對資料可能的刪減、更正或增補。(300~1500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=4, visible = False)
                output = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                check_btn.click(fn=inference, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="伍、訓練方式"):
                input = gr.Textbox(max_lines=7, lines=7, label="說明模型的訓練方法與過程。(400~1000字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=5, visible = False)
                output = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                check_btn.click(fn=inference, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="陸、分析與結論"):
                input = gr.Textbox(max_lines=7, lines=7, label="分析所使用的模型及其成效，簡述未來可能改進的方向。可將幾個成功的和失敗的例子附上並說明之。(400~2500字)")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=6, visible = False)
                output = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                check_btn.click(fn=inference, inputs=[input,part,Passed], outputs=output)
                
            with gr.Tab(label="捌、使用的外部資源與參考文獻"):
                input = gr.Textbox(max_lines=7, lines=7, label="參考文獻請以APA或IEEE格式為主。")
                check_btn = gr.Button("Check")
                part = gr.Slider(1, 8, value=8, visible = False)
                output = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                check_btn.click(fn=inference, inputs=[input,part,Passed], outputs=output)
        
    
        
if __name__ == "__main__":
    demo.queue(max_size=10).launch(share=False)