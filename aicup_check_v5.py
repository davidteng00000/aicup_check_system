import gradio as gr
from queue import Queue
from APIs.Openai.sync_api import *

# 初始化 OpenAI agent
openai_agent_4o = OpenAI_agent(
    model_name="gpt-4o",
    temperature=0.3,
)

condition = [
    "",
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


def make_prompt(input_text: str, part: int) -> str:
    base_prompt = f"段落: {input_text}。\n請嚴格檢查段落是否符合以下條件: {condition[part]}。\n回覆開頭必須為「檢查通過！原因如下：」或是「檢查未通過！原因如下：」或是「請提供正確的報告段落!檢查未通過！原因如下：」，隨後列出理由。"

    if part == 3:
        sys_prompt = """
        將收到一個AI比賽報告裡面的其中一個段落，請替參賽者檢查該段落是否符合以下規定: 提到創新性，不必嚴格檢查，不需要細節。
        請根據以下格式回答：如果段落符合規定，回覆開頭為「檢查通過！原因如下：」，並以列點的形式列出支持的理由。
        如果段落不符合規定，回覆開頭為「檢查未通過！原因如下：」，並列出支持的理由。
        如果段落內容與主題明顯無關，則回覆開頭為「請提供正確的報告段落！ 檢查未通過！原因如下：」，並列出支持的理由。
        回答時請勿複述段落內容。請盡量簡化和明確你的回答。
        """
    else:
        sys_prompt = """
        將收到內容規定和報告段落，請替參賽者檢查該段落是否符合規定，請嚴格檢查是否相關。
        請根據以下格式回答：如果段落符合規定，回覆開頭為「檢查通過！原因如下：」，並以列點的形式列出支持的理由，請勿覆述段落內容。
        如果段落不符合規定，回覆開頭為「檢查未通過！原因如下：」，並列出支持的理由，請勿覆述段落內容。
        如果段落內容與規定明顯無關，則回覆開頭為「請提供正確的報告段落！ 檢查未通過！原因如下：」，並列出支持的理由。
        回答時請勿複述段落內容。請盡量簡化和明確你的回答。
        """

    return f"<|system|>\n{sys_prompt}\n<|user|>\n{base_prompt}"


def inference_openai(input_text: str, part: int, Passed: list[int]):
    if len(input_text.strip()) < minLength[part]:
        msg = "字數不足，檢核未通過 !"
        Passed[part] = 0
        return msg

    prompt = make_prompt(input_text, part)
    messages, response = openai_agent_4o.update_messages(prompt)
    if "檢查通過" in response:
        Passed[part] = 1
        gr.Info("檢核通過 !")
    else:
        Passed[part] = 0
        gr.Warning("檢核未通過 !")
    return response


def update_label(Passed: list[int]):
    progress = int((Passed[1] + Passed[2] + Passed[3] + Passed[4] + Passed[5] + Passed[6] + Passed[8]) / 7 * 100)
    return {
        f"完成度{progress}%": 1,
        "壹、環境": Passed[1],
        "貳、演算方法與模型架構": Passed[2],
        "參、創新性": Passed[3],
        "肆、資料處理": Passed[4],
        "伍、訓練方式": Passed[5],
        "陸、分析與結論": Passed[6],
        "捌、使用的外部資源與參考文獻": Passed[8],
    }


with gr.Blocks() as demo:
    gr.Markdown("""
    # AI CUP 賽後報告檢核系統（OpenAI Agent 版本）
    """)
    Passed = gr.State(value=[1,0,0,0,0,0,0,1,0])

    with gr.Row():
        with gr.Column(scale=1):
            labels = gr.Label(
                label="完成度",
                value=update_label(Passed.value),
                show_label=False,
            )
            labels.change(fn=update_label, inputs=Passed, outputs=labels, every=0.1)

        with gr.Column(scale=3):
            for i, title in zip([1,2,3,4,5,6,8], [
                "壹、環境", "貳、演算方法與模型架構", "參、創新性", "肆、資料處理", 
                "伍、訓練方式", "陸、分析與結論", "捌、使用的外部資源與參考文獻"]):

                with gr.Tab(label=title):
                    input_box = gr.Textbox(max_lines=7, lines=7, label="請輸入段落內容")
                    check_btn = gr.Button("Check")
                    output_box = gr.Textbox(label="段落檢核結果", lines=15, max_lines=15)
                    check_btn.click(fn=inference_openai, inputs=[input_box, gr.State(i), Passed], outputs=output_box)

if __name__ == "__main__":
    demo.queue().launch(share=False)
