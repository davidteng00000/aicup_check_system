import websockets
import asyncio
import json
import sys

# Constants (replace with your actual values)
#HOST = '203.145.216.157'
#PORT = 56458

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
                

async def main():

    prompt = """<s>[INST] <<SYS>>等一下將會輸入一些條件與一個要檢核的段落，請你幫我檢測要檢核的段落有沒有符合條件。
        輸入的格式為  「輸入開始: 條件 + 要檢核的段落（條件與要檢核的段落中間以“+”分隔）輸入結束」。
        請嚴格檢查(保留一點彈性)要檢核的段落有沒有符合條件。如果有，請「馬上、立刻、立即」回覆"檢核成功Passed"並且提供理由；如果沒有通過，請馬上回覆"檢核失敗"並且提供理由。
        輸入開始:  條件: 內容規定：此處需要提供你所選用的演算法設計、模型架構和模型參數等相關細節，同時也包括了可能採用的特殊處理方式。這是解釋你如何設計模型、選擇演算法以及設定參數的機會。 + <</SYS>>
要檢核的段落: 貳、演算方法與模型架構
為了更好的訓練效果,有在損失函數中引入權重,因此對於每個欲預測之
欄位,必須訓練各自專屬的模型,但模型架構大致上相同。
關於輸入資料部分,每份輸入資料長 L 幀,包含各幀資料或片段屬性資
料,以下為可能包含之子資料:
A. 各幀資料:選手姿勢圖片
為兩選手姿勢之黑白照片,故 shape = (BS, L, 2, 64, 64)
B. 各幀資料:選手姿勢數值
包含兩選手 133 個關鍵點的 X 與 Y,故 shape = (BS, L, 4, 133)
C. 各幀資料:羽球位置
包含 X 與 Y,故 shape = (BS, L, 2)
D. 各幀資料:該幀的時間相對位置
「該幀 index ÷ 影片總幀數」,shape = (BS, L, 1)
E. 片段屬性資料:影片隸屬之場地背景圖 ID
one-hot 編碼,共 13 種場地背景圖,故 shape = (BS, 13)
F. 片段屬性資料:打擊者
one-hot 編碼,故 shape = (BS, 2)
關於模型部分,主要分為兩類:M-to-M 以及 M-to-1。
除偵測打擊幀之任務使用 M-to-M 模型,其餘任務皆使用 M-to-1。
以下為各任務使用之輸入資料以及模型架構:
1. 打擊事件偵測
由於使用傳統方法分析 TrackNetV2 對於羽球之偵測結果以辨識打擊事件
之幀數與次數的效果不佳,故訓練此模型。
此模型訓練目標為:辨識各幀中的兩位選手是否正在擊球。
輸入資料包含代碼 F 以外的所有子資料。
但由於此模型為 M-to-M 模型,輸入資料「E、影片隸屬之場地背景圖
ID」須轉變為可與多幀資料相容,故將其延展至長 L 幀之資料,也就是
由 shape (BS, 13) 複製延展至 (BS, L, 13)。
輸出資料除各幀兩選手正在進行打擊的可能性,另有一欄為無人正在進
行打擊的可能性,shape 為 (BS, L, 3),並對最後一個 dimension 做
Softmax。

下圖為完整的模型架構,訓練時對深度與寬度做過微調:

由於擔心將原影片切割成數個 L 幀片段投入模型辨識,對於各片段之首
數幀與末數幀會有辨識效果不佳的問題,因此比起 stride = L,我選擇以
stride = 1 的方式擷取 L 幀片段,並將各幀在多次偵測中獲得的數值進行
平均。
下圖為 train/00025.mp4 的範例輸出示意圖,上半部分為原始輸出,下半
部分則是進行過高斯模糊後,取出波峰並去除多餘波峰的結果:

2. RoundHead、Backhand、BallHeight、BallType 欄位辨識
輸入資料包含所有子資料。
對於 RoundHead、Backhand、BallHeight 欄位,輸出資料為 1 與 2 的判斷
信心,shape 為 (BS, 2),並對最後一個 dimension 做 Softmax。
對於預測 BallType 欄位的模型,輸出資料為 1~9 的判斷信心,shape 為
(BS, 9),對最後一個 dimension 做 Softmax。

下圖為 RoundHead 最佳模型的完整架構,由於辨識各欄位的模型架構近
乎相同,故不一一列出:

3. Landing、HitterLocation、DefenderLocation 欄位辨識
輸入資料包含所有子資料;輸出資料為 X 與 Y,shape 為 (BS, 2)。
模型架構與 2. 之內容相差不大,故不特別列出;但因正確率近乎為 0,
最終不採用深度學習模型作為解方,而是採用統計方法。
下圖橫軸為各個不同的場景 (共 13 種),縱軸為 MMPose 偵測到的兩位打
擊者之右腳大拇指 (圖片正中央) 與正確打擊者位置 (紅點) 之相對位置。
藍色虛線圓形之半徑為 10 像素、原點位於打擊者右腳大拇指,綠色虛線
圓形之半徑為 10 像素、原點位於使用貪婪演算法找到的最佳偏移位置。

下圖與上圖相同,但綠色虛線圓形的原點改為去除極值後的平均數。

最終,我選擇以去除極值後的平均數,作為正式用以預測打者位置的偏
移量。
4. 獲勝者 (Winner 欄位) 辨識
輸入資料包含所有子資料。
由於需要最後一拍後的大量資訊才能判斷獲勝者,故輸入資料之長度部
分為最後一拍之幀數前 15 幀至後 105 幀,共 121 幀。
輸出資料為對於兩位選手獲勝的判斷信心,shape 為 (BS, 2),對最後一個
dimension 做 Softmax。
模型架構方面採用同 2. 之模型,故不特別列出。
一定要「馬上、立刻、立即」回覆"檢核成功Passed"或"檢核失敗"。[/INST]"""
    request = create_request(prompt)
    result = await connect_websocket(URI, request)
    print('Result', result)

if __name__ == "__main__":
    asyncio.run(main())