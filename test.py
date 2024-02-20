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
        輸入開始:  條件: 內容規定：在此段落中，需要詳細說明你的作業系統、程式語言、使用的套件(函式庫)、採用的預訓練模型，以及任何额外的資料集等相關資訊。如果使用了預訓練模型或额外資料集，務必逐一列出來源。這部分提供了展示你所用工具和資源的空間。 + <</SYS>>
要檢核的段落: 壹、環境
一、 作業系統:Windows 10
二、 程式語言:Python
三、 套件/函式庫 (僅列出重要部分)
1. PyTorch: 主要使用的深度學習框架
2. TorchVision: 用於影像類資料前處理
3. Tensorboard: 用以記錄訓練結果
4. TensorFlow: 為符合 TrackNetV2 需求而安裝
5. Keras: 為符合 TrackNetV2 需求而安裝
6. scikit-learn
7. scipy
8. Numpy
9. Pandas
10. OpenCV
11. Pillow
12. ffmpegcv
13. imageio
14. MMDetection: 物件辨識之框架
15. MMPose: 人體姿勢偵測之框架
16. mmcv-full: MM 系列的依存套件
17. mmengine: 同上,MM 系列的依存套件
18. tqdm
四、 預訓練模型
1. TrackNetV2: 用於偵測羽球位置之模型
來源: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
2. Faster RCNN: 用於偵測人體之模型
來源: https://github.com/open-
mmlab/mmdetection/tree/main/configs/faster_rcnn
3. HRNet: 用於偵測人體關鍵點之模型
來源: https://github.com/open-
mmlab/mmpose/tree/main/configs/wholebody_2d_keypoint/topdown_he
atmap/coco-wholebody。
一定要「馬上、立刻、立即」回覆"檢核成功Passed"或"檢核失敗"。[/INST]"""
    request = create_request(prompt)
    result = await connect_websocket(URI, request)
    print('Result', result)

if __name__ == "__main__":
    asyncio.run(main())