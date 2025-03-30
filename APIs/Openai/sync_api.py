from openai import OpenAI
import os
from dotenv import load_dotenv

class OpenAI_agent:
    def __init__(self, model_name, temperature):
        """
        初始化 OpenAI 同步代理。
        
        Args:
            model_name (str): 要使用的 OpenAI 模型名稱 (如 "gpt-4").
            temperature (float): 溫度參數，決定生成回應的隨機性，範圍為 0.0 到 1.0。
        """
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        load_dotenv()
        print("API_KEY:", os.getenv("OPENAI_API_KEY")[:5] + "..." + os.getenv("OPENAI_API_KEY")[-5:])
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def send_text(self, messages, temperature=1, max_tokens=2048, top_p=1, frequency_penalty=0, presence_penalty=0, stop=[]):
        """
        傳送訊息至 OpenAI API 並取得回應。

        Args:
            messages (List[Dict[str, Union[str, List[Dict[str, str]]]]]): 訊息歷史紀錄，包含 system, user, assistant。
            temperature (Optional[float], 預設 1.0): 調整模型輸出的隨機性 (0.0 為最保守, 1.0 最自由)。
            max_tokens (Optional[int], 預設 2048): 回應的最大 token 數。
            top_p (Optional[float], 預設 1.0): 使用 nucleus sampling (0.0 到 1.0)。
            frequency_penalty (Optional[float], 預設 0.0): 減少重複內容的程度 (-2.0 到 2.0)。
            presence_penalty (Optional[float], 預設 0.0): 增加多樣性的程度 (-2.0 到 2.0)。
            stop (Optional[List[str]], 預設 None): 停止條件。

        Returns:
            str: OpenAI 回應的內容，若發生錯誤則回傳錯誤訊息。
        """
        try:
            completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            # stop=stop
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"error: {str(e)}"


    def update_messages(self, input_text, messages = None, sys_message = ""):
        """
        更新對話歷史，並將使用者輸入傳送至 OpenAI API 取得回應。

        Args:
            input_text (str): 使用者的輸入訊息。
            sys_message (Optional[str], 預設 ""): 若有新 system 訊息則加入對話。
            messages (Optional[List[Dict[str, Union[str, List[Dict[str, str]]]]]], 預設 None): 先前的對話紀錄。

        Returns:
            Tuple[List[Dict[str, Union[str, List[Dict[str, str]]]]], str]: 更新後的對話紀錄與 OpenAI 回應。
        """
        if messages == None:
            messages = [
                {"role": "system", "content": str(sys_message)},
            ]
        elif sys_message != "":
            messages.append(
                {"role": "system", "content": str(sys_message)}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                ],
            }
        )
        response = self.send_text(
            messages,
        )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                    "type": "text",
                    "text": response
                    }
                ]
            }
        )
        return messages, response