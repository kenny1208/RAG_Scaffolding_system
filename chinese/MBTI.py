import requests
from dotenv import load_dotenv
import os
import json
import re 

# 加載 .env 文件中的環境變量
load_dotenv()

# 從環境變量中獲取 API 密鑰
api_key = os.getenv("GEMINI_API_KEY")

# 檢查是否成功加載 API 密鑰
if not api_key:
    print("[ERROR] Google API Key not found in .env file")
    print("[INFO] Please create a .env file and add GEMINI_API_KEY=your_api_key")
    exit(1)

# 定義 API URL 和請求的內容
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
headers = {'Content-Type': 'application/json'}

# 定義要發送給 API 的 prompt
prompt = "請生成 30 題與 MBTI 測試相關的問題，這些問題應該有 5 個選項: 非常同意，同意，中立，不同意，非常不同意，並主要是用作測試使用者的學習態度及其個性。"

# 發送請求
response = requests.post(
    API_URL,
    headers=headers,
    json={
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
)

# 假設 response_json 是從 API 返回的生成問題的 JSON
response_json = response.json()

# 提取問題文本
questions_text = response_json["candidates"][0]["content"]["parts"][0]["text"]

# 使用正則表達式來提取所有問題，包括問題編號
# \d+ 表示匹配問題編號，如 1, 2, 3...，\s* 匹配後續的空格，(.*?) 是問題文本
questions = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|\Z)", questions_text)

# 檢查問題數量是否為50，如果不足，補充或調整
if len(questions) < 50:
    print(f"提取到的問題數量不足：{len(questions)}，將進行調整")

# 顯示問題並讓用戶回答
def ask_questions(questions):
    answers = {
        "E/I": 0,  # 外向 vs 內向
        "S/N": 0,  # 實際 vs 直覺
        "T/F": 0,  # 思考 vs 情感
        "J/P": 0   # 判斷 vs 知覺
    }
    user_answers = []

    # 確保顯示所有問題
    for i, question in enumerate(questions):
        if i > 51:  # 如果超過50個問題就停止
            break
        
        print(f"問題 {i+1}: {question}")
        print("選項: 非常同意、同意、中立、不同意、非常不同意")
        
        while True:  # 這裡使用循環來確保用戶只能選擇有效答案
            answer = input("請選擇您的答案（1-5）：")
            
            # 檢查答案是否有效
            if answer == "1":
                score = 5  # 非常同意
                break
            elif answer == "2":
                score = 4  # 同意
                break
            elif answer == "3":
                score = 3  # 中立
                break
            elif answer == "4":
                score = 2  # 不同意
                break
            elif answer == "5":
                score = 1  # 非常不同意
                break
            else:
                print("無效選項，請重新選擇。")
        
        # 根據問題的類型來更新對應的維度分數
        if i % 4 == 0:  # 假設每四個問題對應一個維度
            answers["E/I"] += score
        elif i % 4 == 1:
            answers["S/N"] += score
        elif i % 4 == 2:
            answers["T/F"] += score
        elif i % 4 == 3:
            answers["J/P"] += score

        # 收集用戶答案
        user_answers.append({"question": question, "answer": answer})

    # 根據分數來計算 MBTI 類型
    mbti = calculate_mbti_type(answers)
    print(f"您的 MBTI 類型是: {mbti}")

    # 創建一個字典來保存最終的結果
    result = {
        "answers": user_answers,
        "mbti_type": mbti
    }

    # 將結果寫入 JSON 文件
    with open("mbti_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# 根據累積的分數來確定 MBTI 類型
def calculate_mbti_type(answers):
    mbti_type = ""

    # 假設每個維度的分數範圍從 1 到 5
    mbti_type += "E" if answers["E/I"] > 0 else "I"
    mbti_type += "S" if answers["S/N"] > 0 else "N"
    mbti_type += "T" if answers["T/F"] > 0 else "F"
    mbti_type += "J" if answers["J/P"] > 0 else "P"

    return mbti_type

# 問題回答並展示 MBTI 類型
ask_questions(questions)