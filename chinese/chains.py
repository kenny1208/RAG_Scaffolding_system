# -*- coding: big5 -*-

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStoreRetriever 

# Import models if needed for context (usually not directly needed here)
# from models import ...

def format_docs(docs):
    """Helper function to format retrieved documents."""
    return "\n\n".join([d.page_content for d in docs])

# --- Chain Creation Functions ---

def create_learning_style_survey(chat_model: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專精於學習風格評估的教育專家。
        請設計一份簡潔但有效的學習風格評估問卷，包含 5 個多選題。
        每個問題應有 3 個選項，用於判斷學生是否主要是：
        1. 視覺型學習者
        2. 聽覺型學習者
        3. 動覺型學習者

        請將您的回應格式化為一份問卷，包含編號的問題和字母標記的選項。"""),
        HumanMessagePromptTemplate.from_template("設計一份學習風格評估問卷。")
    ])
    return prompt | chat_model | StrOutputParser()

def create_pretest_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專精於教育評估設計的專家。
        根據提供的內容，設計一份前測（Pre-Test），以評估學生在該主題上的現有知識水平。

        請設計涵蓋不同難度級別的問題：簡單、中等和困難。
        對於每個問題，請提供：
        1. 問題文本
        2. 四個單選選項（A, B, C, D）
        3. 正確答案
        4. 為什麼正確的解釋
        5. 難度級別

        您必須遵循以下精確的 JSON 格式：
        {
          "title": "前測：[主題]",
          "description": "此測驗將評估您對[主題]的現有知識",
          "questions": [
            {
              "question": "問題文本？",
              "choices": ["A. 選項 A", "B. 選項 B", "C. 選項 C", "D. 選項 D"],
              "correct_answer": "A. 選項 A",
              "explanation": "為什麼 A 是正確答案的解釋",
              "difficulty": "簡單"
            }
          ]
        }

        請根據提供的內容生成總共 5 個問題，並包含不同難度級別的問題。
        """),
        HumanMessagePromptTemplate.from_template("""根據以下內容生成一份前測：

        {context}
        """)
    ])

    pretest_chain = (
        {
            # Pass the input directly to the retriever, then format
            "context": RunnableLambda(lambda inputs: inputs.get("topic", "general knowledge")) | retriever | format_docs
        }
        | prompt
        | chat_model
        | JsonOutputParser() 
    )
    return pretest_chain


def create_learning_path_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
         SystemMessage(content="""您是一位專精於個人化學習路徑設計的教育課程設計專家。
        根據提供的學生檔案、測驗結果和內容，創建一條適合他自學的學習路徑。

        您的學習路徑應該：
        1. 針對學生的學習風格、知識水平和興趣進行量身定制
        2. 包含清晰的學習目標
        3. 遵循鷹架原則，逐步增加難度並減少支持

        您的回應必須遵循以下精確的 JSON 格式：
        {
          "title": "針對[主題]的個人化學習路徑",
          "description": "此學習路徑針對[name]的學習風格和當前知識水平進行量身定制",
          "objectives": ["目標 1", "目標 2", "目標 3"],
          "modules": [
            {
              "title": "章節 1: [標題]",
              "description": "章節描述",
              "activities": [
                {
                  "type": "閱讀",
                  "title": "活動標題",
                  "description": "活動描述",
                  "difficulty": "初學者"
                }
              ],
              "resources": ["講義章節1-1", "講義章節1-2"],
            }
          ]
        }
        """),
        HumanMessagePromptTemplate.from_template("""根據以下內容生成個人化學習路徑：

        學生檔案：
        {profile}

        測驗結果：
        {test_results}

        相關內容：
        {context}
        """)
    ])

    learning_path_chain = (
        {
            "profile": RunnablePassthrough(), # Pass profile dict directly
            "test_results": RunnablePassthrough(), # Pass results dict directly
            # Retrieve context based on a general topic or derived from profile/results if needed
            "context": RunnableLambda(lambda inputs: inputs.get("topic", "relevant subject matter")) | retriever | format_docs
        }
        | prompt
        | chat_model
        | JsonOutputParser() # Assumes LearningPath model or dict output
    )
    return learning_path_chain

def create_peer_discussion_ai(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是「學習夥伴」，一個友善且有幫助的 AI 同儕，與學生進行有建設性的討論。
        您的角色是：
        1. 模擬一位也在學習該主題但有一定見解的同儕
        2. 提出促進批判性思考的深思熟慮的問題
        3. 提供溫和的指導，而不是直接給出答案
        4. 以對話的方式表達想法，像是學生之間的交流
        5. 使用蘇格拉底式提問法幫助學生發現答案
        6. 鼓勵並保持積極的態度

        根據提供的相關內容回應，但不要只是簡單地背誦資訊。
        而是以自然的方式進行來回討論，就像一起學習一樣。
        """),
        HumanMessagePromptTemplate.from_template("""學生希望討論這個主題：

        主題: {topic}

        相關內容:
        {context}

        學生訊息:
        {message}
        """)
    ])

    discussion_chain = (
        {
            "topic": RunnablePassthrough(), # Pass topic string directly
            "message": RunnablePassthrough(), # Pass message string directly
            "context": RunnableLambda(lambda inputs: inputs["topic"]) | retriever | format_docs
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return discussion_chain

def create_posttest_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育評估設計師。
        根據提供的學習章節內容和學生的當前知識水平，設計一份後測，包含多選題以評估學生的學習成果。

        難度應與學生的當前水平相符：
        - 初學者：更多簡單問題（70%），一些中等問題（30%）
        - 中級者：一些簡單問題（30%），主要是中等問題（50%），一些困難問題（20%）
        - 高級者：一些中等問題（30%），主要是困難問題（70%）

        設計的問題應測試學生對內容的理解、應用和分析能力。

        對於每個問題，請提供：
        1. 問題文本
        2. 四個多選選項（A, B, C, D）
        3. 正確答案
        4. 為什麼正確的解釋
        5. 難度級別

        您必須遵循以下精確的 JSON 格式：
        {
          "title": "後測：[主題]",
          "description": "此測驗將評估您對[主題]的學習成果",
          "questions": [
            {
              "question": "問題文本？",
              "choices": ["A. 選項 A", "B. 選項 B", "C. 選項 C", "D. 選項 D"],
              "correct_answer": "A. 選項 A",
              "explanation": "為什麼 A 是正確答案的解釋",
              "difficulty": "簡單"
            }
          ]
        }

        根據學生的水平生成總共 5 個問題，並適當分配難度。
        """),
        HumanMessagePromptTemplate.from_template("""根據以下內容生成一份後測：

        學生的當前知識水平: {knowledge_level}
        章節內容主題: {module_topic}
        相關內容:
        {context}
        """)
    ])

    posttest_chain = (
        {
            "knowledge_level": RunnablePassthrough(), # Pass level string directly
            "module_topic": RunnablePassthrough(), # Pass topic string directly
            "context": RunnableLambda(lambda inputs: inputs["module_topic"]) | retriever | format_docs
        }
        | prompt
        | chat_model
        | JsonOutputParser() # Assumes Test model or dict output
    )
    return posttest_chain

def create_learning_log_prompter(chat_model: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位反思學習教練，專門幫助學生創建有意義的學習日誌。
        根據學生完成的學習章節和測驗結果，引導他們反思自己的學習。

        提出深思熟慮的開放式問題以促進反思，包括：
        1. 他們學到了什麼（關鍵概念和見解）
        2. 他們對學習過程的感受
        3. 他們覺得有挑戰的地方
        4. 他們仍然有什麼問題

        您的目標是幫助學生創建一份豐富且有反思性的學習日誌，對他們的成長有價值。
        """),
        HumanMessagePromptTemplate.from_template("""幫助學生基於以下內容創建學習日誌反思：

        完成的章節: {module_title}

        章節內容摘要: {module_summary}

        測驗結果: {test_results}
        """)
    ])
    return prompt | chat_model | StrOutputParser()

def create_learning_log_analyzer(chat_model: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育分析師，專門分析學生的學習日誌。
        根據學生的學習日誌，評估：

        1. 對關鍵概念的理解程度
        2. 優勢和自信的領域
        3. 混淆或誤解的領域
        4. 對材料的情感反應
        5. 學習風格的指標

        將您的回應格式化為以下精確的 JSON 結構:
        {
          "understanding_level": "高/中/低",
          "strengths": ["優勢 1", "優勢 2"],
          "areas_for_improvement": ["改進領域 1", "改進領域 2"],
          "emotional_response": "對情感反應的描述",
          "learning_style_indicators": ["指標 1", "指標 2"],
          "recommended_next_steps": ["建議步驟 1", "建議步驟 2"],
          "suggested_resources": ["資源 1", "資源 2"]
        }
        """),
        HumanMessagePromptTemplate.from_template("""分析以下學習日誌：

        學生: {student_name}
        主題: {topic}
        學習日誌內容:
        {log_content}
        """)
    ])
    return prompt | chat_model | JsonOutputParser() # Assumes dict output

def create_knowledge_level_assessor(chat_model: ChatGoogleGenerativeAI):
    # Note: Original code had StrOutputParser but the prompt asks for JSON.
    # Assuming JSON output is desired based on the prompt's format specification.
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位教育評估專家。
        根據學生的測驗結果，確定他們在此特定主題上的知識水平。

        考慮：
        1. 正確答案的數量
        2. 正確回答的問題難度
        3. 答案模式（一致的理解與差距）

        將學生的知識水平分類為：
        - 初學者：基本熟悉，理解簡單概念
        - 中級者：良好的核心概念理解，一定的應用能力
        - 高級者：深刻理解，能將概念應用於新情境

        為您的評估提供簡短的理由。

        將您的回應格式化為 JSON 對象：
        {
          "knowledge_level": "初學者/中級者/高級者",
          "justification": "對評估的簡短解釋",
          "strengths": ["優勢 1", "優勢 2"],
          "areas_for_improvement": ["改進領域 1", "改進領域 2"],
          "recommended_focus": "學生接下來應該專注於什麼"
        }
        """),
        HumanMessagePromptTemplate.from_template("""根據以下測驗結果評估學生的知識水平：

        測驗: {test_title}

        問題和答案:
        {test_results_details}
        """) # Changed key name for clarity
    ])
    return prompt | chat_model | JsonOutputParser() # Changed to JsonOutputParser

def create_module_content_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育內容創作者。
        根據提供的章節主題以及學生的學習風格和知識水平，創建引人入勝的教育內容。

        您的內容應：
        1. 針對學生的學習風格（視覺型、聽覺型或動覺型）進行量身定制
        2. 完全符合鷹架教育理論
        3. 適合學生的知識水平
        4. 包含關鍵概念的清晰解釋
        5. 結構清晰，包含明確的部分和標題
        6. 以關鍵點的簡短總結結尾
        7. 不要使用圖表或圖像，可以使用表格

        使用 markdown 格式化您的內容以提高可讀性。
        """),
        HumanMessagePromptTemplate.from_template("""為以下內容創建教育內容：

        章節主題: {module_topic}
        學生學習風格: {learning_style}
        學生知識水平: {knowledge_level}

        相關來源材料:
        {context}
        """)
    ])

    content_chain = (
        {
            "module_topic": RunnablePassthrough(), # Pass topic string
            "learning_style": RunnablePassthrough(), # Pass style string
            "knowledge_level": RunnablePassthrough(), # Pass level string
            "context": RunnableLambda(lambda inputs: inputs["module_topic"]) | retriever | format_docs
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return content_chain