# -*- coding: big5 -*-


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional, Any
import nltk
import os
import json
import datetime
import uuid
import pathlib
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from dotenv import load_dotenv
from glob import glob

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("learning_logs", exist_ok=True)
os.makedirs("student_profiles", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

console = Console()
# Download the required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    console.print("[bold red]ERROR: Google API Key not found in .env file[/bold red]")
    console.print("[yellow]Please create a .env file with GOOGLE_API_KEY=your_api_key[/yellow]")
    exit(1)

# Define output parsers for structured data
class Question(BaseModel):
    question: str = Field(description="The question text")
    choices: List[str] = Field(description="List of multiple choice options")
    correct_answer: str = Field(description="The correct answer")
    explanation: str = Field(description="Explanation of the correct answer")
    difficulty: str = Field(description="Difficulty level: 'easy', 'medium', or 'hard'")

class Test(BaseModel):
    title: str = Field(description="Title of the test")
    description: str = Field(description="Brief description of the test")
    questions: List[Question] = Field(description="List of questions")

class LearningPath(BaseModel):
    title: str = Field(description="Title of the learning path")
    description: str = Field(description="Description of the learning path")
    objectives: List[str] = Field(description="Learning objectives")
    modules: List[Dict[str, Any]] = Field(description="List of learning modules")

class StudentProfile(BaseModel):
    id: str = Field(description="Unique student ID")
    name: str = Field(description="Student name")
    learning_style: str = Field(description="Visual, auditory, or kinesthetic")
    current_knowledge_level: str = Field(description="Beginner, intermediate, or advanced")
    strengths: List[str] = Field(description="Academic strengths")
    areas_for_improvement: List[str] = Field(description="Areas that need improvement")
    interests: List[str] = Field(description="Academic interests")
    learning_history: List[Dict[str, Any]] = Field(description="History of learning activities")

class LearningLog(BaseModel):
    id: str = Field(description="Unique log ID")
    student_id: str = Field(description="Student ID")
    timestamp: str = Field(description="ISO format timestamp")
    topic: str = Field(description="Topic studied")
    content: str = Field(description="Log content")
    reflections: List[str] = Field(description="Student reflections")
    questions: List[str] = Field(description="Questions raised by student")
    next_steps: List[str] = Field(description="Planned next steps")

# Initialize language models and embeddings
def initialize_models():
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-pro-latest",
        temperature=0.2
    )
    
    embedding = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key, 
        model="models/embedding-001"
    )
    
    return chat_model, embedding

# Initialize RAG system with document loading and chunking
def initialize_rag_system(embedding):
    # Check if we already have a persisted vector store
    if os.path.exists("vectorstore") and len(os.listdir("vectorstore")) > 0:
        console.print("[yellow]Loading existing vector store...[/yellow]")
        vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embedding)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # If not, create a new one from documents
    console.print("[yellow]Creating new vector store from documents...[/yellow]")
    pdf_paths = glob("data/*.pdf")
    
    if not pdf_paths:
        console.print("[bold red]No PDF documents found in data/ directory![/bold red]")
        console.print("[yellow]Please add PDF documents to the data/ directory[/yellow]")
        exit(1)
    
    all_pages = []
    for path in pdf_paths:
        console.print(f"[green]Loading document: {path}[/green]")
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        all_pages.extend(pages)
    
    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_pages)
    
    console.print(f"[green]Created {len(chunks)} text chunks for retrieval[/green]")
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding,
        persist_directory="vectorstore"
    )
    vectorstore.persist()
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Create or load student profile
def manage_student_profile():
    profiles = glob("student_profiles/*.json")
    
    if profiles:
        console.print("[bold]Select an existing profile or create a new one:[/bold]")
        console.print("0. Create new student profile")
        
        for i, profile_path in enumerate(profiles, 1):
            profile_name = pathlib.Path(profile_path).stem
            console.print(f"{i}. {profile_name}")
        
        choice = int(Prompt.ask("Enter your choice", choices=[str(i) for i in range(len(profiles) + 1)]))
        
        if choice == 0:
            return create_new_profile()
        else:
            with open(profiles[choice-1], "r") as f:
                return StudentProfile.parse_obj(json.load(f))
    else:
        return create_new_profile()

def create_new_profile():
    name = Prompt.ask("Enter student name")
    student_id = str(uuid.uuid4())[:8]
    
    profile = StudentProfile(
        id=student_id,
        name=name,
        learning_style="",
        current_knowledge_level="",
        strengths=[],
        areas_for_improvement=[],
        interests=[],
        learning_history=[]
    )
    
    # Save the basic profile
    with open(f"student_profiles/{student_id}.json", "w") as f:
        f.write(profile.json(indent=4))
    
    return profile

# Create chains for different functionalities
def create_learning_style_survey(chat_model):
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

def create_pretest_generator(chat_model, retriever):
    # Define the prompt template for generating pre-test questions
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專精於教育評估設計的專家。
        根據提供的內容，設計一份前測（Pre-Test），以評估學生在該主題上的現有知識水平。
        
        請設計涵蓋不同難度級別的問題：簡單、中等和困難。
        對於每個問題，請提供：
        1. 問題文本
        2. 四個多選選項（A, B, C, D）
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
    
    # Create the chain
    json_parser = JsonOutputParser()
    
    pretest_chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return pretest_chain

def create_learning_path_generator(chat_model, retriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專精於個人化學習路徑設計的教育課程設計專家。
        根據提供的學生檔案、測驗結果和內容，創建一條個人化的學習路徑。
        
        您的學習路徑應該：
        1. 針對學生的學習風格、知識水平和興趣進行量身定制
        2. 包含清晰的學習目標
        3. 將內容分解為邏輯模組，並估算所需時間
        4. 包括適合學生學習風格的練習活動和資源
        5. 遵循鷹架原則，逐步增加難度並減少支持
        
        您的回應必須遵循以下精確的 JSON 格式：
        {
          "title": "針對[主題]的個人化學習路徑",
          "description": "此學習路徑針對[name]的學習風格和當前知識水平進行量身定制",
          "objectives": ["目標 1", "目標 2", "目標 3"],
          "modules": [
            {
              "title": "模組 1: [標題]",
              "description": "模組描述",
              "activities": [
                {
                  "type": "閱讀",
                  "title": "活動標題",
                  "description": "活動描述",
                  "estimated_time": "20 分鐘",
                  "difficulty": "初學者"
                }
              ],
              "resources": ["資源 1", "資源 2"],
              "assessment": "模組評估描述"
            }
          ]
        }
        """),
        HumanMessagePromptTemplate.from_template("""根據以下內容生成個人化學習路徑：
        
        學生檔案：
        {profile}
        
        測驗結果：
        {test_results}
        
        內容：
        {context}
        """)
    ])
    
    learning_path_chain = (
        {
            "profile": RunnablePassthrough(),
            "test_results": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return learning_path_chain

def create_peer_discussion_ai(chat_model, retriever):
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
            "topic": RunnablePassthrough(),
            "message": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return discussion_chain

def create_posttest_generator(chat_model, retriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育評估設計師。
        根據提供的學習模組內容和學生的當前知識水平，設計一份後測，包含多選題以評估學生的學習成果。
        
        難度應與學生的當前水平相符：
        - 初學者：更多簡單問題（70%），一些中等問題（30%）
        - 中級者：一些簡單問題（30%），主要是中等問題（50%），一些困難問題（20%）
        - 高級者：一些中等問題（40%），主要是困難問題（60%）
        
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
        模組內容: {context}
        """)
    ])
    
    posttest_chain = (
        {
            "knowledge_level": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return posttest_chain

def create_learning_log_prompter(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位反思學習教練，專門幫助學生創建有意義的學習日誌。
        根據學生完成的學習模組和測驗結果，引導他們反思自己的學習。
        
        提出深思熟慮的開放式問題以促進反思，包括：
        1. 他們學到了什麼（關鍵概念和見解）
        2. 他們對學習過程的感受
        3. 他們覺得有挑戰的地方
        4. 他們仍然有什麼問題
        5. 他們如何應用所學的內容
        6. 他們接下來想學什麼
        
        您的目標是幫助學生創建一份豐富且有反思性的學習日誌，對他們的成長有價值。
        """),
        HumanMessagePromptTemplate.from_template("""幫助學生基於以下內容創建學習日誌反思：
        
        完成的模組: {module_title}
        
        模組內容摘要: {module_summary}
        
        測驗結果: {test_results}
        """)
    ])
    
    learning_log_chain = prompt | chat_model | StrOutputParser()
    
    return learning_log_chain

def create_learning_log_analyzer(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育分析師，專門分析學生的學習日誌。
        根據學生的學習日誌，評估：
        
        1. 對關鍵概念的理解程度
        2. 優勢和自信的領域
        3. 混淆或誤解的領域
        4. 對材料的情感反應
        5. 學習風格的指標
        6. 潛在的學習下一步
        
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
    
    learning_log_analysis_chain = prompt | chat_model | StrOutputParser()
    
    return learning_log_analysis_chain

def create_knowledge_level_assessor(chat_model):
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
        {test_results}
        """)
    ])
    
    knowledge_level_chain = prompt | chat_model | StrOutputParser()
    
    return knowledge_level_chain

def create_module_content_generator(chat_model, retriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育內容創作者。
        根據提供的模組主題以及學生的學習風格和知識水平，創建引人入勝的教育內容。
        
        您的內容應：
        1. 針對學生的學習風格（視覺型、聽覺型或動覺型）進行量身定制
        2. 適合學生的知識水平
        3. 包含關鍵概念的清晰解釋
        4. 使用示例和類比來說明要點
        5. 包括適合知識水平的鷹架元素
        6. 結構清晰，包含明確的部分和標題
        7. 以關鍵點的簡短總結結尾
        
        使用 markdown 格式化您的內容以提高可讀性。
        """),
        HumanMessagePromptTemplate.from_template("""為以下內容創建教育內容：
        
        模組主題: {module_topic}
        學生學習風格: {learning_style}
        學生知識水平: {knowledge_level}
        
        相關來源材料:
        {context}
        """)
    ])
    
    content_chain = (
        {
            "module_topic": RunnablePassthrough(),
            "learning_style": RunnablePassthrough(),
            "knowledge_level": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return content_chain

# Core system functions
def conduct_learning_style_survey(chat_model, student_profile):
    console.print("\n[bold cyan]===== 學習風格評估 =====[/bold cyan]")
    
    survey_chain = create_learning_style_survey(chat_model)
    survey = survey_chain.invoke({})
    
    console.print(Markdown(survey))
    console.print("\n[bold yellow]請回答每個問題以幫助確定您的學習風格。[bold yellow]")
    
    # 處理問卷結果
    console.print("\n[bold]完成問卷後，哪種學習風格最符合您？[/bold]")
    console.print("1. 視覺型學習者")
    console.print("2. 聽覺型學習者")
    console.print("3. 動覺型學習者")
    
    style_choice = Prompt.ask("選擇您的主要學習風格", choices=["1", "2", "3"])
    
    learning_styles = {
        "1": "視覺型",
        "2": "聽覺型",
        "3": "動覺型"
    }
    
    student_profile.learning_style = learning_styles[style_choice]
    administer_pretest
    # 儲存更新後的檔案
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return student_profile

def administer_pretest(chat_model, retriever, student_profile):
    console.print("\n[bold cyan]===== 前測 =====[/bold cyan]")
    console.print("[yellow]根據內容生成前測...[/yellow]")
    
    pretest_chain = create_pretest_generator(chat_model, retriever)
    pretest_json = pretest_chain.invoke("")
    
    try:
        pretest = json.loads(pretest_json)
    except json.JSONDecodeError:
        console.print("[bold red]解析前測 JSON 時出錯。使用備用方法。[bold red]")
        # 如果 JSON 被其他文字包圍，提取 JSON
        import re
        json_match = re.search(r'({[\s\S]*})', pretest_json)
        if json_match:
            try:
                pretest = json.loads(json_match.group(1))
            except:
                console.print("[bold red]無法解析前測 JSON。使用預設測驗。[bold red]")
                pretest = {
                    "title": "預設前測",
                    "description": "由於解析問題，這是一個備用測驗。",
                    "questions": [
                        {
                            "question": "RAG 在教育系統中的主要目的是什麼？",
                            "choices": ["A. 生成隨機問題", "B. 檢索生成所需的相關資訊", 
                                       "C. 取代教師", "D. 在學習過程中播放音樂"],
                            "correct_answer": "B. 檢索生成所需的相關資訊",
                            "explanation": "RAG（檢索增強生成）有助於準確的資訊檢索。",
                            "difficulty": "中等"
                        }
                    ]
                }
    
    console.print(f"\n[bold green]{pretest['title']}[/bold green]")
    console.print(f"[italic]{pretest['description']}[/italic]\n")
    
    # 執行測驗
    score = 0
    total_questions = len(pretest['questions'])
    results = []
    
    for i, q in enumerate(pretest['questions']):
        console.print(f"\n[bold]問題 {i+1}:[/bold] {q['question']}")
        for choice in q['choices']:
            console.print(f"  {choice}")
        
        answer = Prompt.ask("\n您的答案 (A, B, C 或 D)").upper()
        correct_letter = q['correct_answer'][0].upper()
        
        if answer == correct_letter:
            score += 1
            console.print("[bold green]正確！[/bold green]")
        else:
            console.print(f"[bold red]錯誤。正確答案是 {q['correct_answer']}[/bold red]")
        
        console.print(f"[italic]{q['explanation']}[/italic]")
        
        results.append({
            "question": q['question'],
            "student_answer": answer,
            "correct_answer": q['correct_answer'],
            "is_correct": answer == correct_letter,
            "difficulty": q['difficulty']
        })
    
    # 計算分數
    percentage = (score / total_questions) * 100
    console.print(f"\n[bold]測驗完成！您的分數：{score}/{total_questions} ({percentage:.1f}%)[/bold]")
    
    # 評估知識水平
    if percentage >= 80:
        knowledge_level = "高級"
    elif percentage >= 50:
        knowledge_level = "中級"
    else:
        knowledge_level = "初學者"
    
    console.print(f"[yellow]根據您的前測結果，您的當前知識水平是：[bold]{knowledge_level}[/bold][/yellow]")
    
    # 更新學生檔案
    student_profile.current_knowledge_level = knowledge_level
    student_profile.learning_history.append({
        "activity_type": "前測",
        "timestamp": datetime.datetime.now().isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "knowledge_level": knowledge_level
    })
    
    # 儲存更新後的檔案
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return pretest, results, knowledge_level

def generate_learning_path(chat_model, retriever, student_profile, pretest_results):
    console.print("\n[bold cyan]===== 生成個人化學習路徑 =====[/bold cyan]")
    console.print("[yellow]根據您的檔案和測驗結果創建學習路徑...[/yellow]")
    
    # 格式化測驗結果以供鏈使用
    test_results_formatted = json.dumps({
        "score_percentage": (sum(1 for r in pretest_results if r["is_correct"]) / len(pretest_results)) * 100,
        "knowledge_level": student_profile.current_knowledge_level,
        "strengths": [r["question"] for r in pretest_results if r["is_correct"]],
        "weaknesses": [r["question"] for r in pretest_results if not r["is_correct"]]
    })
    
    # 格式化檔案以供鏈使用
    profile_formatted = json.dumps({
        "name": student_profile.name,
        "learning_style": student_profile.learning_style,
        "current_knowledge_level": student_profile.current_knowledge_level,
        "interests": student_profile.interests
    })
    
    learning_path_chain = create_learning_path_generator(chat_model, retriever)
    learning_path_json = learning_path_chain.invoke({
    "profile": profile_formatted,
    "test_results": test_results_formatted,
    "context": ""
    })
    
    try:
        learning_path = json.loads(learning_path_json)
    except json.JSONDecodeError:
        console.print("[bold red]解析學習路徑 JSON 時出錯。使用備用方法。[bold red]")
        # 如果 JSON 被其他文字包圍，提取 JSON
        import re
        json_match = re.search(r'({[\s\S]*})', learning_path_json)
        if json_match:
            try:
                learning_path = json.loads(json_match.group(1))
            except:
                console.print("[bold red]無法解析學習路徑 JSON。使用預設路徑。[bold red]")
                learning_path = {
                    "title": f"針對 {student_profile.name} 的預設學習路徑",
                    "description": f"此學習路徑針對 {student_profile.name} 的 {student_profile.learning_style} 學習風格和 {student_profile.current_knowledge_level} 知識水平進行量身定制",
                    "objectives": ["學習核心概念", "建立實用技能", "準備評估"],
                    "modules": [
                        {
                            "title": "模組 1: 核心概念介紹",
                            "description": "該主題領域基本概念的概述",
                            "activities": [
                                {
                                    "type": "閱讀",
                                    "title": "核心概念介紹",
                                    "description": "涵蓋基礎知識的閱讀材料",
                                    "estimated_time": "20 分鐘",
                                    "difficulty": student_profile.current_knowledge_level
                                }
                            ],
                            "resources": ["主要課程材料"],
                            "assessment": "基礎概念的簡短測驗"
                        }
                    ]
                }
    
    # Display the learning path
    console.print(f"\n[bold green]{learning_path['title']}[/bold green]")
    console.print(f"[italic]{learning_path['description']}[/italic]\n")
    
    console.print("[bold]Learning Objectives:[/bold]")
    for objective in learning_path['objectives']:
        console.print(f"? {objective}")
    
    console.print("\n[bold]Learning Modules:[/bold]")
    for i, module in enumerate(learning_path['modules']):
        console.print(f"\n[bold cyan]Module {i+1}: {module['title']}[/bold cyan]")
        console.print(f"[italic]{module['description']}[/italic]")
        
        console.print("\n[bold]Activities:[/bold]")
        for activity in module['activities']:
            console.print(f"? {activity['title']} ({activity['type']}, {activity['estimated_time']})")
            console.print(f"  {activity['description']}")
        
        if 'resources' in module:
            console.print("\n[bold]Resources:[/bold]")
            for resource in module['resources']:
                console.print(f"? {resource}")
        
        if 'assessment' in module:
            console.print(f"\n[bold]Assessment:[/bold] {module['assessment']}")
    
    return learning_path

def deliver_module_content(chat_model, retriever, student_profile, module):
    console.print(f"\n[bold cyan]===== Module: {module['title']} =====[/bold cyan]")
    console.print(f"[italic]{module['description']}[/italic]\n")
    
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    
    content_chain = create_module_content_generator(chat_model, retriever)
    content = content_chain.invoke(
        module_topic,
        student_profile.learning_style,
        student_profile.current_knowledge_level
    )
    
    console.print(Markdown(content))
    
    # 給予時間閱讀內容
    console.print("\n[yellow]請花時間閱讀並理解內容。[yellow]")
    Prompt.ask("\n準備好繼續時請按 Enter")
    
    return content

def engage_peer_discussion(chat_model, retriever, topic):
    console.print(f"\n[bold cyan]===== 同儕討論: {topic} =====[/bold cyan]")
    console.print("[yellow]與您的 AI 學習夥伴見面！提出問題或討論主題以加深您的理解。[yellow]")
    
    discussion_chain = create_peer_discussion_ai(chat_model, retriever)
    
    console.print("\n[bold green]學習夥伴:[/bold green] 嗨！我也在學習這個主題。您想討論哪些方面或有什麼問題想問？")
    
    conversation_history = []
    
    while True:
        user_message = Prompt.ask("\n[bold blue]您[/bold blue]")
        
        if user_message.lower() in ["退出", "結束", "再見", "結束討論"]:
            console.print("\n[bold green]學習夥伴:[/bold green] 很棒的討論！如果您想稍後再聊，請告訴我。")
            break
        
        response = discussion_chain.invoke(topic, user_message)
        console.print(f"\n[bold green]學習夥伴:[/bold green] {response}")
        
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": response})
        
        if len(conversation_history) >= 10:  # 限制討論長度
            console.print("\n[yellow]我們已經進行了一次很好的討論。準備好繼續嗎？[yellow]")
            if Confirm.ask("結束討論？"):
                break
    
    return conversation_history

def administer_posttest(chat_model, retriever, module, student_profile):
    console.print(f"\n[bold cyan]===== 後測: {module['title']} =====[/bold cyan]")
    console.print("[yellow]測試您對模組內容的理解...[/yellow]")
    
    # 獲取模組主題
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    
    # 根據模組內容和學生水平生成後測問題
    posttest_chain = create_posttest_generator(chat_model, retriever)
    posttest_json = posttest_chain.invoke(student_profile.current_knowledge_level, module_topic)
    
    try:
        posttest = json.loads(posttest_json)
    except json.JSONDecodeError:
        console.print("[bold red]解析後測 JSON 時出錯。使用備用方法。[bold red]")
        # 如果 JSON 被其他文字包圍，提取 JSON
        import re
        json_match = re.search(r'({[\s\S]*})', posttest_json)
        if json_match:
            try:
                posttest = json.loads(json_match.group(1))
            except:
                console.print("[bold red]無法解析後測 JSON。使用預設測驗。[bold red]")
                posttest = {
                    "title": f"後測: {module_topic}",
                    "description": f"此測驗將評估您對 {module_topic} 的學習成果",
                    "questions": [
                        {
                            "question": f"{module_topic} 的一個關鍵概念是什麼？",
                            "choices": ["A. 選項 A", "B. 選項 B", "C. 選項 C", "D. 選項 D"],
                            "correct_answer": "A. 選項 A",
                            "explanation": "根據模組內容，這是正確答案。",
                            "difficulty": "中等"
                        }
                    ]
                }
    
    console.print(f"\n[bold green]{posttest['title']}[/bold green]")
    console.print(f"[italic]{posttest['description']}[/italic]\n")
    
    # 執行測驗
    score = 0
    total_questions = len(posttest['questions'])
    results = []
    
    for i, q in enumerate(posttest['questions']):
        console.print(f"\n[bold]問題 {i+1}:[/bold] {q['question']}")
        for choice in q['choices']:
            console.print(f"  {choice}")
        
        answer = Prompt.ask("\n您的答案 (A, B, C 或 D)").upper()
        correct_letter = q['correct_answer'][0].upper()
        
        if answer == correct_letter:
            score += 1
            console.print("[bold green]正確！[/bold green]")
        else:
            console.print(f"[bold red]錯誤。正確答案是 {q['correct_answer']}[/bold red]")
        
        console.print(f"[italic]{q['explanation']}[/italic]")
        
        results.append({
            "question": q['question'],
            "student_answer": answer,
            "correct_answer": q['correct_answer'],
            "is_correct": answer == correct_letter,
            "difficulty": q['difficulty']
        })
    
    # 計算分數
    percentage = (score / total_questions) * 100
    console.print(f"\n[bold]測驗完成！您的分數：{score}/{total_questions} ({percentage:.1f}%)[/bold]")
    
    # 評估進步並根據需要調整知識水平
    previous_level = student_profile.current_knowledge_level
    
    if percentage >= 80 and previous_level != "高級":
        if previous_level == "初學者":
            new_level = "中級"
        else:
            new_level = "高級"
        console.print(f"[bold green]進步很大！您的知識水平已從 {previous_level} 提升到 {new_level}。[bold green]")
        student_profile.current_knowledge_level = new_level
    elif percentage < 50 and previous_level != "初學者":
        if previous_level == "高級":
            new_level = "中級"
        else:
            new_level = "初學者"
        console.print(f"[yellow]您可能需要更多練習。您的知識水平已從 {previous_level} 調整為 {new_level}。[yellow]")
        student_profile.current_knowledge_level = new_level
    else:
        console.print(f"[yellow]您的知識水平保持在 {previous_level}。[yellow]")
    
    # 更新學生檔案
    student_profile.learning_history.append({
        "activity_type": "後測",
        "module": module['title'],
        "timestamp": datetime.datetime.now().isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "previous_level": previous_level,
        "current_level": student_profile.current_knowledge_level
    })
    
    # 儲存更新後的檔案
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return posttest, results

def create_learning_log(chat_model, module, test_results, student_profile):
    console.print(f"\n[bold cyan]===== 學習日誌: {module['title']} =====[/bold cyan]")
    
    # 獲取模組主題和摘要
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    module_summary = module['description']
    
    # 格式化測驗結果
    test_results_str = f"分數: {sum(1 for r in test_results if r['is_correct'])}/{len(test_results)} 題正確\n"
    test_results_str += "優勢: " + ", ".join([r['question'] for r in test_results if r['is_correct']])
    test_results_str += "\n需要改進的領域: " + ", ".join([r['question'] for r in test_results if not r['is_correct']])
    
    # 生成反思提示
    learning_log_chain = create_learning_log_prompter(chat_model)
    reflection_prompts = learning_log_chain.invoke({
        "module_title": module_topic,
        "module_summary": module_summary,
        "test_results": test_results_str
    })
    
    console.print(Markdown(reflection_prompts))
    
    # 獲取學生反思
    console.print("\n[bold yellow]請根據上述提示撰寫您的學習日誌反思：[bold yellow]")
    log_content = ""
    
    while True:
        line = input()
        if line.lower() == "done":
            break
        log_content += line + "\n"
    
    # 創建並保存學習日誌
    log_id = str(uuid.uuid4())[:8]
    log = LearningLog(
        id=log_id,
        student_id=student_profile.id,
        timestamp=datetime.datetime.now().isoformat(),
        topic=module_topic,
        content=log_content,
        reflections=[],  # 將由分析填充
        questions=[],    # 將由分析填充
        next_steps=[]    # 將由分析填充
    )
    
    # 分析學習日誌
    analyzer_chain = create_learning_log_analyzer(chat_model)
    analysis_json = analyzer_chain.invoke({
        "student_name": student_profile.name,
        "topic": module_topic,
        "log_content": log_content
    })
    
    try:
        analysis = json.loads(analysis_json)
        
        # 根據分析更新學生檔案
        if "strengths" in analysis:
            for strength in analysis["strengths"]:
                if strength not in student_profile.strengths:
                    student_profile.strengths.append(strength)
        
        if "areas_for_improvement" in analysis:
            for area in analysis["areas_for_improvement"]:
                if area not in student_profile.areas_for_improvement:
                    student_profile.areas_for_improvement.append(area)
        
        if "recommended_next_steps" in analysis:
            log.next_steps = analysis["recommended_next_steps"]
        
        # 顯示分析摘要
        console.print("\n[bold green]學習日誌分析結果:[/bold green]")
        console.print(f"[bold]理解程度:[/bold] {analysis.get('understanding_level', '未確定')}")
        
        console.print("\n[bold]優勢:[/bold]")
        for strength in analysis.get("strengths", []):
            console.print(f"? {strength}")
        
        console.print("\n[bold]需要改進的領域:[/bold]")
        for area in analysis.get("areas_for_improvement", []):
            console.print(f"? {area}")
        
        console.print("\n[bold]建議的下一步行動:[/bold]")
        for step in analysis.get("recommended_next_steps", []):
            console.print(f"? {step}")
    
    except (json.JSONDecodeError, KeyError):
        console.print("[bold red]分析學習日誌時出錯。僅保存原始日誌。[bold red]")
    
    # 保存學習日誌
    with open(f"learning_logs/{log_id}.json", "w") as f:
        f.write(log.json(indent=4))
    
    # 更新並保存學生檔案
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return log

# Main application function
def main():
    console.print("[bold cyan]=====================================[/bold cyan]")
    console.print("[bold cyan]== RAG 鷹架教育系統 ==[/bold cyan]")
    console.print("[bold cyan]=====================================[/bold cyan]\n")
    
    # 初始化模型和 RAG 系統
    console.print("[yellow]正在初始化系統...[/yellow]")
    chat_model, embedding = initialize_models()
    retriever = initialize_rag_system(embedding)
    
    # 管理學生檔案
    student_profile = manage_student_profile()
    console.print(f"\n[bold green]歡迎, {student_profile.name}![/bold green]")
    
    # 如果尚未完成學習風格評估，進行評估
    if not student_profile.learning_style:
        student_profile = conduct_learning_style_survey(chat_model, student_profile)
    else:
        console.print(f"\n[yellow]您的學習風格是: [bold]{student_profile.learning_style}[/bold][/yellow]")
    
    # 進行前測
    pretest, pretest_results, knowledge_level = administer_pretest(chat_model, retriever, student_profile)
    
    # 生成個人化學習路徑
    learning_path = generate_learning_path(chat_model, retriever, student_profile, pretest_results)
    
    # 學習過程
    for module_index, module in enumerate(learning_path['modules']):
        console.print(f"\n[bold cyan]===== 開始模組 {module_index + 1}/{len(learning_path['modules'])} =====[/bold cyan]")
        
        # 詢問使用者是否準備好進行此模組
        proceed = Confirm.ask(f"準備好開始模組: {module['title']}?")
        if not proceed:
            continue
        
        # 提供模組內容
        module_content = deliver_module_content(chat_model, retriever, student_profile, module)
        
        # 進行同儕討論
        module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
        discussion_history = engage_peer_discussion(chat_model, retriever, module_topic)
        
        # 進行後測
        posttest, posttest_results = administer_posttest(chat_model, retriever, module, student_profile)
        
        # 創建學習日誌
        learning_log = create_learning_log(chat_model, module, posttest_results, student_profile)
        
        # 詢問使用者是否想繼續進行下一個模組
        if module_index < len(learning_path['modules']) - 1:
            continue_learning = Confirm.ask("是否繼續進行下一個模組?")
            if not continue_learning:
                break
    
    console.print("\n[bold green]===== 學習路徑完成 =====[/bold green]")
    console.print("[yellow]感謝您使用 RAG 鷹架教育系統！[/yellow]")
    console.print(f"[yellow]您的當前知識水平: [bold]{student_profile.current_knowledge_level}[/bold][/yellow]")
    
    # 總結學習進度
    console.print("\n[bold]您的學習旅程摘要:[/bold]")
    console.print(f"? 完成了 {len(student_profile.learning_history)} 項學習活動")
    console.print(f"? 當前優勢: {', '.join(student_profile.strengths) if student_profile.strengths else '尚未確定'}")
    console.print(f"? 需要改進的領域: {', '.join(student_profile.areas_for_improvement) if student_profile.areas_for_improvement else '尚未確定'}")
    
    console.print("\n[bold]持續學習的建議:[/bold]")
    if student_profile.current_knowledge_level == "初學者":
        console.print("? 專注於掌握基礎概念")
        console.print("? 多練習初學者到中級的範例")
    elif student_profile.current_knowledge_level == "中級":
        console.print("? 加深對複雜主題的理解")
        console.print("? 開始將概念應用於實際問題")
    else:  # 高級
        console.print("? 探索該領域的專業主題")
        console.print("? 考慮教學或指導他人以鞏固您的知識")

if __name__ == "__main__":
    main()