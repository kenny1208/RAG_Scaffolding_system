#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_chroma import Chroma
from glob import glob
import nltk
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    logger.error("Google API Key not found in environment variables")
    raise ValueError("GOOGLE_API_KEY is required. Please set it in your .env file.")

def initialize_models():
    """Initialize language models and embeddings."""
    logger.info("Initializing language models and embeddings")
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash-latest",
        temperature=0.2
    )
    
    embedding = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key, 
        model="models/embedding-001"
    )
    
    return chat_model, embedding

def initialize_rag_system(embedding, rebuild=False):
    """Initialize RAG system with document loading and chunking."""
    vectorstore_path = "vectorstore"
    data_path = "data"
    
    # Check if we already have a persisted vector store and not rebuilding
    if os.path.exists(vectorstore_path) and len(os.listdir(vectorstore_path)) > 0 and not rebuild:
        logger.info("Loading existing vector store...")
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # If not, create a new one from documents
    logger.info("Creating new vector store from documents...")
    pdf_paths = glob(os.path.join(data_path, "*.pdf"))
    
    if not pdf_paths:
        logger.warning("No PDF documents found in data directory!")
        # Create an empty vector store
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    all_pages = []
    for path in pdf_paths:
        logger.info(f"Loading document: {path}")
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Error loading PDF {path}: {str(e)}")
    
    # Try to load any text files as well
    txt_paths = glob(os.path.join(data_path, "*.txt"))
    for path in txt_paths:
        logger.info(f"Loading document: {path}")
        try:
            loader = TextLoader(path, encoding='utf-8')
            documents = loader.load()
            all_pages.extend(documents)
        except Exception as e:
            logger.error(f"Error loading text file {path}: {str(e)}")
    
    if not all_pages:
        logger.warning("No documents were successfully loaded!")
        # Create an empty vector store
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_pages)
    
    logger.info(f"Created {len(chunks)} text chunks for retrieval")
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding,
        persist_directory=vectorstore_path
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def generate_test(chat_model, retriever, is_pretest=True, knowledge_level="初學者"):
    """Generate a test using the RAG system."""
    logger.info(f"Generating {'pretest' if is_pretest else 'posttest'} for knowledge level: {knowledge_level}")
    
    test_type = "前測" if is_pretest else "後測"
    
    # Fix: Split the string to avoid f-string with backslash issue
    if is_pretest:
        difficulty_text = "請設計包含不同難度等級的問題：簡單、中等和困難。"
        level_text = ""
        final_instruction = "請根據提供的內容生成總共 5 個問題，且包含不同難度等級的問題。"
    else:
        difficulty_text = ""
        level_text = """難度應與學生的當前水平相匹配：
        - 初學者：較多簡單問題（70%），一些中等問題（30%）
        - 中級者：一些簡單問題（30%），主要是中等問題（50%），一些困難問題（20%）
        - 高級者：一些中等問題（40%），主要是困難問題（60%）"""
        final_instruction = f"學生的當前知識水平: {knowledge_level}\n根據學生的水平生成總共 5 個問題，並適當分配難度。"
    
    purpose_text = "評估學生在該主題上的現有知識水平" if is_pretest else "評估學生的學習成果"
    description_text = "對[主題]的現有知識" if is_pretest else "對[主題]的學習成果"
    
    prompt = f"""你是一位專精於教學評估設計的專家。
    根據提供的內容，設計一份{test_type}，以{purpose_text}。
    
    {difficulty_text}
    {level_text}
    
    對於每個問題，請提供：
    1. 問題文本
    2. 四個多選選項（A, B, C, D）
    3. 正確答案
    4. 為什麼正確的解釋
    5. 難度等級
    
    你必須遵循以下嚴格的 JSON 格式：
    {{
      "title": "{test_type}：[主題]",
      "description": "此測驗將評估你{description_text}",
      "questions": [
        {{
          "question": "問題文本？",
          "choices": ["A. 選項 A", "B. 選項 B", "C. 選項 C", "D. 選項 D"],
          "correct_answer": "A. 選項 A",
          "explanation": "為什麼 A 是正確答案的解釋",
          "difficulty": "簡單"
        }}
      ]
    }}

    {final_instruction}
    """
    
    # Get some documents from the retriever
    documents = retriever.get_relevant_documents(test_type)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    messages = [{"role": "user", "content": f"{prompt}\n\n內容：\n{context}"}]
    response = chat_model.invoke(messages)
    
    return response.content

def generate_learning_path(chat_model, retriever, profile_data, test_results):
    """Generate a personalized learning path."""
    logger.info(f"Generating learning path for {profile_data.get('name', 'unknown')}")
    
    prompt = f"""你是一位專精於個人化學習路徑設計的教育課程設計專家。
    根據學生檔案和測驗結果，創建適合學生的學習路徑。
    
    學生資料：
    {profile_data}
    
    測驗結果：
    {test_results}
    
    你的學習路徑應該：
    1. 針對學生的學習風格和當前知識水平進行量身定制
    2. 包含清晰的學習目標
    3. 遵循漸進原則，逐步增加難度並減少支持
    
    你的回應必須遵循以下嚴格的 JSON 格式：
    {{
      "title": "針對[主題]的個人化學習路徑",
      "description": "此學習路徑針對[name]的學習風格和當前知識水平進行量身定制",
      "objectives": ["目標 1", "目標 2", "目標 3"],
      "modules": [
        {{
          "title": "模組 1: [標題]",
          "description": "模組描述",
          "activities": [
            {{
              "type": "活動",
              "title": "活動標題",
              "description": "活動描述",
              "difficulty": "初學者"
            }}
          ],
          "resources": ["資源模組1-1", "資源模組1-2"],
        }}
      ]
    }}
    """
    
    # Get relevant documents
    documents = retriever.get_relevant_documents("學習路徑")
    context = "\n\n".join([doc.page_content for doc in documents])
    
    messages = [{"role": "user", "content": f"{prompt}\n\n內容：\n{context}"}]
    response = chat_model.invoke(messages)
    
    return response.content

def generate_module_content(chat_model, retriever, module_topic, learning_style, knowledge_level):
    """Generate personalized content for a learning module."""
    logger.info(f"Generating content for module: {module_topic}")
    
    prompt = f"""你是一位專業的教學內容創作者。
    根據提供的模組主題以及學生的學習風格和知識水平，創建引人入勝的教學內容。
    
    你的內容應該：
    1. 針對學生的學習風格（{learning_style}）進行量身定制
    2. 符合學生的知識水平（{knowledge_level}）
    3. 包含關鍵概念的清晰解釋
    4. 使用範例和比喻來說明要點
    5. 包含符合知識水平的練習活動
    6. 結構清晰，包含明確的段落和標題
    7. 使用 markdown 格式提高可讀性
    
    模組主題: {module_topic}
    """
    
    # Get relevant documents
    documents = retriever.get_relevant_documents(module_topic)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    messages = [{"role": "user", "content": f"{prompt}\n\n相關內容：\n{context}"}]
    response = chat_model.invoke(messages)
    
    return response.content

def analyze_learning_log(chat_model, log_data):
    """Analyze a learning log to provide insights and feedback."""
    logger.info(f"Analyzing learning log: {log_data.get('id', 'unknown')}")
    
    prompt = f"""你是一位專業的教學分析師，精於分析學生的學習日誌。
    根據學生的學習日誌，評估：
    
    1. 對關鍵概念的理解程度
    2. 優點和自信的領域
    3. 困惑或誤解的領域
    4. 對情感反應的感知
    5. 學習風格的指示
    
    將你的回應格式化為以下嚴格的 JSON 結構:
    {
      "understanding_level": "高/中/低",
      "strengths": ["優點 1", "優點 2"],
      "areas_for_improvement": ["改進領域 1", "改進領域 2"],
      "emotional_response": "對情感反應的描述",
      "learning_style_indicators": ["指示 1", "指示 2"],
      "recommended_next_steps": ["建議步驟 1", "建議步驟 2"],
      "suggested_resources": ["資源 1", "資源 2"]
    }
    """
    
    messages = [{"role": "user", "content": f"{prompt}\n\n學生: {log_data.get('student_id', 'unknown')}\n主題: {log_data['topic']}\n學習日誌內容:\n{log_data['content']}"}]
    response = chat_model.invoke(messages)
    
    return response.content

def generate_peer_discussion(chat_model, retriever, topic, message, learning_style, knowledge_level):
    """Generate responses for the peer discussion feature."""
    logger.info(f"Generating peer discussion response for topic: {topic}")
    
    prompt = f"""你是「學習夥伴」，一個親切且有啟發性的 AI 同學，與學生進行有建設性的討論。
    你的角色是：
    1. 模擬一位也在學習該主題但有一定見解的同學
    2. 提出促進深度判斷性思考的開放問題
    3. 提供啟發性的指導，而不是直接給出答案
    4. 以對話的形式表達思考，就像是學生之間的交流
    5. 使用蘇格拉底式提問法幫助學生發現答案
    6. 鼓勵並保持良好的氛圍
    
    根據提供的相關內容回應，但不要只是簡單地重述資訊。
    而是以自然的方式進行交流，就像一起學習一樣。
    
    討論主題: {topic}
    學生學習風格: {learning_style}
    學生知識水平: {knowledge_level}
    學生訊息: {message}
    """
    
    # Get relevant documents based on topic and message
    combined_query = f"{topic} {message}"
    documents = retriever.get_relevant_documents(combined_query)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    messages = [{"role": "user", "content": f"{prompt}\n\n相關內容：\n{context}"}]
    response = chat_model.invoke(messages)
    
    return response.content

def generate_learning_style_survey(chat_model):
    """Generate a learning style survey."""
    logger.info("Generating learning style survey")
    
    prompt = """你是一位專精於學習風格評估的教育專家。
    請設計一份簡短但有效的學習風格測驗問卷，包含 5 個多選題。
    每個問題應有 3 個選項，用於判定學生是否主要是：
    1. 視覺型學習者
    2. 聽覺型學習者
    3. 動覺型學習者
    
    請將你的回應格式化為一個JSON對象，包含問題數組，每個問題有文本和選項。例如：
    {
      "questions": [
        {
          "text": "當你學習新知識時，你偏好：",
          "options": [
            {"text": "看圖表和視覺輔助", "value": "視覺型"},
            {"text": "聆聽講解和討論", "value": "聽覺型"},
            {"text": "動手實作和體驗", "value": "動覺型"}
          ]
        }
      ]
    }
    
    請確保每個問題都有針對三種不同學習風格的選項。"""
    
    messages = [{"role": "user", "content": prompt}]
    response = chat_model.invoke(messages)
    
    return response.content