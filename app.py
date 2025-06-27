from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import json
import datetime
import random
import re
from flask import render_template, request
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional, Any
import nltk

import time
import shutil

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['VECTOR_DB_DIR'] = 'vectordbs'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['STUDENT_PROFILES_DIR'] = 'student_profiles'
app.config['COURSE_PROFILES_DIR'] = 'course_profiles'
app.config['LEARNING_LOGS_DIR'] = 'learning_logs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_DB_DIR'], exist_ok=True)
os.makedirs(app.config['STUDENT_PROFILES_DIR'], exist_ok=True)
os.makedirs(app.config['COURSE_PROFILES_DIR'], exist_ok=True)
os.makedirs(app.config['LEARNING_LOGS_DIR'], exist_ok=True)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Download NLTK data
nltk.download('punkt')

# Pydantic models for structured data
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
    felder_silverman_profile: Optional[Dict[str, Any]] = Field(default=None, description="Detailed Felder-Silverman learning style profile")
    courses: List[str] = Field(default_factory=list, description="List of course IDs the student is enrolled in")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), description="Profile creation timestamp")
    current_knowledge_level: str = Field(description="Beginner, intermediate, or advanced")
    strengths: List[str] = Field(description="Academic strengths")
    areas_for_improvement: List[str] = Field(description="Areas that need improvement")
    interests: List[str] = Field(description="Academic interests")
    learning_history: List[Dict[str, Any]] = Field(description="History of learning activities")
    learning_path: Optional[Dict[str, Any]] = Field(default=None, description="Personalized learning path")
    current_module_index: Optional[int] = Field(default=0, description="Current module index")
    learning_path_confirmed: Optional[bool] = Field(default=False, description="Whether the learning path has been confirmed")

class CourseProfile(BaseModel):
    id: str = Field(description="Unique course ID")
    student_id: str = Field(description="Student ID this course belongs to")
    title: str = Field(description="Course title")
    description: str = Field(description="Course description")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), description="Course creation timestamp")
    current_knowledge_level: str = Field(description="Beginner, intermediate, or advanced")
    strengths: List[str] = Field(default_factory=list, description="Academic strengths")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas that need improvement")
    interests: List[str] = Field(default_factory=list, description="Academic interests")
    learning_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of learning activities")
    learning_path: Optional[Dict[str, Any]] = Field(default=None, description="Personalized learning path")
    current_module_index: Optional[int] = Field(default=0, description="Current module index")
    learning_path_confirmed: Optional[bool] = Field(default=False, description="Whether the learning path has been confirmed")
    session_id: Optional[str] = Field(default=None, description="Current session ID for this course")
    learning_completed: Optional[bool] = Field(default=False, description="Whether the course has been completed")
    completion_date: Optional[str] = Field(default=None, description="Course completion timestamp")

class LearningLog(BaseModel):
    id: str = Field(description="Unique log ID")
    student_id: str = Field(description="Student ID")
    timestamp: str = Field(description="ISO format timestamp")
    topic: str = Field(description="Topic studied")
    content: str = Field(description="Log content")
    reflections: List[str] = Field(description="Student reflections")
    questions: List[str] = Field(description="Questions raised by student")
    next_steps: List[str] = Field(description="Planned next steps")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_pdfs(file_paths, session_id):
    """Process PDF files and return chunks and a summary"""
    try:
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Load PDFs
        all_pages = []
        for path in file_paths:
            try:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"PDF file not found: {path}")
                
                loader = PyPDFLoader(path)
                pages = loader.load_and_split()
                if not pages:
                    raise ValueError(f"No content could be extracted from PDF: {path}")
                
                # Add metadata to each page
                filename = os.path.basename(path)
                for i, page in enumerate(pages):
                    page.metadata.update({
                        'source': filename,
                        'page': i + 1  # Page numbers start from 1
                    })
                
                all_pages.extend(pages)
                app.logger.info(f"Successfully loaded PDF: {path}")
            except Exception as e:
                app.logger.error(f'Error loading PDF {path}: {str(e)}')
                raise Exception(f'Error loading PDF {os.path.basename(path)}: {str(e)}')
        
        if not all_pages:
            raise Exception('No content could be extracted from the PDFs')
        
        # Split text into chunks
        try:
            text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(all_pages)
            app.logger.info(f"Successfully split text into {len(chunks)} chunks")
        except Exception as e:
            app.logger.error(f'Error splitting text: {str(e)}')
            raise Exception(f'Error processing text content: {str(e)}')
        
        # Create embeddings and vectorstore
        try:
            # Ensure vector DB directory exists
            persist_directory = os.path.join(app.config['VECTOR_DB_DIR'], session_id)
            if os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize embedding model with retry mechanism
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    embedding = GoogleGenerativeAIEmbeddings(
                        google_api_key=api_key,
                        model="models/embedding-001"
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise Exception(f"Failed to initialize embedding model after {max_retries} attempts: {str(e)}")
                    app.logger.warning(f"Retry {retry_count} initializing embedding model")
                    time.sleep(1)
            
            # Create vector store with smaller batch size
            batch_size = 50
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                try:
                    if i == 0:
                        # First batch - create new collection
                        vectorstore = Chroma.from_documents(
                            documents=batch_chunks,
                            embedding=embedding,
                            persist_directory=persist_directory,
                            collection_name=session_id
                        )
                    else:
                        # Subsequent batches - add to existing collection
                        vectorstore.add_documents(batch_chunks)
                    
                    app.logger.info(f"Processed chunks {i+1} to {min(i+batch_size, total_chunks)} of {total_chunks}")
                except Exception as e:
                    app.logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                    raise Exception(f"Error creating vector store: {str(e)}")
            
            # Save to disk
            vectorstore.persist()
            app.logger.info(f"Successfully created vector store in {persist_directory}")
            
        except Exception as e:
            app.logger.error(f'Error creating vector store: {str(e)}')
            raise Exception(f'Error creating search index: {str(e)}')
        
        return chunks
        
    except Exception as e:
        app.logger.error(f'Error in process_pdfs: {str(e)}')
        raise

def get_vectorstore(session_id):
    """Retrieve the vectorstore for the given session ID"""
    persist_directory = os.path.join(app.config['VECTOR_DB_DIR'], session_id)
    embedding = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name=session_id
    )
    return vectorstore


# Helper functions for profile management
def get_student_profile(student_id):
    """Get student profile from disk"""
    profile_path = os.path.join(app.config['STUDENT_PROFILES_DIR'], f"{student_id}.json")
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as f:
            return StudentProfile.model_validate(json.load(f))
    return None

def save_student_profile(profile):
    """Save student profile to disk"""
    profile_path = os.path.join(app.config['STUDENT_PROFILES_DIR'], f"{profile.id}.json")
    with open(profile_path, 'w', encoding='utf-8') as f:
        f.write(profile.model_dump_json(indent=4))

def get_course_profile(course_id):
    """Get course profile from disk"""
    profile_path = os.path.join(app.config['COURSE_PROFILES_DIR'], f"{course_id}.json")
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as f:
            return CourseProfile.model_validate(json.load(f))
    return None

def save_course_profile(profile):
    """Save course profile to disk"""
    profile_path = os.path.join(app.config['COURSE_PROFILES_DIR'], f"{profile.id}.json")
    with open(profile_path, 'w', encoding='utf-8') as f:
        f.write(profile.model_dump_json(indent=4))

def create_or_get_student_profile(name=None):
    """Create a new student profile or retrieve an existing one"""
    if 'student_id' in session:
        profile = get_student_profile(session['student_id'])
        if profile:
            return profile
    
    # Create new profile
    student_id = str(uuid.uuid4())[:8]
    name = name or f"Student_{student_id}"
    
    profile = StudentProfile(
        id=student_id,
        name=name,
        learning_style="",
        courses=[],
        created_at=datetime.datetime.now().isoformat()
    )
    
    # Save the profile
    session['student_id'] = student_id
    save_student_profile(profile)
    
    return profile

def create_new_course(student_id, title, description):
    """Create a new course for a student"""
    course_id = str(uuid.uuid4())[:8]
    
    course = CourseProfile(
        id=course_id,
        student_id=student_id,
        title=title,
        description=description,
        created_at=datetime.datetime.now().isoformat(),
        current_knowledge_level="",
        learning_path=None,
        current_module_index=0,
        learning_path_confirmed=False
    )
    
    # Save course profile
    save_course_profile(course)
    
    # Update student profile with new course
    student_profile = get_student_profile(student_id)
    if student_profile:
        student_profile.courses.append(course_id)
        save_student_profile(student_profile)
    
    return course

def process_learning_style_answers(survey, answers):
    """Process learning style survey answers and determine the dominant style"""
    styles = {"visual": 0, "auditory": 0, "kinesthetic": 0}
    
    for i, answer in enumerate(answers):
        question = survey["questions"][i]
        answer_index = ord(answer.upper()) - ord('A')
        
        if answer_index == question["visual_index"]:
            styles["visual"] += 1
        elif answer_index == question["auditory_index"]:
            styles["auditory"] += 1
        elif answer_index == question["kinesthetic_index"]:
            styles["kinesthetic"] += 1
    
    dominant_style = max(styles, key=styles.get)
    return dominant_style

def create_pretest(session_id, topic=""):
    """Generate a pretest based on the uploaded documents，並且嚴格遵守4選1單選的題型"""
    vectorstore = get_vectorstore(session_id)
    
    # 使用兩種檢索策略
    # 1. 相關性檢索
    relevance_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 20  # 獲取相關的內容
        }
    )
    
    # 2. 全面檢索 - 獲取所有文檔
    all_docs_data = vectorstore.get()
    all_docs = []
    for i, doc in enumerate(all_docs_data["documents"]):
        metadata = all_docs_data["metadatas"][i] if i < len(all_docs_data["metadatas"]) else {}
        all_docs.append({
            "page_content": doc,
            "metadata": metadata
        })
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-lite"
    )
    
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

        請根據提供的內容生成總共 10 個問題，並確保：
        1. 問題要涵蓋教材中的所有重要概念和主題
        2. 包含不同難度級別的問題：
           - 4 題簡單題目（基礎概念理解）
           - 4 題中等難度（概念應用）
           - 2 題困難題目（進階分析和整合）
        3. 每個問題都要有明確的學習目標
        4. 確保問題的選項都是合理的，且只有一個正確答案
        5. 確保問題涵蓋所有提供的教材內容，不要遺漏任何重要概念
        """),
        HumanMessagePromptTemplate.from_template("""根據以下內容生成一份前測：
        
        相關內容：
        {relevant_context}
        
        完整教材：
        {all_context}
        """)
    ])
    
    # Create the chain
    pretest_chain = (
        RunnablePassthrough()
        | (lambda _: {
            "relevant_context": "\n\n".join([f"來源: {doc.metadata.get('source', 'unknown')}, 頁碼: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}" for doc in relevance_retriever.invoke(topic)]),
            "all_context": "\n\n".join([f"來源: {doc['metadata'].get('source', 'unknown')}, 頁碼: {doc['metadata'].get('page', 'unknown')}\n{doc['page_content']}" for doc in all_docs])
        })
        | prompt
        | chat_model
        | JsonOutputParser()
    )
    
    return pretest_chain.invoke(topic)

def evaluate_knowledge_level(score_percentage):
    """Determine knowledge level based on test score percentage"""
    if score_percentage >= 80:
        return "advanced"
    elif score_percentage >= 50:
        return "intermediate"
    else:
        return "beginner"

def generate_learning_path(session_id, profile, test_results):
    """Generate a personalized learning path based on profile and test results"""
    vectorstore = get_vectorstore(session_id)
    
    # 使用兩種檢索策略
    # 1. 相關性檢索
    relevance_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 20
        }
    )
    
    # 2. 全面檢索 - 獲取所有文檔
    all_docs_data = vectorstore.get()
    all_docs = []
    for i, doc in enumerate(all_docs_data["documents"]):
        metadata = all_docs_data["metadatas"][i] if i < len(all_docs_data["metadatas"]) else {}
        all_docs.append({
            "page_content": doc,
            "metadata": metadata
        })
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-lite"
    )
    
    # Get student requirements if they exist
    student_requirements = test_results.get('student_requirements', '')
    requirements_prompt = f"\nStudent's specific requirements: {student_requirements}" if student_requirements else ""
    
    # Get student profile if we have a course profile
    student_profile = None
    if isinstance(profile, CourseProfile):
        student_profile = get_student_profile(profile.student_id)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""您是一位專精於個人化學習路徑設計的教育課程設計專家。
        根據提供的學生檔案、測驗結果和內容，創建一條適合他自學的學習路徑。
        
        重要：您必須先分析教材內容，建立清晰的課程結構，然後為每個章節分配特定的內容範圍。
        
        您的學習路徑設計流程：
        1. 首先分析教材內容，識別主要主題和概念
        2. 建立邏輯性的章節結構，確保知識的遞進關係
        3. 為每個章節分配特定的內容範圍，避免重複
        4. 確保章節之間有清晰的連接和過渡
        
        您的學習路徑必須：
        1. 建立清晰的章節分工，每個章節專注於特定主題
        2. 確保章節之間有邏輯順序和知識遞進
        3. 針對學生的學習風格、知識水平進行量身調整
        4. 遵循鷹架原則，逐步增加難度並減少支持
        5. Take into account any specific requirements or preferences expressed by the student{requirements_prompt}
        6. 根據學生的知識水平調整內容深度和廣度：
           - 初學者：更詳細的基礎概念解釋，更多例子
           - 中級者：平衡的理論和應用，適中的難度
           - 進階者：更深入的理論探討，更多分析和整合的內容
        
        您的回應必須遵循這個精確的 JSON 格式：
        {{
          "title": "針對[主題]的個人化學習路徑",
          "description": "此學習路徑針對[name]的學習風格和當前知識水平進行量身定制",
          "objectives": ["目標 1", "目標 2", "目標 3"],
          "modules": [
            {{
              "title": "章節 1: [標題]",
              "description": "章節描述",
              "content_scope": "此章節專注的具體內容範圍",
              "prerequisites": ["前置知識 1", "前置知識 2"],
              "learning_outcomes": ["學習成果 1", "學習成果 2"],
              "activities": [
                {{
                  "content": "列出該章節要學會的內容",
                  "source": "資料來源"
                }}
              ],
              "module_index": 0
            }}
          ]
        }}
        
        請確保：
        1. 每個章節的content_scope明確且不重複
        2. 章節之間有清晰的prerequisites關係
        3. 學習成果逐步遞進
        4. 總章節數控制在3-5個，避免過於分散
        """),
        HumanMessagePromptTemplate.from_template("""根據以下內容生成個人化學習路徑：
        
        學生檔案：
        {profile}
        
        測驗結果：
        {test_results}
        
        相關內容：
        {relevant_context}
        
        完整教材：
        {all_context}
        """)
    ])
    
    # Format profile and test results
    profile_json = json.dumps({
        "name": student_profile.name if student_profile else "Student",
        "learning_style": student_profile.learning_style if student_profile else profile.learning_style,
        "current_knowledge_level": profile.current_knowledge_level,
        "interests": student_profile.interests if student_profile else []
    })
    
    # Create the chain
    learning_path_chain = (
        RunnablePassthrough()
        | (lambda _: {
            "profile": profile_json,
            "test_results": json.dumps(test_results),
            "relevant_context": "\n\n".join([f"來源: {doc.metadata.get('source', 'unknown')}, 頁碼: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}" for doc in relevance_retriever.invoke("")]),
            "all_context": "\n\n".join([f"來源: {doc['metadata'].get('source', 'unknown')}, 頁碼: {doc['metadata'].get('page', 'unknown')}\n{doc['page_content']}" for doc in all_docs])
        })
        | prompt
        | chat_model
        | JsonOutputParser()
    )
    
    return learning_path_chain.invoke("")

def generate_module_content(session_id, module, profile):
    """Generate educational content for a module"""
    vectorstore = get_vectorstore(session_id)
    
    # 使用兩種檢索策略
    # 1. 相關性檢索 - 針對章節特定內容
    relevance_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 15
        }
    )
    
    # 2. 全面檢索 - 獲取所有文檔（用於上下文理解）
    all_docs_data = vectorstore.get()
    all_docs = []
    for i, doc in enumerate(all_docs_data["documents"]):
        metadata = all_docs_data["metadatas"][i] if i < len(all_docs_data["metadatas"]) else {}
        all_docs.append({
            "page_content": doc,
            "metadata": metadata
        })
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-lite"
    )
    
    # Get module topic and content scope
    module_topic = module["title"].split(": ", 1)[1] if ": " in module["title"] else module["title"]
    content_scope = module.get("content_scope", module_topic)
    prerequisites = module.get("prerequisites", [])
    learning_outcomes = module.get("learning_outcomes", [])
    module_index = module.get("module_index", 0)
    
    # Check if this is a retry
    is_retry = module.get('retry_count', 0) > 0
    retry_count = module.get('retry_count', 0)
    
    # Determine scaffolding level based on multiple factors
    knowledge_level = profile.current_knowledge_level
    emotional_state = module.get('emotional_analysis', {})
    base_scaffolding_level = module.get('scaffolding_level', 'medium')
    
    # Adjust scaffolding level based on emotional state
    if emotional_state:
        if emotional_state.get('frustration_level') == 'high' or emotional_state.get('confidence_level') == 'low':
            base_scaffolding_level = 'high'
        elif emotional_state.get('engagement_level') == 'high' and emotional_state.get('confidence_level') == 'high':
            base_scaffolding_level = 'low'
    
    # Further adjust based on knowledge level and retry count
    if knowledge_level == "advanced" and base_scaffolding_level != 'high':
        scaffolding_level = "low"
    elif knowledge_level == "intermediate":
        if retry_count == 0 and base_scaffolding_level != 'high':
            scaffolding_level = "medium"
        else:
            scaffolding_level = "high"
    else:
        scaffolding_level = "high"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""您是一位專業的教育內容創作者，專精於鷹架學習理論。
        根據章節的特定內容範圍和學生的學習風格、知識水平，創造引人入勝的教育內容。
        
        重要：您必須專注於此章節的特定內容範圍，不要重複涵蓋其他章節的內容。
        
        您的內容應該：
        1. 專注於章節的特定內容範圍：{content_scope}
        2. 針對學生的學習風格進行量身定制(盡量不要使用圖片，如果需要圖例，請使用markdown或者mermaid格式化)
        3. 適合學生的知識水平
        4. 包含清晰的關鍵概念解釋
        5. 使用例子和比喻來闡明觀點
        6. 根據鷹架支持程度 ({scaffolding_level}) 調整內容：
           - 高鷹架支持：提供詳細的步驟說明、更多例子、提示和引導性問題
           - 中鷹架支持：提供適中的解釋和例子，加入一些思考問題
           - 低鷹架支持：提供基本概念，鼓勵自主探索和思考
        
        7. 結構清晰，包含以下部分：
           - 學習目標（基於learning_outcomes）
           - 前置知識提醒（基於prerequisites）
           - 主要內容（專注於content_scope）
           - 關鍵概念總結
           - 自我檢查問題
           - 延伸思考問題
           - 與下一章節的連接提示
        
        8. 每個章節都要根據context的內容標註來源(source)，格式為：
           [來源: 檔名.pdf, 頁碼: X]
           如果有多個來源，只列出第一個。
        
        9. 加入互動元素：
           - 思考問題
           - 反思提示
        
        10. 確保內容與章節的特定範圍相關，避免重複其他章節的內容。
        
        使用markdown格式化您的內容，以提高可讀性。
        """),
        HumanMessagePromptTemplate.from_template("""為以下內容創建教育內容：
        
        章節主題：{module_topic}
        章節內容範圍：{content_scope}
        前置知識：{prerequisites}
        學習成果：{learning_outcomes}
        章節順序：{module_index}
        學生學習風格：{learning_style}
        學生知識水平：{knowledge_level}
        鷹架支持程度：{scaffolding_level}
        
        相關內容（專注於此章節範圍）：
        {relevant_context}
        
        完整教材（用於上下文理解）：
        {all_context}
        """)
    ])
    
    # Create the chain
    content_chain = (
        RunnablePassthrough()
        | (lambda _: {
            "module_topic": module_topic,
            "content_scope": content_scope,
            "prerequisites": ", ".join(prerequisites) if prerequisites else "無",
            "learning_outcomes": ", ".join(learning_outcomes) if learning_outcomes else "理解本章節內容",
            "module_index": module_index,
            "learning_style": profile.learning_style,
            "knowledge_level": profile.current_knowledge_level,
            "scaffolding_level": scaffolding_level,
            "relevant_context": "\n\n".join([f"來源: {doc.metadata.get('source', 'unknown')}, 頁碼: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}" for doc in relevance_retriever.invoke(content_scope)]),
            "all_context": "\n\n".join([f"來源: {doc['metadata'].get('source', 'unknown')}, 頁碼: {doc['metadata'].get('page', 'unknown')}\n{doc['page_content']}" for doc in all_docs])
        })
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return content_chain.invoke("")

def generate_module_transition(session_id, current_module, next_module, profile):
    """Generate transition content between modules"""
    if not next_module:
        return ""
    
    vectorstore = get_vectorstore(session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-lite"
    )
    
    current_topic = current_module["title"].split(": ", 1)[1] if ": " in current_module["title"] else current_module["title"]
    next_topic = next_module["title"].split(": ", 1)[1] if ": " in next_module["title"] else next_module["title"]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育內容設計師，專精於創建章節之間的過渡內容。
        您的任務是創建一個簡短但有效的過渡段落，幫助學生從當前章節順利過渡到下一個章節。
        
        過渡內容應該：
        1. 簡短總結當前章節的關鍵概念
        2. 預告下一個章節的主要內容
        3. 建立兩個章節之間的邏輯連接
        4. 激發學生對下一個章節的興趣
        5. 提供學習建議和準備提示
        
        使用markdown格式化，保持簡潔明瞭。
        """),
        HumanMessagePromptTemplate.from_template("""創建章節過渡內容：
        
        當前章節：{current_topic}
        下一個章節：{next_topic}
        
        相關內容：
        {context}
        """)
    ])
    
    transition_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "current_topic": current_topic,
            "next_topic": next_topic,
            "context": "\n\n".join([d.page_content for d in docs])
        })
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return transition_chain.invoke(f"{current_topic} {next_topic}")

def simulate_peer_discussion(session_id, topic, message):
    """Simulate a peer discussion with an AI learning partner"""
    vectorstore = get_vectorstore(session_id)
    
    # 使用針對特定主題的檢索
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-lite"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是「學習夥伴」，一個友善且有幫助的 AI 同儕，與學生進行有建設性的討論。
        您的角色是：
        1. 模擬一位也在學習該主題但有一定見解的同儕
        2. 提出有助於促進批判性思考的問題
        3. 專注於討論的主題範圍，不要偏離主題
        4. 以對話的方式表達想法，像是學生之間的交流
        5. 鼓勵並保持積極的態度
        
        根據提供的相關內容回應，但不要只是簡單地背誦資訊。
        而是以自然的方式進行來回討論，就像一起學習一樣，但是還是要能回答同學的問題而不是一直反問。
        確保討論內容與主題相關，避免偏離到其他章節的內容。
        """),
        HumanMessagePromptTemplate.from_template("""學生想要討論這個主題：
        
        主題：{topic}
        
        相關內容：
        {context}
        
        學生訊息：
        {message}
        """)
    ])
    
    # Create the chain
    discussion_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "topic": topic,
            "message": message,
            "context": "\n\n".join([d.page_content for d in docs])
        })
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return discussion_chain.invoke(topic)

def create_posttest(session_id, module, profile):
    """Generate a post-test for a module"""
    vectorstore = get_vectorstore(session_id)
    
    # 使用針對章節特定內容的檢索
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-lite"
    )
    
    # Get module topic and content scope
    module_topic = module["title"].split(": ", 1)[1] if ": " in module["title"] else module["title"]
    content_scope = module.get("content_scope", module_topic)
    learning_outcomes = module.get("learning_outcomes", [])
    
    # Get knowledge level from profile
    knowledge_level = profile.current_knowledge_level
    
    # Check if this is a retry with easier test
    is_easier_test = module.get('easier_test', False)
    retry_count = module.get('retry_count', 0)
    
    # Determine scaffolding level for test
    scaffolding_level = "high"  # Default to high scaffolding
    if knowledge_level == "advanced":
        scaffolding_level = "low"
    elif knowledge_level == "intermediate":
        if retry_count == 0:
            scaffolding_level = "medium"
        else:
            scaffolding_level = "high"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""您是一位專業的教育評估設計師。
        根據章節的特定內容範圍和學生的當前知識水平，設計一份後測，包含多選題，以評估學生的學習成果。
        
        重要：您必須專注於此章節的特定內容範圍：{content_scope}
        不要測試其他章節的內容，確保題目與章節的學習成果相關。
        
        難度應該符合學生的當前水平：
        - 初學者：更多容易的問題 (70%)，一些中等難度 (30%)
        - 中級者：一些容易的問題 (30%)，大多數是中等難度 (50%)，一些難題 (20%)
        - 進階者：大多數是中等難度 (40%)，大多數是難題 (60%)
        
        {'如果這是重新學習的測驗，請將所有問題的難度降低一級，並增加更多基礎概念的問題。' if is_easier_test else ''}
        
        根據鷹架支持程度 ({scaffolding_level}) 調整測驗：
        - 高鷹架支持：
          * 提供更多提示和引導
          * 增加基礎概念問題
          * 加入部分答案選項的解釋
        - 中鷹架支持：
          * 適中的提示
          * 平衡的難度分布
        - 低鷹架支持：
          * 較少的提示
          * 更多應用和分析問題
        
        題目應該測試學生對章節特定內容的理解、應用和分析能力。
        
        每個問題應該包含：
        1. 問題文字
        2. 四個多選題選項 (A, B, C, D)
        3. 正確答案
        4. 為什麼它是正確的解釋
        5. 難度等級
        6. 相關提示（根據鷹架支持程度）
        
        您必須遵循這個精確的 JSON 格式：
        {{
          "title": "後測: [主題]",
          "description": "這個測驗將評估您對 [主題] 的學習成果",
          "questions": [
            {{
              "question": "問題文字?",
              "choices": ["A. 選項 A", "B. 選項 B", "C. 選項 C", "D. 選項 D"],
              "correct_answer": "A. 選項 A",
              "explanation": "為什麼 A 是正確的解釋",
              "difficulty": "easy",
              "hints": ["提示1", "提示2"]
            }}
          ]
        }}

        根據學生的水平生成總共 5 個問題，並且適當的難度分布。
        確保所有問題都與章節的特定內容範圍相關。
        """),
        HumanMessagePromptTemplate.from_template("""Generate a post-test based on:
        
        章節主題：{module_topic}
        章節內容範圍：{content_scope}
        學習成果：{learning_outcomes}
        學生的當前知識水平：{knowledge_level}
        鷹架支持程度：{scaffolding_level}
        章節相關內容：{context}
        """)
    ])
    
    # Create the chain
    posttest_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "module_topic": module_topic,
            "content_scope": content_scope,
            "learning_outcomes": ", ".join(learning_outcomes) if learning_outcomes else "理解本章節內容",
            "knowledge_level": knowledge_level,
            "scaffolding_level": scaffolding_level,
            "context": "\n\n".join([d.page_content for d in docs])
        })
        | prompt
        | chat_model
        | JsonOutputParser()
    )
    
    return posttest_chain.invoke(content_scope)

def analyze_learning_log(student_name, topic, log_content):
    """Analyze a student's learning log"""
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-lite"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育分析師，專精於分析學生的學習日誌。
        根據學生的學習日誌，評估：
        
        1. 對於關鍵概念的理解程度
        2. 學生的強項和信心
        3. 學生可能的困惑或錯誤理解
        4. 學生對材料的情感反應和情緒狀態
        5. 學習風格的指標
        6. 建議的鷹架支持等級（高/中/低）
        
        您的回應必須遵循以下精確的 JSON 結構：
        {
          "understanding_level": "high/medium/low",
          "strengths": ["強項 1", "強項 2"],
          "areas_for_improvement": ["需要改進的領域 1", "需要改進的領域 2"],
          "emotional_state": {
            "primary_emotion": "主要情緒",
            "confidence_level": "high/medium/low",
            "frustration_level": "high/medium/low",
            "engagement_level": "high/medium/low"
          },
          "learning_style_indicators": ["學習風格指標 1", "學習風格指標 2"],
          "recommended_next_steps": ["建議的下一步 1", "建議的下一步 2"],
          "suggested_resources": ["資源 1", "資源 2"],
          "recommended_scaffolding_level": "high/medium/low"
        }
        """),
        HumanMessagePromptTemplate.from_template("""分析以下學習日誌：
        
        學生：{student_name}
        主題：{topic}
        學習日誌內容：
        {log_content}
        """)
    ])
    
    analyzer_chain = prompt | chat_model | JsonOutputParser()
    
    return analyzer_chain.invoke({
        "student_name": student_name,
        "topic": topic,
        "log_content": log_content
    })

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_user')
def select_user():
    # Get list of existing profiles
    profiles = []
    profile_dir = 'student_profiles'
    if os.path.exists(profile_dir):
        for file in os.listdir(profile_dir):
            if file.endswith('.json'):
                with open(os.path.join(profile_dir, file), 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    profiles.append({
                        'id': profile['id'],
                        'name': profile['name']
                    })
    return render_template('select_user.html', profiles=profiles)

@app.route('/select_profile/<profile_id>')
def select_profile(profile_id):
    # Set the student_id in session
    session['student_id'] = profile_id
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{profile_id}.json")
    if not os.path.exists(profile_path):
        return redirect(url_for('select_user'))
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    
    # Get or create session_id
    if 'session_id' not in session:
        # Try to find existing session_id from profile
        if 'current_session_id' in profile:
            session['session_id'] = profile['current_session_id']
        else:
            # Create new session_id
            session['session_id'] = str(uuid.uuid4())
            profile['current_session_id'] = session['session_id']
            # Save updated profile
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=4)
    
    # Check if the profile has a learning path
    if profile.get('learning_path'):
        # If student has a learning path, redirect to learning page
        return redirect(url_for('learning'))
    else:
        # If no learning path exists, redirect to upload PDF
        return redirect(url_for('upload_pdf'))

@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
    if request.method == 'POST':
        name = request.form['name']
        student_id = str(uuid.uuid4())[:8]
        
        # Create basic profile
        profile = {
            'id': student_id,
            'name': name,
            'learning_style': '',
            'current_knowledge_level': '',
            'strengths': [],
            'areas_for_improvement': [],
            'interests': [],
            'learning_history': [],
            'current_module_index': 0,
            'learning_path_confirmed': False
        }
        
        # Save profile
        os.makedirs('student_profiles', exist_ok=True)
        with open(f'student_profiles/{student_id}.json', 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=4)
        
        session['student_id'] = student_id
        # Redirect to learning style survey instead of upload_pdf
        return redirect(url_for('learning_style_survey'))
    
    return render_template('create_user.html')

@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    if 'student_id' not in session or 'course_id' not in session:
        return redirect(url_for('course_selection'))
    
    if request.method == 'POST':
        try:
            if 'pdfs' not in request.files:
                app.logger.error('No files part in request')
                flash('No files selected')
                return redirect(request.url)
            
            files = request.files.getlist('pdfs')
            if not files or files[0].filename == '':
                app.logger.error('No files selected')
                flash('No files selected')
                return redirect(request.url)
            
            # Generate a unique session ID for this upload
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            
            # Create session directory
            session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            os.makedirs(session_dir, exist_ok=True)
            app.logger.info(f"Created session directory: {session_dir}")
            
            file_paths = []
            for file in files:
                if file and allowed_file(file.filename):
                    try:
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(session_dir, filename)
                        file.save(file_path)
                        file_paths.append(file_path)
                        app.logger.info(f"Successfully saved file: {filename}")
                    except Exception as e:
                        app.logger.error(f"Error saving file {file.filename}: {str(e)}")
                        flash(f"Error saving file {file.filename}: {str(e)}")
                        continue
            
            if not file_paths:
                app.logger.error('No valid PDF files uploaded')
                flash('No valid PDF files uploaded')
                return redirect(request.url)
            
            try:
                # Process all PDFs and create vector store
                app.logger.info("Starting PDF processing")
                chunks = process_pdfs(file_paths, session_id)
                
                # Store file paths in session
                session['pdf_files'] = [os.path.basename(path) for path in file_paths]
                
                # Update course profile with session ID
                course = get_course_profile(session['course_id'])
                if course:
                    course.session_id = session_id
                    save_course_profile(course)
                
                # Redirect to pretest
                app.logger.info("PDF processing completed successfully")
                return redirect(url_for('pretest'))
                
            except Exception as e:
                app.logger.error(f'Error processing PDFs: {str(e)}')
                flash(f'Error processing files: {str(e)}')
                return redirect(request.url)
                
        except Exception as e:
            app.logger.error(f'Error in upload_pdf: {str(e)}')
            flash(f'An unexpected error occurred: {str(e)}')
            return redirect(request.url)
    
    return render_template('upload_pdf.html')

@app.route('/course_selection')
def course_selection():
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
    # Get student profile
    profile = get_student_profile(session['student_id'])
    if not profile:
        return redirect(url_for('select_user'))
    
    # Get all courses for this student
    courses = []
    for course_id in profile.courses:
        course = get_course_profile(course_id)
        if course:
            courses.append(course.model_dump())
    
    return render_template('course_selection.html', courses=courses)

@app.route('/create_course', methods=['GET', 'POST'])
def create_course():
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
    if request.method == 'POST':
        data = request.json
        title = data.get('title')
        description = data.get('description')
        
        if not title:
            return jsonify({'error': 'Course title is required'}), 400
        
        # Create new course
        course = create_new_course(session['student_id'], title, description)
        
        # Set current course in session
        session['course_id'] = course.id
        
        return jsonify({
            'success': True,
            'course': course.model_dump(),
            'redirect': url_for('upload_pdf')
        })
    
    return render_template('create_course.html')

@app.route('/select_course/<course_id>')
def select_course(course_id):
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
    # Verify course belongs to student
    profile = get_student_profile(session['student_id'])
    if not profile or course_id not in profile.courses:
        return redirect(url_for('course_selection'))
    
    # Get course profile
    course = get_course_profile(course_id)
    if not course:
        return redirect(url_for('course_selection'))
    
    # Set current course in session
    session['course_id'] = course_id
    session['session_id'] = course.session_id
    
    # Check if course has learning path
    if course.learning_path:
        return redirect(url_for('learning'))
    else:
        return redirect(url_for('upload_pdf'))

@app.route('/pretest', methods=['GET', 'POST'])
def pretest():
    if 'student_id' not in session or 'course_id' not in session:
        return redirect(url_for('course_selection'))
    
    if request.method == 'POST':
        data = request.json
        answers = data.get('answers', [])
        pretest_data = session.get('pretest')
        
        if not pretest_data:
            return jsonify({'error': 'No pretest available'}), 400
        
        if len(answers) != len(pretest_data['questions']):
            return jsonify({'error': 'Number of answers does not match number of questions'}), 400
        
        # Calculate score and determine knowledge level
        correct_count = 0
        results = []
        
        for i, answer in enumerate(answers):
            question = pretest_data['questions'][i]
            correct_letter = question['correct_answer'][0].upper()
            is_correct = answer.upper() == correct_letter
            
            if is_correct:
                correct_count += 1
            
            results.append({
                'question': question['question'],
                'student_answer': answer,
                'correct_answer': question['correct_answer'],
                'is_correct': is_correct,
                'explanation': question['explanation'],
                'difficulty': question['difficulty']
            })
        
        score_percentage = (correct_count / len(answers)) * 100
        knowledge_level = evaluate_knowledge_level(score_percentage)
        
        # Get course profile
        course = get_course_profile(session['course_id'])
        if not course:
            return jsonify({'error': 'Course profile not found'}), 400
        
        # Update course profile with learning path
        course.current_knowledge_level = knowledge_level
        course.learning_path = generate_learning_path(session['session_id'], course, {
            'score_percentage': score_percentage,
            'knowledge_level': knowledge_level,
            'results': results
        })
        course.current_module_index = 0
        course.learning_path_confirmed = False  # Reset confirmation status
        save_course_profile(course)
        
        # Return detailed results
        return jsonify({
            'score': correct_count,
            'total': len(answers),
            'percentage': score_percentage,
            'knowledge_level': knowledge_level,
            'results': results,
            'redirect': url_for('learning_path_discussion')
        })
    
    # Generate pretest
    pretest_data = create_pretest(session['session_id'])
    session['pretest'] = pretest_data
    return render_template('pretest.html', pretest=pretest_data)

@app.route('/learning', methods=['GET', 'POST'])
def learning():
    if 'student_id' not in session or 'course_id' not in session:
        return redirect(url_for('course_selection'))
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return redirect(url_for('course_selection'))
    
    # Get student profile
    profile = get_student_profile(session['student_id'])
    if not profile:
        return redirect(url_for('select_user'))
    
    # Check if student has completed learning style survey
    if not profile.learning_style:
        return redirect(url_for('learning_style_survey'))
    
    # Check if course has a learning path
    if not course.learning_path:
        return redirect(url_for('pretest'))
    
    # Get current module index
    current_module_index = course.current_module_index or 0
    
    # Validate current module index
    if current_module_index >= len(course.learning_path['modules']):
        # If student has completed all modules, redirect to summary
        return redirect(url_for('summary'))
    
    # Get current module
    current_module = course.learning_path['modules'][current_module_index]
    
    # Calculate scaffolding level for display
    retry_count = current_module.get('retry_count', 0)
    knowledge_level = course.current_knowledge_level
    emotional_state = current_module.get('emotional_analysis', {})
    base_scaffolding_level = current_module.get('scaffolding_level', 'medium')
    
    # Adjust scaffolding level based on emotional state
    if emotional_state:
        if emotional_state.get('frustration_level') == 'high' or emotional_state.get('confidence_level') == 'low':
            base_scaffolding_level = 'high'
        elif emotional_state.get('engagement_level') == 'high' and emotional_state.get('confidence_level') == 'high':
            base_scaffolding_level = 'low'
    
    # Further adjust based on knowledge level and retry count
    if knowledge_level == "advanced" and base_scaffolding_level != 'high':
        scaffolding_level = "low"
    elif knowledge_level == "intermediate":
        if retry_count == 0 and base_scaffolding_level != 'high':
            scaffolding_level = "medium"
        else:
            scaffolding_level = "high"
    else:
        scaffolding_level = "high"
        
    scaffolding_map = {
        "high": {"label": "高", "description": "提供詳細的步驟、更多範例和引導性問題，適合初學者或需要更多支持的學習者。"},
        "medium": {"label": "中", "description": "提供適度的解釋和範例，並加入一些思考性問題，適合已有基礎的學習者。"},
        "low": {"label": "低", "description": "提供核心概念，鼓勵學習者自主探索和深入思考，適合進階學習者。"}
    }
    scaffolding_info = scaffolding_map.get(scaffolding_level, scaffolding_map["medium"])

    # Get next module for transition content
    next_module = None
    if current_module_index + 1 < len(course.learning_path['modules']):
        next_module = course.learning_path['modules'][current_module_index + 1]
    
    # Generate module content if not already present
    if 'content' not in current_module:
        try:
            content = generate_module_content(course.session_id, current_module, profile)
            
            # Add transition content if there's a next module
            if next_module:
                transition_content = generate_module_transition(course.session_id, current_module, next_module, profile)
                if transition_content:
                    content += f"\n\n## 章節過渡\n\n{transition_content}"
            
            current_module['content'] = content
            # Save the updated course profile with content
            save_course_profile(course)
        except Exception as e:
            current_module['content'] = f"Error generating content: {str(e)}"
    
    # Calculate progress percentage
    progress_percentage = 0
    if course.learning_path and course.learning_path['modules']:
        progress_percentage = (current_module_index / len(course.learning_path['modules']) * 100)
    
    return render_template('learning.html', 
                         student_profile=profile.model_dump(),
                         course=course.model_dump(),
                         current_module=current_module,
                         next_module=next_module,
                         progress_percentage=progress_percentage,
                         scaffolding_info=scaffolding_info)

@app.route('/api/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        data = request.json
        name = data.get('name', '')
        profile = create_or_get_student_profile(name)
        return jsonify(profile.model_dump())
    else:
        profile = create_or_get_student_profile()
        return jsonify(profile.model_dump())

@app.route('/api/learning-style-survey', methods=['GET', 'POST'])
def learning_style():
    if request.method == 'POST':
        data = request.json
        survey = data.get('survey', {})
        answers = data.get('answers', [])
        
        if not survey or not answers:
            return jsonify({'error': 'Survey and answers are required'}), 400
        
        dominant_style = process_learning_style_answers(survey, answers)
        
        # Update student profile
        profile = create_or_get_student_profile()
        profile.learning_style = dominant_style
        save_student_profile(profile)
        
        return jsonify({
            'learning_style': dominant_style,
            'profile': profile.model_dump()
        })
    else:
        return jsonify(survey)

@app.route('/api/learning-path', methods=['GET'])
def get_learning_path():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    pretest_results = session.get('pretest_results')
    if not pretest_results:
        return jsonify({'error': 'Complete the pretest first'}), 400
    
    profile = create_or_get_student_profile()
    
    try:
        learning_path = generate_learning_path(session_id, profile, pretest_results)
        session['learning_path'] = learning_path  # Store for later use
        return jsonify(learning_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/module-content/<int:module_index>', methods=['GET'])
def get_module_content(module_index):
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    learning_path = session.get('learning_path')
    if not learning_path:
        return jsonify({'error': 'No learning path available'}), 400
    
    if module_index >= len(learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400
    
    module = learning_path['modules'][module_index]
    profile = create_or_get_student_profile()
    
    try:
        content = generate_module_content(session_id, module, profile)
        return jsonify({
            'module': module,
            'content': content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/peer-discussion', methods=['POST'])
def peer_discussion():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    data = request.json
    topic = data.get('topic', '')
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = simulate_peer_discussion(session_id, topic, message)
        return jsonify({
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/posttest/<int:module_index>', methods=['GET'])
def get_posttest(module_index):
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400
    
    # Check if course has learning path
    if not course.learning_path:
        return jsonify({'error': 'No learning path found'}), 400
    
    if module_index >= len(course.learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400
    
    module = course.learning_path['modules'][module_index]
    
    try:
        # Check if vector DB exists for this session
        vector_db_path = os.path.join(app.config['VECTOR_DB_DIR'], course.session_id)
        if not os.path.exists(vector_db_path):
            return jsonify({'error': 'No documents have been processed yet. Please upload documents first.'}), 400
        
        posttest_data = create_posttest(course.session_id, module, course)
        
        # Store posttest data in file instead of session
        posttest_dir = os.path.join(app.config['UPLOAD_FOLDER'], course.session_id, 'posttests')
        os.makedirs(posttest_dir, exist_ok=True)
        posttest_path = os.path.join(posttest_dir, f'posttest_{module_index}.json')
        
        with open(posttest_path, 'w', encoding='utf-8') as f:
            json.dump(posttest_data, f, ensure_ascii=False, indent=4)
        
        return jsonify(posttest_data)
    except Exception as e:
        app.logger.error(f"Error creating posttest: {str(e)}")
        return jsonify({'error': f'Error creating posttest: {str(e)}'}), 500

@app.route('/api/evaluate-posttest/<int:module_index>', methods=['POST'])
def evaluate_posttest(module_index):
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400
    
    data = request.json
    answers = data.get('answers', [])
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400
    
    # Check if course has learning path
    if not course.learning_path:
        return jsonify({'error': 'No learning path found'}), 400
    
    if module_index >= len(course.learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400
    
    module = course.learning_path['modules'][module_index]
    
    # Load posttest data from file
    posttest_path = os.path.join(app.config['UPLOAD_FOLDER'], course.session_id, 'posttests', f'posttest_{module_index}.json')
    if not os.path.exists(posttest_path):
        return jsonify({'error': 'No posttest available for this module'}), 400
    
    with open(posttest_path, 'r', encoding='utf-8') as f:
        posttest_data = json.load(f)
    
    if len(answers) != len(posttest_data['questions']):
        return jsonify({'error': 'Number of answers does not match number of questions'}), 400
    
    # Calculate score
    correct_count = 0
    results = []
    
    for i, answer in enumerate(answers):
        question = posttest_data['questions'][i]
        correct_letter = question['correct_answer'][0].upper()
        is_correct = answer.upper() == correct_letter
        
        if is_correct:
            correct_count += 1
        
        results.append({
            'question': question['question'],
            'student_answer': answer,
            'correct_answer': question['correct_answer'],
            'is_correct': is_correct,
            'explanation': question['explanation'],
            'difficulty': question['difficulty']
        })
    
    score_percentage = (correct_count / len(answers)) * 100
    
    # Update course profile
    previous_level = course.current_knowledge_level
    
    # Potentially adjust knowledge level based on score
    if score_percentage >= 80 and previous_level != "advanced":
        if previous_level == "beginner":
            new_level = "intermediate"
        else:
            new_level = "advanced"
        course.current_knowledge_level = new_level
    elif score_percentage < 50 and previous_level != "beginner":
        if previous_level == "advanced":
            new_level = "intermediate"
        else:
            new_level = "beginner"
        course.current_knowledge_level = new_level
    else:
        new_level = previous_level
    
    # Add to learning history
    course.learning_history.append({
        'activity_type': 'posttest',
        'module': module['title'],
        'timestamp': datetime.datetime.now().isoformat(),
        'score': f'{correct_count}/{len(answers)}',
        'percentage': score_percentage,
        'previous_level': previous_level,
        'current_level': new_level
    })
    
    # Save updated course profile
    save_course_profile(course)
    
    # Store results in file
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], course.session_id, 'posttests', f'results_{module_index}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'score': correct_count,
            'total': len(answers),
            'percentage': score_percentage,
            'results': results
        }, f, ensure_ascii=False, indent=4)
    
    return jsonify({
        'score': correct_count,
        'total': len(answers),
        'percentage': score_percentage,
        'previous_level': previous_level,
        'new_level': new_level,
        'results': results,
        'profile': course.model_dump()
    })

@app.route('/api/learning-log/<int:module_index>', methods=['POST'])
def create_learning_log(module_index):
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400

    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400

    # Check if course has learning path
    if not course.learning_path:
        return jsonify({'error': 'No learning path found'}), 400

    if module_index >= len(course.learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400

    module = course.learning_path['modules'][module_index]
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    
    data = request.json
    log_content = data.get('content', '')
    is_retry = data.get('retry', False)
    
    if not log_content:
        return jsonify({'error': 'No log content provided'}), 400
    
    # Get student profile for name
    student_profile = get_student_profile(session['student_id'])
    if not student_profile:
        return jsonify({'error': 'Student profile not found'}), 400
    
    # Create learning log
    log_id = str(uuid.uuid4())[:8]
    log = LearningLog(
        id=log_id,
        student_id=student_profile.id,
        timestamp=datetime.datetime.now().isoformat(),
        topic=module_topic,
        content=log_content,
        reflections=[],
        questions=[],
        next_steps=[]
    )
    
    # Analyze learning log
    try:
        analysis = analyze_learning_log(student_profile.name, module_topic, log_content)
        
        # Update log with analysis results
        if "recommended_next_steps" in analysis:
            log.next_steps = analysis["recommended_next_steps"]
        
        # Update course profile strengths and areas for improvement
        if "strengths" in analysis:
            for strength in analysis["strengths"]:
                if strength not in course.strengths:
                    course.strengths.append(strength)
        
        if "areas_for_improvement" in analysis:
            for area in analysis["areas_for_improvement"]:
                if area not in course.areas_for_improvement:
                    course.areas_for_improvement.append(area)
        
        # Update module's scaffolding level based on emotional analysis
        if "emotional_state" in analysis and "recommended_scaffolding_level" in analysis:
            emotional_state = analysis["emotional_state"]
            recommended_level = analysis["recommended_scaffolding_level"]
            
            # Store emotional analysis in module
            module['emotional_analysis'] = emotional_state
            module['scaffolding_level'] = recommended_level
            
            # If frustration is high or confidence is low, increase scaffolding
            if emotional_state['frustration_level'] == 'high' or emotional_state['confidence_level'] == 'low':
                module['scaffolding_level'] = 'high'
            # If engagement is high and confidence is high, decrease scaffolding
            elif emotional_state['engagement_level'] == 'high' and emotional_state['confidence_level'] == 'high':
                module['scaffolding_level'] = 'low'
        
        # If this is a retry, mark the module for easier test
        if is_retry:
            module['retry_count'] = module.get('retry_count', 0) + 1
            module['easier_test'] = True
            module['scaffolding_level'] = 'high'  # Increase scaffolding for retry
        
        # Save the course profile
        save_course_profile(course)
        
        # Save the learning log
        with open(os.path.join(app.config['LEARNING_LOGS_DIR'], f"{log_id}.json"), 'w', encoding='utf-8') as f:
            f.write(log.model_dump_json(indent=4))
        
        return jsonify({
            'log': log.model_dump(),
            'analysis': analysis,
            'scaffolding_level': module.get('scaffolding_level', 'medium')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/update-module-index', methods=['POST'])
def update_module_index():
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400
    
    data = request.json
    new_module_index = data.get('module_index')
    
    if new_module_index is None:
        return jsonify({'error': 'No module index provided'}), 400
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400
    
    # Check if learning path exists and get total number of modules
    if not course.learning_path:
        return jsonify({'error': 'No learning path found'}), 400
    
    total_modules = len(course.learning_path['modules'])
    
    # Check if the student has completed all modules
    if new_module_index >= total_modules:
        # Update the course profile to mark completion
        course.current_module_index = total_modules - 1
        course.learning_completed = True
        course.completion_date = datetime.datetime.now().isoformat()
        
        # Save the updated course profile
        save_course_profile(course)
        
        return jsonify({
            'success': True,
            'new_index': new_module_index,
            'total_modules': total_modules,
            'finished': True
        })
    
    # Validate the new module index
    if not isinstance(new_module_index, int) or new_module_index < 0:
        return jsonify({
            'error': f'Invalid module index. Must be between 0 and {total_modules-1}',
            'current_index': course.current_module_index,
            'total_modules': total_modules
        }), 400
    
    # Update the module index
    course.current_module_index = new_module_index
    
    # Save the updated course profile
    save_course_profile(course)
    
    return jsonify({
        'success': True,
        'new_index': new_module_index,
        'total_modules': total_modules,
        'finished': False
    })

@app.route('/summary')
def summary():
    if 'student_id' not in session or 'course_id' not in session:
        return redirect(url_for('course_selection'))
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return redirect(url_for('course_selection'))
    
    # Get student profile
    profile = get_student_profile(session['student_id'])
    if not profile:
        return redirect(url_for('select_user'))
    
    # Read all learning logs for this course
    logs = []
    for log_file in os.listdir(app.config['LEARNING_LOGS_DIR']):
        if log_file.endswith('.json'):
            try:
                with open(os.path.join(app.config['LEARNING_LOGS_DIR'], log_file), 'r', encoding='utf-8') as lf:
                    log = json.load(lf)
                    if log['student_id'] == profile.id:
                        logs.append(log)
            except Exception as e:
                app.logger.error(f"Error reading log file {log_file}: {str(e)}")
                continue
    
    # Sort logs by timestamp
    logs.sort(key=lambda x: x['timestamp'])
    
    # Get learning history from course profile
    learning_history = course.learning_history if course.learning_history else []
    
    return render_template('summary.html', 
                         student_profile=profile.model_dump(),
                         course=course.model_dump(),
                         logs=logs,
                         learning_history=learning_history)

@app.route('/learning_path_discussion')
def learning_path_discussion():
    if 'student_id' not in session or 'course_id' not in session:
        return redirect(url_for('course_selection'))
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return redirect(url_for('course_selection'))
    
    # Get student profile
    profile = get_student_profile(session['student_id'])
    if not profile:
        return redirect(url_for('select_user'))
    
    # Check if learning path exists
    if not course.learning_path:
        return redirect(url_for('pretest'))
    
    # Check if learning path has already been confirmed
    if course.learning_path_confirmed:
        return redirect(url_for('learning'))
    
    return render_template('learning_path_discussion.html', 
                         learning_path=course.learning_path)

@app.route('/api/discuss-learning-path', methods=['POST'])
def discuss_learning_path():
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400
    
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400
    
    # Get learning path
    learning_path = course.learning_path
    if not learning_path:
        return jsonify({'error': 'No learning path found'}), 400
    
    # Get student profile
    student_profile = get_student_profile(session['student_id'])
    if not student_profile:
        return jsonify({'error': 'Student profile not found'}), 400
    
    # Get vectorstore
    vectorstore = get_vectorstore(course.session_id)
    
    # 使用兩種檢索策略
    # 1. 相關性檢索
    relevance_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 20
        }
    )
    
    # 2. 全面檢索 - 獲取所有文檔
    all_docs_data = vectorstore.get()
    all_docs = []
    for i, doc in enumerate(all_docs_data["documents"]):
        metadata = all_docs_data["metadatas"][i] if i < len(all_docs_data["metadatas"]) else {}
        all_docs.append({
            "page_content": doc,
            "metadata": metadata
        })
    
    # Create chat model with retry mechanism
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            chat_model = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-2.0-flash-lite"
            )
            break
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                return jsonify({'error': f'Failed to initialize chat model after {max_retries} attempts: {str(e)}'}), 500
            time.sleep(1)
    
    # Create prompt for discussion
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
            你是一位專業的學習路徑設計師，正在與學生討論他們的個人化學習路徑。
            你已經擁有學生的學習路徑和個人檔案。
            當學生提出修改需求時，請直接根據這些資訊自動修改學習路徑，並且只需用簡短的中文回覆說明修改了什麼。
            禁止要求學生再提供任何資訊，也不要詢問學生「請提供 learning_path 或 profile」。
            
            在修改學習路徑時，請確保：
            1. 完整涵蓋教材中的所有重要內容和概念
            2. 針對學生的學習風格、知識水平進行量身調整
            3. 遵循鷹架原則，逐步增加難度並減少支持
            4. 確保每章節都涵蓋相關的教材內容，不要遺漏任何重要概念
            5. 確保學習路徑的連續性和完整性，章節之間要有清晰的關聯
            
            回覆格式要求：
            1. 只回覆修改的內容，不要回覆完整的學習路徑
            2. 使用簡短的中文說明修改了什麼
            3. 如果沒有需要修改的地方，請簡短說明原因
            4. 回覆要非常簡潔，不要包含任何多餘的內容
            5. 例如: 已經按照指示改成2個章節!
            """),
        HumanMessagePromptTemplate.from_template("""根據以下內容進行學習路徑討論：
        
        學生檔案：
        {profile}
        
        當前學習路徑：
        {learning_path}
        
        相關內容：
        {relevant_context}
        
        完整教材：
        {all_context}
        
        學生訊息：{message}
        """)
    ])
    
    # Create the chain
    discussion_chain = (
        RunnablePassthrough()
        | (lambda _: {
            "profile": json.dumps({
                "name": student_profile.name,
                "learning_style": student_profile.learning_style,
                "current_knowledge_level": course.current_knowledge_level
            }, ensure_ascii=False),
            "learning_path": json.dumps(learning_path, ensure_ascii=False),
            "message": message,
            "relevant_context": "\n\n".join([f"來源: {doc.metadata.get('source', 'unknown')}, 頁碼: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}" for doc in relevance_retriever.invoke(message)]),
            "all_context": "\n\n".join([f"來源: {doc['metadata'].get('source', 'unknown')}, 頁碼: {doc['metadata'].get('page', 'unknown')}\n{doc['page_content']}" for doc in all_docs])
        })
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    try:
        # Add retry mechanism for the discussion chain
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = discussion_chain.invoke("")
                # 清理回應，移除多餘的內容
                response = response.strip()
                if response.startswith("修改"):
                    response = response[2:].strip()
                if response.endswith("。"):
                    response = response[:-1]
                break
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    return jsonify({'error': f'Failed to generate discussion response after {max_retries} attempts: {str(e)}'}), 500
                time.sleep(1)
        
        # Add retry mechanism for learning path generation
        retry_count = 0
        while retry_count < max_retries:
            try:
                new_learning_path = generate_learning_path(
                    course.session_id,
                    course,
                    {
                        'score_percentage': course.current_knowledge_level,
                        'knowledge_level': course.current_knowledge_level,
                        'student_requirements': message
                    }
                )
                break
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    return jsonify({'error': f'Failed to generate new learning path after {max_retries} attempts: {str(e)}'}), 500
                time.sleep(1)
        
        # Update course profile with new learning path
        course.learning_path = new_learning_path
        
        # Save updated course profile
        save_course_profile(course)
        
        return jsonify({
            'response': response,
            'path_adjusted': True
        })
    except Exception as e:
        app.logger.error(f"Error in discuss_learning_path: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/api/adjust-learning-path', methods=['POST'])
def adjust_learning_path():
    if 'student_id' not in session:
        return jsonify({'error': 'No student session found'}), 400
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return jsonify({'error': 'Student profile not found'}), 400
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    try:
        # Generate new learning path
        new_learning_path = generate_learning_path(
            session['session_id'],
            StudentProfile.model_validate(student_profile),
            {
                'score_percentage': student_profile.get('pretest_score', 0),
                'knowledge_level': student_profile['current_knowledge_level']
            }
        )
        
        # Update student profile with new learning path
        student_profile['learning_path'] = new_learning_path
        
        # Save updated profile
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(student_profile, f, ensure_ascii=False, indent=4)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/confirm-learning-path', methods=['POST'])
def confirm_learning_path():
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400
    
    # Mark learning path as confirmed
    course.learning_path_confirmed = True
    
    # Save updated course profile
    save_course_profile(course)
    
    # Return success with redirect to learning page
    return jsonify({
        'success': True,
        'redirect': url_for('learning')
    })

@app.route('/api/get-current-learning-path', methods=['GET'])
def get_current_learning_path():
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400
    
    # Check if learning path exists
    if not course.learning_path:
        return jsonify({'error': 'No learning path found'}), 400
    
    return jsonify({
        'learning_path': course.learning_path
    })

@app.route('/view_pdf/<path:filename>')
def view_pdf(filename):
    """Serve PDF files for viewing"""
    if 'session_id' not in session:
        return redirect(url_for('select_user'))
    
    # Get the PDF file path
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], session['session_id'], filename)
    
    if not os.path.exists(pdf_path):
        return "PDF file not found", 404
    
    return send_file(pdf_path, mimetype='application/pdf')

@app.route('/api/module-transition/<int:module_index>', methods=['GET'])
def get_module_transition(module_index):
    if 'student_id' not in session or 'course_id' not in session:
        return jsonify({'error': 'No student or course session found'}), 400
    
    # Get course profile
    course = get_course_profile(session['course_id'])
    if not course:
        return jsonify({'error': 'Course profile not found'}), 400
    
    # Check if course has learning path
    if not course.learning_path:
        return jsonify({'error': 'No learning path found'}), 400
    
    if module_index >= len(course.learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400
    
    # Get current and next module
    current_module = course.learning_path['modules'][module_index]
    next_module = None
    if module_index + 1 < len(course.learning_path['modules']):
        next_module = course.learning_path['modules'][module_index + 1]
    
    if not next_module:
        return jsonify({'transition': ''})
    
    # Get student profile
    profile = get_student_profile(session['student_id'])
    if not profile:
        return jsonify({'error': 'Student profile not found'}), 400
    
    try:
        transition_content = generate_module_transition(course.session_id, current_module, next_module, profile)
        return jsonify({
            'transition': transition_content,
            'next_module': next_module
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_course/<course_id>', methods=['POST'])
def delete_course(course_id):
    if 'student_id' not in session:
        return jsonify({'error': 'No student session found'}), 400

    # 1. 從 student_profile 移除該課程ID
    student_profile = get_student_profile(session['student_id'])
    if not student_profile:
        return jsonify({'error': 'Student profile not found'}), 400
    if course_id in student_profile.courses:
        student_profile.courses.remove(course_id)
        save_student_profile(student_profile)

    # 2. 刪除 course_profiles 下對應的 json
    course_profile_path = os.path.join(app.config['COURSE_PROFILES_DIR'], f"{course_id}.json")
    session_id = None
    if os.path.exists(course_profile_path):
        # 取得 session_id 以便刪除相關資料夾
        try:
            with open(course_profile_path, 'r', encoding='utf-8') as f:
                course_data = json.load(f)
                session_id = course_data.get('session_id')
        except Exception:
            pass
        os.remove(course_profile_path)

    # 3. 刪除 data 和 vectordbs 下該課程的 session_id 資料夾（如果有）
    if session_id:
        for base_dir in [app.config['UPLOAD_FOLDER'], app.config['VECTOR_DB_DIR']]:
            target_dir = os.path.join(base_dir, session_id)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)

    # 4. 若目前 session 的 course_id 是被刪除的，也從 session 移除
    if session.get('course_id') == course_id:
        session.pop('course_id', None)
        session.pop('session_id', None)

    return jsonify({'success': True})

def load_questions():
    with open("question.txt", encoding="utf-8") as f:
        lines = f.read().splitlines()

    categories = {
        "Active vs Reflective": ("active", "reflective"),
        "Sensing vs Intuitive": ("sensing", "intuitive"),
        "Visual vs Verbal": ("visual", "verbal"),
        "Sequential vs Global": ("sequential", "global"),
    }

    questions = {
        "active": [],
        "sensing": [],
        "visual": [],
        "sequential": [],
    }

    current_cat = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("===") and "===" in line:
            for key in categories:
                if key in line:
                    current_cat = key
                    break
            i += 1
            continue

        if line and not line.startswith("draw_count:") and not line.startswith("===") and not line.startswith("(a)") and not line.startswith("(b)"):
            q_text = line
            a_text, b_text = "", ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("(a)"):
                a_text = lines[i + 1].strip()[3:].strip()
            if i + 2 < len(lines) and lines[i + 2].strip().startswith("(b)"):
                b_text = lines[i + 2].strip()[3:].strip()

            type_a, type_b = categories[current_cat]
            questions[type_a].append({
                "question": q_text,
                "option_a": a_text,
                "option_b": b_text,
                "type_a": type_a,
                "type_b": type_b,
            })
            i += 3
        else:
            i += 1

    final_questions = []
    for dim in ["active", "sensing", "visual", "sequential"]:
        final_questions.extend(random.sample(questions[dim], 4))

    return final_questions


@app.route("/learning_style_survey", methods=["GET", "POST"])
def learning_style_survey():
    if request.method == "POST":
        score = {
            "active": 0, "reflective": 0,
            "sensing": 0, "intuitive": 0,
            "visual": 0, "verbal": 0,
            "sequential": 0, "global": 0
        }
        for i in range(16):
            selected = request.form.get(f"q{i}")
            type_ = request.form.get(f"type{i}")
            opposite_map = {
                "active": "reflective",
                "sensing": "intuitive",
                "visual": "verbal",
                "sequential": "global"
            }
            if selected == "a":
                score[type_] += 1
            elif selected == "b" and type_ in opposite_map:
                score[opposite_map[type_]] += 1

        result = {
            "active_vs_reflective": "Active" if score["active"] >= score["reflective"] else "Reflective",
            "sensing_vs_intuitive": "Sensing" if score["sensing"] >= score["intuitive"] else "Intuitive",
            "visual_vs_verbal": "Visual" if score["visual"] >= score["verbal"] else "Verbal",
            "sequential_vs_global": "Sequential" if score["sequential"] >= score["global"] else "Global"
        }

        # ✅ 儲存結果到學生 JSON 檔案
        if 'student_id' in session:
            profile = get_student_profile(session['student_id'])
            if profile:
                profile.felder_silverman_profile = result
                profile.learning_style = result["visual_vs_verbal"]  # 可根據需要選擇一個主軸
                save_student_profile(profile)

        return render_template("learning_style_result.html", result=result)

    # GET request: 顯示問卷
    questions = load_questions()
    return render_template("learning_style_survey.html", questions=questions)


# 你原本的其他頁面路由可放在這裡

if __name__ == '__main__':
    app.run(debug=True)