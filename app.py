from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import json
import datetime
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

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['VECTOR_DB_DIR'] = 'vectordbs'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['STUDENT_PROFILES_DIR'] = 'student_profiles'
app.config['LEARNING_LOGS_DIR'] = 'learning_logs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_DB_DIR'], exist_ok=True)
os.makedirs(app.config['STUDENT_PROFILES_DIR'], exist_ok=True)
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
    current_knowledge_level: str = Field(description="Beginner, intermediate, or advanced")
    strengths: List[str] = Field(description="Academic strengths")
    areas_for_improvement: List[str] = Field(description="Areas that need improvement")
    interests: List[str] = Field(description="Academic interests")
    learning_history: List[Dict[str, Any]] = Field(description="History of learning activities")
    learning_path: Optional[Dict[str, Any]] = Field(default=None, description="Personalized learning path")
    current_module_index: Optional[int] = Field(default=0, description="Current module index")
    learning_path_confirmed: Optional[bool] = Field(default=False, description="Whether the learning path has been confirmed")

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
        
        # Create summary
        try:
            chat_model = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model="gemini-1.5-flash"
            )
            
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful assistant that summarizes technical documents."),
                HumanMessagePromptTemplate.from_template("""Summarize the following content into a bullet-point outline for review:

{context}

Summary:""")
            ])

            summary_chain = (
                RunnablePassthrough()
                | (lambda docs: {"context": "\n\n".join([d.page_content for d in docs])})
                | summary_prompt
                | chat_model
                | StrOutputParser()
            )
            
            summary = summary_chain.invoke(chunks)
            app.logger.info("Successfully generated document summary")
        except Exception as e:
            app.logger.error(f'Error generating summary: {str(e)}')
            raise Exception(f'Error generating document summary: {str(e)}')
        
        return chunks, summary
        
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

def get_answer(session_id, question):
    """Get an answer to a question using the retriever"""
    vectorstore = get_vectorstore(session_id)
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專精於回答問題的專家。
        您將被提供一份內容和一個問題。
        請根據提供的內容回答問題。"""),
        HumanMessagePromptTemplate.from_template("""根據給定的內容回答問題。
        內容：{context}
        問題：{question}
        答案：""")
    ])
    
    chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return chain.invoke(question)

# Educational system functions
def create_or_get_student_profile(name=None):
    """Create a new student profile or retrieve an existing one"""
    if 'student_id' in session:
        # Try to load existing profile
        profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                return StudentProfile.model_validate(json.load(f))
    
    # Create new profile
    student_id = str(uuid.uuid4())[:8]
    name = name or f"Student_{student_id}"
    
    profile = StudentProfile(
        id=student_id,
        name=name,
        learning_style="",
        current_knowledge_level="",
        strengths=[],
        areas_for_improvement=[],
        interests=[],
        learning_history=[],
        learning_path=None,
        current_module_index=0,
        learning_path_confirmed=False
    )
    
    # Save the profile
    session['student_id'] = student_id
    session['session_id'] = str(uuid.uuid4())
    with open(os.path.join('student_profiles', f"{student_id}.json"), 'w', encoding='utf-8') as f:
        f.write(profile.model_dump_json(indent=4))
    
    return profile

def save_student_profile(profile):
    """Save a student profile to disk"""
    profile_path = os.path.join('student_profiles', f"{profile.id}.json")
    with open(profile_path, 'w', encoding='utf-8') as f:
        f.write(profile.model_dump_json(indent=4))

    """Generate a learning style assessment survey"""
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert in learning style assessment.
        Design a concise but effective learning style assessment questionnaire with 5 multiple-choice questions.
        Each question should have 3 options, aimed at determining if the student is primarily:
        1. Visual learner
        2. Auditory learner
        3. Kinesthetic learner
        
        Format your response as a questionnaire with numbered questions and lettered options.
        
        Return the result as a valid JSON object with this structure:
        {
            "title": "Learning Style Assessment",
            "description": "This assessment will help identify your primary learning style",
            "questions": [
                {
                    "question": "Question text?",
                    "choices": ["A. Option A (Visual)", "B. Option B (Auditory)", "C. Option C (Kinesthetic)"],
                    "visual_index": 0,
                    "auditory_index": 1,
                    "kinesthetic_index": 2
                }
            ]
        }
        """),
        HumanMessagePromptTemplate.from_template("Create a learning style assessment questionnaire.")
    ])
    
    chain = prompt | chat_model | JsonOutputParser()
    return chain.invoke({})

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
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
        """),
        HumanMessagePromptTemplate.from_template("""根據以下內容生成一份前測：
        
        {context}
        """)
    ])
    
    # Create the chain
    pretest_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {"context": "\n\n".join([d.page_content for d in docs])})
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    # Get student requirements if they exist
    student_requirements = test_results.get('student_requirements', '')
    requirements_prompt = f"\nStudent's specific requirements: {student_requirements}" if student_requirements else ""
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""您是一位專精於個人化學習路徑設計的教育課程設計專家。
        根據提供的學生檔案、測驗結果和內容，創建一條適合他自學的學習路徑。
        
        您的學習路徑必須：
        1. 完整涵蓋教材中的所有重要內容和概念
        2. 針對學生的學習風格、知識水平進行量身調整
        3. 遵循鷹架原則，逐步增加難度並減少支持
        4. Take into account any specific requirements or preferences expressed by the student{requirements_prompt}
        
        您的回應必須遵循這個精確的 JSON 格式：
        {{
          "title": "針對[主題]的個人化學習路徑",
          "description": "此學習路徑針對[name]的學習風格和當前知識水平進行量身定制",
          "objectives": ["目標 1", "目標 2", "目標 3"],
          "modules": [
            {{
              "title": "章節 1: [標題]",
              "description": "章節描述",
              "activities": [
                {{
                  "content": "列出該章節要學會的內容",
                  "source": "資料來源"
                }}
              ],
              
            }}
          ]
        }}
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
    
    # Format profile and test results
    profile_json = json.dumps({
        "name": profile.name,
        "learning_style": profile.learning_style,
        "current_knowledge_level": profile.current_knowledge_level,
        "interests": profile.interests
    })
    
    # Create the chain
    learning_path_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "profile": profile_json,
            "test_results": json.dumps(test_results),
            "context": "\n\n".join([d.page_content for d in docs])
        })
        | prompt
        | chat_model
        | JsonOutputParser()
    )
    
    return learning_path_chain.invoke("")

def generate_module_content(session_id, module, profile):
    """Generate educational content for a module"""
    vectorstore = get_vectorstore(session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    # Get module topic
    module_topic = module["title"].split(": ", 1)[1] if ": " in module["title"] else module["title"]
    
    # Check if this is a retry
    is_retry = module.get('retry_count', 0) > 0
    retry_count = module.get('retry_count', 0)
    
    # Determine scaffolding level based on knowledge level and retry count
    knowledge_level = profile.current_knowledge_level
    scaffolding_level = "high"  # Default to high scaffolding
    
    if knowledge_level == "advanced":
        scaffolding_level = "low"
    elif knowledge_level == "intermediate":
        if retry_count == 0:
            scaffolding_level = "medium"
        else:
            scaffolding_level = "high"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""您是一位專業的教育內容創作者，專精於鷹架學習理論。
        根據模組主題和學生的學習風格、知識水平，創造引人入勝的教育內容。
        
        您的內容應該：
        1. 針對學生的學習風格進行量身定制(盡量不要使用圖片，如果需要圖例，請使用markdown或者mermaid格式化)
        2. 適合學生的知識水平
        3. 包含清晰的關鍵概念解釋
        4. 使用例子和比喻來闡明觀點
        5. 根據鷹架支持程度 ({scaffolding_level}) 調整內容：
           - 高鷹架支持：提供詳細的步驟說明、更多例子、提示和引導性問題
           - 中鷹架支持：提供適中的解釋和例子，加入一些思考問題
           - 低鷹架支持：提供基本概念，鼓勵自主探索和思考
        
        6. 結構清晰，包含以下部分：
           - 學習目標
           - 前置知識提醒
           - 主要內容（分為小節）
           - 關鍵概念總結
           - 自我檢查問題
           - 延伸思考問題
        
        7. 每個章節都要根據context的內容標註來源(source)，格式為：
           [來源: 檔名.pdf, 頁碼: X]
           如果有多個來源，請分別列出。
        
        8. 加入互動元素：
           - 思考問題
           - 反思提示
        
        使用markdown格式化您的內容，以提高可讀性。
        """),
        HumanMessagePromptTemplate.from_template("""為以下內容創建教育內容：
        
        模組主題：{module_topic}
        學生學習風格：{learning_style}
        學生知識水平：{knowledge_level}
        鷹架支持程度：{scaffolding_level}
        資料來源：{context}
        """)
    ])
    
    # Create the chain
    content_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "module_topic": module_topic,
            "learning_style": profile.learning_style,
            "knowledge_level": profile.current_knowledge_level,
            "scaffolding_level": scaffolding_level,
            "context": "\n\n".join([f"來源: {doc.metadata.get('source', 'unknown')}, 頁碼: {doc.metadata.get('page', 'unknown')}\n{doc.page_content}" for doc in docs])
        })
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return content_chain.invoke("")

def simulate_peer_discussion(session_id, topic, message):
    """Simulate a peer discussion with an AI learning partner"""
    vectorstore = get_vectorstore(session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是「學習夥伴」，一個友善且有幫助的 AI 同儕，與學生進行有建設性的討論。
        您的角色是：
        1. 模擬一位也在學習該主題但有一定見解的同儕
        2. 提出有助於促進批判性思考的問題
        4. 以對話的方式表達想法，像是學生之間的交流
        5. 鼓勵並保持積極的態度
        
        根據提供的相關內容回應，但不要只是簡單地背誦資訊。
        而是以自然的方式進行來回討論，就像一起學習一樣，但是還是要能回答同學的問題而不是一直反問。
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    # Get module topic
    module_topic = module["title"].split(": ", 1)[1] if ": " in module["title"] else module["title"]
    
    # Get knowledge level from profile dictionary
    knowledge_level = profile.get('current_knowledge_level', 'beginner')
    
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
        根據模組內容和學生的當前知識水平，設計一份後測，包含多選題，以評估學生的學習成果。
        
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
        
        題目應該測試學生的理解、應用和分析能力。
        
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
        """),
        HumanMessagePromptTemplate.from_template("""Generate a post-test based on:
        
        學生的當前知識水平：{knowledge_level}
        鷹架支持程度：{scaffolding_level}
        章節內容：{context}
        """)
    ])
    
    # Create the chain
    posttest_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "knowledge_level": knowledge_level,
            "scaffolding_level": scaffolding_level,
            "context": "\n\n".join([d.page_content for d in docs])
        })
        | prompt
        | chat_model
        | JsonOutputParser()
    )
    
    return posttest_chain.invoke(module_topic)

def analyze_learning_log(student_name, topic, log_content):
    """Analyze a student's learning log"""
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""您是一位專業的教育分析師，專精於分析學生的學習日誌。
        根據學生的學習日誌，評估：
        
        1. 對於關鍵概念的理解程度
        2. 學生的強項和信心
        3. 學生可能的困惑或錯誤理解
        4. 學生對材料的情感反應
        5. 學習風格的指標
        
        您的回應必須遵循以下精確的 JSON 結構：
        {
          "understanding_level": "high/medium/low",
          "strengths": ["強項 1", "強項 2"],
          "areas_for_improvement": ["需要改進的領域 1", "需要改進的領域 2"],
          "emotional_response": "學生對材料的情感反應",
          "learning_style_indicators": ["學習風格指標 1", "學習風格指標 2"],
          "recommended_next_steps": ["建議的下一步 1", "建議的下一步 2"],
          "suggested_resources": ["資源 1", "資源 2"]
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
        return redirect(url_for('upload_pdf'))
    
    return render_template('create_user.html')

@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
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
                chunks, summary = process_pdfs(file_paths, session_id)
                
                # Store file paths in session
                session['pdf_files'] = [os.path.basename(path) for path in file_paths]
                
                # Redirect to learning style survey
                app.logger.info("PDF processing completed successfully")
                return redirect(url_for('learning_style_survey'))
                
            except Exception as e:
                app.logger.error(f'Error processing PDFs: {str(e)}')
                flash(f'Error processing files: {str(e)}')
                return redirect(request.url)
                
        except Exception as e:
            app.logger.error(f'Error in upload_pdf: {str(e)}')
            flash(f'An unexpected error occurred: {str(e)}')
            return redirect(request.url)
    
    return render_template('upload_pdf.html')

@app.route('/learning_style_survey', methods=['GET', 'POST'])
def learning_style_survey():
    if 'student_id' not in session or 'session_id' not in session:
        return redirect(url_for('select_user'))
    
    if request.method == 'POST':
        data = request.json
        felder_results = data.get('felder_silverman_results', {})
        
        if not felder_results:
            return jsonify({'error': 'Learning style assessment results are required'}), 400
        
        # Map the primary visual/verbal dimension to the learning style for compatibility with existing code
        # While also saving the full Felder-Silverman profile for future use
        primary_style = felder_results.get('dimension3', 'visual')  # Default to visual if missing
        
        # Update student profile
        profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
            except UnicodeDecodeError:
                # If UTF-8 fails, try with a different encoding
                with open(profile_path, 'r', encoding='big5') as f:
                    profile = json.load(f)
        else:
            # Create a new profile if it doesn't exist
            profile = {
                'id': session['student_id'],
                'name': f"Student_{session['student_id']}",
                'learning_style': '',
                'current_knowledge_level': '',
                'strengths': [],
                'areas_for_improvement': [],
                'interests': [],
                'learning_history': [],
                'learning_path': None,
                'current_module_index': 0,
                'learning_path_confirmed': False
            }
        
        # Update the profile with both simplified and detailed learning style info
        profile['learning_style'] = primary_style
        profile['felder_silverman_profile'] = felder_results
        
        # Save the updated profile
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=4)
        
        # Redirect to pretest
        return jsonify({'success': True, 'redirect': url_for('pretest')})
    
    # For GET requests, just render the template with the pre-designed questions
    return render_template('learning_style_survey.html')

@app.route('/pretest', methods=['GET', 'POST'])
def pretest():
    if 'student_id' not in session or 'session_id' not in session:
        return redirect(url_for('select_user'))
    
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
        
        # Update student profile
        profile = create_or_get_student_profile()
        profile.current_knowledge_level = knowledge_level
        profile.learning_history.append({
            'activity_type': 'pretest',
            'timestamp': datetime.datetime.now().isoformat(),
            'score': f'{correct_count}/{len(answers)}',
            'percentage': score_percentage,
            'knowledge_level': knowledge_level
        })
        profile.learning_path = generate_learning_path(session['session_id'], profile, {
            'score_percentage': score_percentage,
            'knowledge_level': knowledge_level,
            'results': results
        })
        profile.current_module_index = 0
        profile.learning_path_confirmed = False  # Reset confirmation status
        save_student_profile(profile)
        
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
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return redirect(url_for('select_user'))
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = StudentProfile.model_validate(json.load(f))
    
    # Check if student has completed learning style survey
    if not student_profile.learning_style:
        return redirect(url_for('learning_style_survey'))
    
    # Check if student has a learning path
    if not student_profile.learning_path:
        return redirect(url_for('pretest'))
    
    # Get current module index
    current_module_index = student_profile.current_module_index or 0
    
    # Validate current module index
    if current_module_index >= len(student_profile.learning_path['modules']):
        # If student has completed all modules, redirect to summary
        return redirect(url_for('summary'))
    
    # Get current module
    current_module = student_profile.learning_path['modules'][current_module_index]
    
    # Generate module content if not already present
    if 'content' not in current_module:
        try:
            content = generate_module_content(session['session_id'], current_module, student_profile)
            current_module['content'] = content
            # Save the updated profile with content
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(student_profile.model_dump(), f, ensure_ascii=False, indent=4)
        except Exception as e:
            current_module['content'] = f"Error generating content: {str(e)}"
    
    return render_template('learning.html', 
                         student_profile=student_profile.model_dump(),
                         current_module=current_module)

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
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return jsonify({'error': 'Student profile not found'}), 400
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    learning_path = student_profile.get('learning_path')
    if not learning_path:
        return jsonify({'error': 'No learning path available'}), 400
    
    if module_index >= len(learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400
    
    module = learning_path['modules'][module_index]
    
    try:
        posttest_data = create_posttest(session_id, module, student_profile)
        
        # Store posttest data in file instead of session
        posttest_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'posttests')
        os.makedirs(posttest_dir, exist_ok=True)
        posttest_path = os.path.join(posttest_dir, f'posttest_{module_index}.json')
        
        with open(posttest_path, 'w', encoding='utf-8') as f:
            json.dump(posttest_data, f, ensure_ascii=False, indent=4)
        
        return jsonify(posttest_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate-posttest/<int:module_index>', methods=['POST'])
def evaluate_posttest(module_index):
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    data = request.json
    answers = data.get('answers', [])
    
    # Load posttest data from file
    posttest_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'posttests', f'posttest_{module_index}.json')
    if not os.path.exists(posttest_path):
        return jsonify({'error': 'No posttest available for this module'}), 400
    
    with open(posttest_path, 'r', encoding='utf-8') as f:
        posttest_data = json.load(f)
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return jsonify({'error': 'Student profile not found'}), 400
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    learning_path = student_profile.get('learning_path')
    if not learning_path or module_index >= len(learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400
    
    module = learning_path['modules'][module_index]
    
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
    
    # Update student profile
    previous_level = student_profile['current_knowledge_level']
    
    # Potentially adjust knowledge level based on score
    if score_percentage >= 80 and previous_level != "advanced":
        if previous_level == "beginner":
            new_level = "intermediate"
        else:
            new_level = "advanced"
        student_profile['current_knowledge_level'] = new_level
    elif score_percentage < 50 and previous_level != "beginner":
        if previous_level == "advanced":
            new_level = "intermediate"
        else:
            new_level = "beginner"
        student_profile['current_knowledge_level'] = new_level
    else:
        new_level = previous_level
    
    # Add to learning history
    student_profile['learning_history'].append({
        'activity_type': 'posttest',
        'module': module['title'],
        'timestamp': datetime.datetime.now().isoformat(),
        'score': f'{correct_count}/{len(answers)}',
        'percentage': score_percentage,
        'previous_level': previous_level,
        'current_level': new_level
    })
    
    # Save updated profile
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(student_profile, f, ensure_ascii=False, indent=4)
    
    # Store results in file
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id, 'posttests', f'results_{module_index}.json')
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
        'profile': student_profile
    })

@app.route('/api/learning-log/<int:module_index>', methods=['POST'])
def create_learning_log(module_index):
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400

    # 這裡改成從 student_profile 讀取
    profile = create_or_get_student_profile()
    learning_path = profile.learning_path
    if not learning_path or module_index >= len(learning_path['modules']):
        return jsonify({'error': 'Invalid module index'}), 400

    module = learning_path['modules'][module_index]
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    
    data = request.json
    log_content = data.get('content', '')
    is_retry = data.get('retry', False)
    
    if not log_content:
        return jsonify({'error': 'No log content provided'}), 400
    
    # Create learning log
    log_id = str(uuid.uuid4())[:8]
    log = LearningLog(
        id=log_id,
        student_id=profile.id,
        timestamp=datetime.datetime.now().isoformat(),
        topic=module_topic,
        content=log_content,
        reflections=[],
        questions=[],
        next_steps=[]
    )
    
    # Analyze learning log
    try:
        analysis = analyze_learning_log(profile.name, module_topic, log_content)
        
        # Update log with analysis results
        if "recommended_next_steps" in analysis:
            log.next_steps = analysis["recommended_next_steps"]
        
        # Update profile strengths and areas for improvement
        if "strengths" in analysis:
            for strength in analysis["strengths"]:
                if strength not in profile.strengths:
                    profile.strengths.append(strength)
        
        if "areas_for_improvement" in analysis:
            for area in analysis["areas_for_improvement"]:
                if area not in profile.areas_for_improvement:
                    profile.areas_for_improvement.append(area)
        
        # If this is a retry, mark the module for easier test
        if is_retry:
            module['retry_count'] = module.get('retry_count', 0) + 1
            module['easier_test'] = True
        
        # Save the profile
        save_student_profile(profile)
        
        # Save the learning log
        with open(os.path.join(app.config['LEARNING_LOGS_DIR'], f"{log_id}.json"), 'w', encoding='utf-8') as f:
            f.write(log.model_dump_json(indent=4))
        
        return jsonify({
            'log': log.model_dump(),
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No documents have been uploaded yet'}), 400
    
    # Check if vector DB exists for this session
    vector_db_path = os.path.join(app.config['VECTOR_DB_DIR'], session_id)
    if not os.path.exists(vector_db_path):
        return jsonify({'error': 'No documents have been processed yet'}), 400
    
    try:
        answer = get_answer(session_id, question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-module-index', methods=['POST'])
def update_module_index():
    if 'student_id' not in session:
        return jsonify({'error': 'No student session found'}), 400
    
    data = request.json
    new_module_index = data.get('module_index')
    
    if new_module_index is None:
        return jsonify({'error': 'No module index provided'}), 400
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return jsonify({'error': 'Student profile not found'}), 400
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    # Check if learning path exists and get total number of modules
    if not student_profile.get('learning_path'):
        return jsonify({'error': 'No learning path found'}), 400
    
    total_modules = len(student_profile['learning_path']['modules'])
    
    # Check if the student has completed all modules
    if new_module_index >= total_modules:
        # Update the profile to mark completion
        student_profile['current_module_index'] = total_modules - 1
        student_profile['learning_completed'] = True
        student_profile['completion_date'] = datetime.datetime.now().isoformat()
        
        # Save the updated profile
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(student_profile, f, ensure_ascii=False, indent=4)
        
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
            'current_index': student_profile.get('current_module_index', 0),
            'total_modules': total_modules
        }), 400
    
    # Update the module index
    student_profile['current_module_index'] = new_module_index
    
    # Save the updated profile
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(student_profile, f, ensure_ascii=False, indent=4)
    
    return jsonify({
        'success': True,
        'new_index': new_module_index,
        'total_modules': total_modules,
        'finished': False
    })

@app.route('/summary')
def summary():
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
    # 讀取學生檔案
    with open(f'student_profiles/{session["student_id"]}.json', 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    # 讀取所有學習日誌
    logs = []
    for log_file in os.listdir('learning_logs'):
        if log_file.endswith('.json'):
            try:
                # 嘗試使用不同的編碼方式讀取文件
                try:
                    with open(os.path.join('learning_logs', log_file), 'r', encoding='utf-8') as lf:
                        log = json.load(lf)
                except UnicodeDecodeError:
                    with open(os.path.join('learning_logs', log_file), 'r', encoding='big5') as lf:
                        log = json.load(lf)
                
                if log['student_id'] == student_profile['id']:
                    logs.append(log)
            except Exception as e:
                app.logger.error(f"Error reading log file {log_file}: {str(e)}")
                continue
    
    # 按時間排序
    logs.sort(key=lambda x: x['timestamp'])
    return render_template('summary.html', student_profile=student_profile, logs=logs)

@app.route('/learning_path_discussion')
def learning_path_discussion():
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return redirect(url_for('select_user'))
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    # Check if learning path exists
    if not student_profile.get('learning_path'):
        return redirect(url_for('pretest'))
    
    # Check if learning path has already been confirmed
    if student_profile.get('learning_path_confirmed'):
        return redirect(url_for('learning'))
    
    return render_template('learning_path_discussion.html', 
                         learning_path=student_profile['learning_path'])

@app.route('/api/discuss-learning-path', methods=['POST'])
def discuss_learning_path():
    if 'student_id' not in session:
        return jsonify({'error': 'No student session found'}), 400
    
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return jsonify({'error': 'Student profile not found'}), 400
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    # Get learning path
    learning_path = student_profile.get('learning_path')
    if not learning_path:
        return jsonify({'error': 'No learning path found'}), 400
    
    # Create chat model
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash"
    )
    
    # Create prompt for discussion
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
            你是一位專業的學習路徑設計師，正在與學生討論他們的個人化學習路徑。
            你已經擁有學生的學習路徑（{learning_path}）和個人檔案（{profile}）。
            當學生提出修改需求時，請直接根據這些資訊自動修改學習路徑，只需用簡短的中文回覆說明修改了什麼，並在需要時加上 [ADJUST_PATH] 標記。
            禁止要求學生再提供任何資訊，也不要詢問學生「請提供 learning_path 或 profile」。
            """),
        HumanMessagePromptTemplate.from_template("學生訊息：{message}")
    ])
    
    # Create the chain
    discussion_chain = (
        RunnablePassthrough()
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    try:
        response = discussion_chain.invoke({
            "learning_path": json.dumps(learning_path, ensure_ascii=False),
            "profile": json.dumps({
                "name": student_profile['name'],
                "learning_style": student_profile['learning_style'],
                "current_knowledge_level": student_profile['current_knowledge_level']
            }, ensure_ascii=False),
            "message": message
        })
        
        # Check if the response indicates path adjustment is needed
        path_adjusted = False
        if "[ADJUST_PATH]" in response:
            # Remove the adjustment marker from the response
            response = response.replace("[ADJUST_PATH]", "").strip()
            
            # Generate new learning path with student's requirements
            new_learning_path = generate_learning_path(
                session['session_id'],
                StudentProfile.model_validate(student_profile),
                {
                    'score_percentage': student_profile.get('pretest_score', 0),
                    'knowledge_level': student_profile['current_knowledge_level'],
                    'student_requirements': message  # Add student's requirements to the test results
                }
            )
            
            # Update student profile with new learning path
            student_profile['learning_path'] = new_learning_path
            
            # Save updated profile
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(student_profile, f, ensure_ascii=False, indent=4)
            
            path_adjusted = True
        
        return jsonify({
            'response': response,
            'path_adjusted': path_adjusted
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    if 'student_id' not in session:
        return jsonify({'error': 'No student session found'}), 400
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return jsonify({'error': 'Student profile not found'}), 400
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    # Mark learning path as confirmed
    student_profile['learning_path_confirmed'] = True
    
    # Save updated profile
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(student_profile, f, ensure_ascii=False, indent=4)
    
    return jsonify({'success': True})

@app.route('/api/get-current-learning-path', methods=['GET'])
def get_current_learning_path():
    if 'student_id' not in session:
        return jsonify({'error': 'No student session found'}), 400
    
    # Load student profile
    profile_path = os.path.join('student_profiles', f"{session['student_id']}.json")
    if not os.path.exists(profile_path):
        return jsonify({'error': 'Student profile not found'}), 400
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    # Check if learning path exists
    if not student_profile.get('learning_path'):
        return jsonify({'error': 'No learning path found'}), 400
    
    return jsonify({
        'learning_path': student_profile['learning_path']
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

if __name__ == '__main__':
    app.run(debug=True)