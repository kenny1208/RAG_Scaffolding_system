from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
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
from chinese.test import (
    initialize_models, initialize_rag_system, manage_student_profile,
    conduct_learning_style_survey, administer_pretest, generate_learning_path,
    deliver_module_content, engage_peer_discussion, administer_posttest,
    create_learning_log
)

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
    # Load PDFs
    all_pages = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        all_pages.extend(pages)
    
    # Split text into chunks
    text_splitter = NLTKTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages)
    
    # Create embeddings and vectorstore
    embedding = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    
    # Create a persistent Chroma DB with the session ID as the collection name
    persist_directory = os.path.join(app.config['VECTOR_DB_DIR'], session_id)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=session_id
    )
    vectorstore.persist()  # Save to disk
    
    # Create summary
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
    
    return chunks, summary

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
        SystemMessage(content="""You are a helpful assistant that answers questions based on the provided context.
        You will be given a context and a question. Provide a concise answer based on the context."""),
        HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
        Context: {context}
        Question: {question}
        Answer: """)
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
        profile_path = os.path.join('chinese/student_profiles', f"{session['student_id']}.json")
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
        current_module_index=0
    )
    
    # Save the profile
    session['student_id'] = student_id
    session['session_id'] = str(uuid.uuid4())
    with open(os.path.join('chinese/student_profiles', f"{student_id}.json"), 'w', encoding='utf-8') as f:
        f.write(profile.model_dump_json(indent=4))
    
    return profile

def save_student_profile(profile):
    """Save a student profile to disk"""
    profile_path = os.path.join('chinese/student_profiles', f"{profile.id}.json")
    with open(profile_path, 'w', encoding='utf-8') as f:
        f.write(profile.model_dump_json(indent=4))

def create_learning_style_survey():
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
        SystemMessage(content="""You are an expert in educational assessment design.
        Based on the provided content, design a pre-test to assess the student's existing knowledge level on the topic.
        
        Design questions covering different difficulty levels: easy, medium, and hard.
        For each question, provide:
        1. Question text
        2. Four multiple-choice options (A, B, C, D)
        3. Correct answer
        4. Explanation of why it's correct
        5. Difficulty level
        
        You must follow this exact JSON format:
        {
          "title": "Pre-test: [Topic]",
          "description": "This test will assess your existing knowledge of [Topic]",
          "questions": [
            {
              "question": "Question text?",
              "choices": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
              "correct_answer": "A. Option A",
              "explanation": "Explanation of why A is correct",
              "difficulty": "easy"
            }
          ]
        }

        Generate a total of 5 questions including different difficulty levels based on the provided content.
        """),
        HumanMessagePromptTemplate.from_template("""Generate a pre-test based on the following content:
        
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
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert in personalized learning path design.
        Based on the provided student profile, test results, and content, create a learning path suitable for self-study.
        
        Your learning path should:
        1. Be tailored to the student's learning style, knowledge level, and interests
        2. Include clear learning objectives
        3. Follow the scaffolding principle, gradually increasing difficulty while reducing support
        
        Your response must follow this exact JSON format:
        {
          "title": "Personalized Learning Path for [Topic]",
          "description": "This learning path is tailored to [name]'s learning style and current knowledge level",
          "objectives": ["Objective 1", "Objective 2", "Objective 3"],
          "modules": [
            {
              "title": "Module 1: [Title]",
              "description": "Module description",
              "activities": [
                {
                  "title": "Activity title",
                  "description": "Activity description",
                  "difficulty": "beginner"
                }
              ],
              "resources": ["Handout 1-1", "Handout 1-2"]
            }
          ]
        }
        """),
        HumanMessagePromptTemplate.from_template("""Generate a personalized learning path based on:
        
        Student Profile:
        {profile}
        
        Test Results:
        {test_results}
        
        Content:
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
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a professional educational content creator.
        Based on the module topic and the student's learning style and knowledge level, create engaging educational content.
        
        Your content should:
        1. Be tailored to the student's learning style (visual, auditory, or kinesthetic)
        2. Be appropriate for the student's knowledge level
        3. Include clear explanations of key concepts
        4. Use examples and analogies to illustrate points
        5. Include scaffolding elements appropriate to the knowledge level
        6. Be well-structured with clear sections and headings
        7. End with a brief summary of key points
        
        Format your content using markdown for readability.
        """),
        HumanMessagePromptTemplate.from_template("""Create educational content for:
        
        Module Topic: {module_topic}
        Student Learning Style: {learning_style}
        Student Knowledge Level: {knowledge_level}
        
        Relevant Source Material:
        {context}
        """)
    ])
    
    # Get module topic
    module_topic = module["title"].split(": ", 1)[1] if ": " in module["title"] else module["title"]
    
    # Create the chain
    content_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "module_topic": module_topic,
            "learning_style": profile.learning_style,
            "knowledge_level": profile.current_knowledge_level,
            "context": "\n\n".join([d.page_content for d in docs])
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
        SystemMessage(content="""You are a 'Learning Partner', a friendly and helpful AI peer who engages in constructive discussions with students.
        Your role is to:
        1. Simulate a peer who is also studying the topic but has some insights
        2. Ask thoughtful questions that promote critical thinking
        3. Provide gentle guidance rather than direct answers
        4. Express ideas conversationally, as one student to another
        5. Use Socratic questioning to help students discover answers
        6. Be encouraging and positive
        
        Respond based on the provided relevant content, but don't simply recite information.
        Instead, engage in a natural back-and-forth as if learning together.
        """),
        HumanMessagePromptTemplate.from_template("""The student wants to discuss this topic:
        
        Topic: {topic}
        
        Relevant content:
        {context}
        
        Student message:
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
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a professional educational assessment designer.
        Based on the module content and the student's current knowledge level, design a post-test with multiple-choice questions to assess the student's learning outcomes.
        
        The difficulty should match the student's current level:
        - Beginner: More easy questions (70%), some medium questions (30%)
        - Intermediate: Some easy questions (30%), mostly medium questions (50%), some hard questions (20%)
        - Advanced: Some medium questions (40%), mostly hard questions (60%)
        
        The questions should test the student's understanding, application, and analysis of the content.
        
        For each question, provide:
        1. Question text
        2. Four multiple-choice options (A, B, C, D)
        3. Correct answer
        4. Explanation of why it's correct
        5. Difficulty level
        
        You must follow this exact JSON format:
        {
          "title": "Post-test: [Topic]",
          "description": "This test will assess your learning outcomes for [Topic]",
          "questions": [
            {
              "question": "Question text?",
              "choices": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
              "correct_answer": "A. Option A",
              "explanation": "Explanation of why A is correct",
              "difficulty": "easy"
            }
          ]
        }

        Generate a total of 5 questions with appropriate difficulty distribution based on the student's level.
        """),
        HumanMessagePromptTemplate.from_template("""Generate a post-test based on:
        
        Student's Current Knowledge Level: {knowledge_level}
        Module Content: {context}
        """)
    ])
    
    # Create the chain
    posttest_chain = (
        RunnablePassthrough()
        | retriever
        | (lambda docs: {
            "knowledge_level": knowledge_level,
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
        SystemMessage(content="""You are a professional educational analyst specializing in analyzing student learning logs.
        Based on the student's learning log, assess:
        
        1. Level of understanding of key concepts
        2. Areas of strength and confidence
        3. Areas of confusion or misconception
        4. Emotional response to the material
        5. Indicators of learning style
        
        Format your response as the following exact JSON structure:
        {
          "understanding_level": "high/medium/low",
          "strengths": ["Strength 1", "Strength 2"],
          "areas_for_improvement": ["Area 1", "Area 2"],
          "emotional_response": "Description of emotional response",
          "learning_style_indicators": ["Indicator 1", "Indicator 2"],
          "recommended_next_steps": ["Suggested step 1", "Suggested step 2"],
          "suggested_resources": ["Resource 1", "Resource 2"]
        }
        """),
        HumanMessagePromptTemplate.from_template("""Analyze the following learning log:
        
        Student: {student_name}
        Topic: {topic}
        Learning Log Content:
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
    profile_dir = 'chinese/student_profiles'
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
            'current_module_index': 0
        }
        
        # Save profile
        os.makedirs('chinese/student_profiles', exist_ok=True)
        with open(f'chinese/student_profiles/{student_id}.json', 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=4)
        
        session['student_id'] = student_id
        return redirect(url_for('upload_pdf'))
    
    return render_template('create_user.html')

@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    if 'student_id' not in session:
        return redirect(url_for('select_user'))
    
    if request.method == 'POST':
        if 'pdfs' not in request.files:
            flash('No files selected')
            return redirect(request.url)
        
        files = request.files.getlist('pdfs')
        if not files or files[0].filename == '':
            flash('No files selected')
            return redirect(request.url)
        
        # Generate a unique session ID for this upload
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Create session directory
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        file_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_dir, filename)
                file.save(file_path)
                file_paths.append(file_path)
        
        if not file_paths:
            flash('No valid PDF files uploaded')
            return redirect(request.url)
        
        try:
            # Process all PDFs and create vector store
            chunks, summary = process_pdfs(file_paths, session_id)
            
            # Store file paths in session
            session['pdf_files'] = [os.path.basename(path) for path in file_paths]
            
            # Redirect to learning style survey
            return redirect(url_for('learning_style_survey'))
            
        except Exception as e:
            flash(f'Error processing files: {str(e)}')
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
        profile_path = os.path.join('chinese/student_profiles', f"{session['student_id']}.json")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
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
                'current_module_index': 0
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
        save_student_profile(profile)
        
        # Redirect to learning interface (return JSON for fetch)
        return jsonify({'redirect': url_for('learning')})
    
    # Generate pretest
    pretest_data = create_pretest(session['session_id'])
    session['pretest'] = pretest_data
    return render_template('pretest.html', pretest=pretest_data)

@app.route('/learning', methods=['GET', 'POST'])
def learning():
    if 'student_id' not in session or 'session_id' not in session:
        return redirect(url_for('select_user'))
    
    # Load student profile
    with open(f'chinese/student_profiles/{session["student_id"]}.json', 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    
    # Start the learning process
    if not student_profile['learning_style']:
        # Conduct learning style survey
        return redirect(url_for('learning_style_survey'))
    
    if not student_profile.get('learning_path'):
        return redirect(url_for('pretest'))
    
    # Get current module
    current_module_index = student_profile['current_module_index']
    current_module = student_profile['learning_path']['modules'][current_module_index]
    
    # Generate module content if not already present
    if 'content' not in current_module:
        try:
            # 轉成 StudentProfile 物件
            profile_obj = StudentProfile.model_validate(student_profile)
            content = generate_module_content(session['session_id'], current_module, profile_obj)
            current_module['content'] = content
            # Save the updated profile with content
            with open(f'chinese/student_profiles/{session["student_id"]}.json', 'w', encoding='utf-8') as f:
                json.dump(student_profile, f, ensure_ascii=False, indent=4)
        except Exception as e:
            current_module['content'] = f"Error generating content: {str(e)}"
    
    return render_template('learning.html', 
                         student_profile=student_profile,
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
        survey = create_learning_style_survey()
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
    profile_path = os.path.join('chinese/student_profiles', f"{session['student_id']}.json")
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
    profile_path = os.path.join('chinese/student_profiles', f"{session['student_id']}.json")
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
        
        # Save the profile
        save_student_profile(profile)
        
        # Save the learning log
        with open(os.path.join(app.config['LEARNING_LOGS_DIR'], f"{log_id}.json"), 'w') as f:
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
    profile_path = os.path.join('chinese/student_profiles', f"{session['student_id']}.json")
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
    with open(f'chinese/student_profiles/{session["student_id"]}.json', 'r', encoding='utf-8') as f:
        student_profile = json.load(f)
    # 讀取所有學習日誌
    logs = []
    for log_file in os.listdir('learning_logs'):
        if log_file.endswith('.json'):
            with open(os.path.join('learning_logs', log_file), 'r', encoding='utf-8') as lf:
                log = json.load(lf)
                if log['student_id'] == student_profile['id']:
                    logs.append(log)
    # 按時間排序
    logs.sort(key=lambda x: x['timestamp'])
    return render_template('summary.html', student_profile=student_profile, logs=logs)

if __name__ == '__main__':
    app.run(debug=True)