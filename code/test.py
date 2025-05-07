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
    current_module_index: int = Field(default=0, description="Index of the current module in the learning path")

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
                data = json.load(f)
                # Backward compatibility: if current_module_index is missing, set to 0
                if "current_module_index" not in data:
                    data["current_module_index"] = 0
                return StudentProfile.parse_obj(data)
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
        learning_history=[],
        current_module_index=0
    )
    
    # Save the basic profile
    with open(f"student_profiles/{student_id}.json", "w") as f:
        f.write(profile.json(indent=4))
    
    return profile

# Create chains for different functionalities
def create_learning_style_survey(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an educational expert who specializes in assessing learning styles.
        Create a concise but effective learning style assessment with 5 multiple-choice questions.
        Each question should have 3 options that help identify whether the student is primarily:
        1. Visual learner
        2. Auditory learner
        3. Kinesthetic learner
        
        Format your response as a survey with numbered questions and lettered options."""),
        HumanMessagePromptTemplate.from_template("Create a learning style assessment.")
    ])
    
    return prompt | chat_model | StrOutputParser()

def create_pretest_generator(chat_model, retriever):
    # Define the prompt template for generating pre-test questions
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert educational assessment designer.
        Based on the provided content, create a pre-test with multiple-choice questions to assess the student's
        current knowledge level on the subject. 
        
        Generate questions that span different difficulty levels: easy, medium, and hard.
        For each question, provide:
        1. The question text
        2. Four multiple-choice options (A, B, C, D)
        3. The correct answer
        4. An explanation of why it's correct
        5. The difficulty level
        
        You must follow this exact JSON format:
        {
          "title": "Pre-Test on [Subject]",
          "description": "This test will assess your current knowledge of [Subject]",
          "questions": [
            {
              "question": "Question text here?",
              "choices": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
              "correct_answer": "A. Option A",
              "explanation": "Explanation of why A is correct",
              "difficulty": "easy"
            }
          ]
        }

        Generate 5 questions total with a mix of difficulty levels based on the content provided.
        """),
        HumanMessagePromptTemplate.from_template("""Generate a pre-test based on the following content:
        
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
        SystemMessage(content="""You are an expert educational curriculum designer specializing in personalized learning paths.
        Based on the provided student profile, test results, and content, create a personalized learning path.
        
        Your learning path should:
        1. Be tailored to the student's learning style, knowledge level, and interests
        2. Include clear learning objectives
        3. Break down the content into logical modules with estimated time commitments
        4. Include practice activities and resources appropriate for the student's learning style
        5. Follow scaffolding principles by gradually increasing difficulty and reducing support
        
        Your response must follow this exact JSON format:
        {
          "title": "Personalized Learning Path for [Subject]",
          "description": "This learning path is tailored to [name]'s learning style and current knowledge level",
          "objectives": ["objective 1", "objective 2", "objective 3"],
          "modules": [
            {
              "title": "Module 1: [Title]",
              "description": "Description of module",
              "activities": [
                {
                  "type": "reading",
                  "title": "Activity title",
                  "description": "Activity description",
                  "estimated_time": "20 minutes",
                  "difficulty": "beginner"
                }
              ],
              "resources": ["Resource 1", "Resource 2"],
              "assessment": "Description of module assessment"
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
        SystemMessage(content="""You are 'Study Buddy', a helpful peer AI that engages in friendly, productive discussions with students.
        Your role is to:
        1. Simulate a peer who is also learning the material but has certain insights
        2. Ask thoughtful questions that promote critical thinking
        3. Provide gentle guidance without directly giving answers
        4. Express thoughts in a conversational, student-like manner
        5. Use the Socratic method to help the student discover answers
        6. Be encouraging and positive
        
        Base your responses on the relevant content provided, but don't simply recite information.
        Instead, engage in a natural back-and-forth as if you're studying together.
        """),
        HumanMessagePromptTemplate.from_template("""The student wants to discuss this topic:
        
        Topic: {topic}
        
        Relevant content:
        {context}
        
        Student message:
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
        SystemMessage(content="""You are an expert educational assessment designer.
        Based on the provided learning module content and student's current knowledge level,
        create a post-test with multiple-choice questions to assess the student's learning.
        
        The difficulty level should match the student's current level:
        - For beginners: more easy questions (70%), some medium (30%)
        - For intermediate: some easy (30%), mostly medium (50%), some hard (20%)
        - For advanced: some medium (40%), mostly hard (60%)
        
        Generate questions that test understanding, application, and analysis of the content.
        
        For each question, provide:
        1. The question text
        2. Four multiple-choice options (A, B, C, D)
        3. The correct answer
        4. An explanation of why it's correct
        5. The difficulty level
        
        You must follow this exact JSON format:
        {
          "title": "Post-Test on [Subject]",
          "description": "This test will assess what you've learned about [Subject]",
          "questions": [
            {
              "question": "Question text here?",
              "choices": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
              "correct_answer": "A. Option A",
              "explanation": "Explanation of why A is correct",
              "difficulty": "easy"
            }
          ]
        }

        Generate 5 questions total with appropriate difficulty distribution based on the student's level.
        """),
        HumanMessagePromptTemplate.from_template("""Generate a post-test based on:
        
        Student's current knowledge level: {knowledge_level}
        Module content: {context}
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
        SystemMessage(content="""You are a reflective learning coach who helps students create meaningful learning logs.
        Based on the student's completed learning module and test results, guide them to reflect on their learning.
        
        Ask thoughtful, open-ended questions to prompt reflection on:
        1. What they learned (key concepts and insights)
        2. How they felt about the learning process
        3. What they found challenging
        4. What questions they still have
        5. How they might apply what they learned
        6. What they want to learn next
        
        Your goal is to help the student create a rich, reflective learning log that will be valuable for their growth.
        """),
        HumanMessagePromptTemplate.from_template("""Help the student create a learning log reflection based on:
        
        Module completed: {module_title}
        
        Module content summary: {module_summary}
        
        Test results: {test_results}
        """)
    ])
    
    learning_log_chain = prompt | chat_model | StrOutputParser()
    
    return learning_log_chain

def create_learning_log_analyzer(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert educational analyst who specializes in analyzing student learning logs.
        Based on the student's learning log, assess:
        
        1. Level of understanding of key concepts
        2. Areas of strength and confidence
        3. Areas of confusion or misconception
        4. Emotional response to the material
        5. Learning style indicators
        6. Potential next steps for learning
        
        Format your response as a JSON object following this exact structure:
        {
          "understanding_level": "High/Medium/Low",
          "strengths": ["strength 1", "strength 2"],
          "areas_for_improvement": ["area 1", "area 2"],
          "emotional_response": "description of emotional response",
          "learning_style_indicators": ["indicator 1", "indicator 2"],
          "recommended_next_steps": ["recommendation 1", "recommendation 2"],
          "suggested_resources": ["resource 1", "resource 2"]
        }
        """),
        HumanMessagePromptTemplate.from_template("""Analyze the following learning log:
        
        Student: {student_name}
        Topic: {topic}
        Learning Log Content:
        {log_content}
        """)
    ])
    
    learning_log_analysis_chain = prompt | chat_model | StrOutputParser()
    
    return learning_log_analysis_chain

def create_knowledge_level_assessor(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert in educational assessment.
        Based on the student's test results, determine their knowledge level for this specific topic.
        
        Consider:
        1. The number of correct answers
        2. The difficulty of questions answered correctly
        3. The pattern of answers (consistent understanding vs. gaps)
        
        Categorize the student's knowledge level as:
        - Beginner: Basic familiarity, understands simple concepts
        - Intermediate: Good understanding of core concepts, some application ability
        - Advanced: Deep understanding, can apply concepts to novel situations
        
        Provide a brief justification for your assessment.
        
        Format your response as a JSON object:
        {
          "knowledge_level": "beginner/intermediate/advanced",
          "justification": "Brief explanation of your assessment",
          "strengths": ["strength 1", "strength 2"],
          "areas_for_improvement": ["area 1", "area 2"],
          "recommended_focus": "What the student should focus on next"
        }
        """),
        HumanMessagePromptTemplate.from_template("""Assess the student's knowledge level based on these test results:
        
        Test: {test_title}
        
        Questions and Answers:
        {test_results}
        """)
    ])
    
    knowledge_level_chain = prompt | chat_model | StrOutputParser()
    
    return knowledge_level_chain

def create_module_content_generator(chat_model, retriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert educational content creator.
        Based on the provided module topic and student's learning style and knowledge level,
        create engaging educational content for the module.
        
        Your content should:
        1. Be tailored to the student's learning style (visual, auditory, or kinesthetic)
        2. Be appropriate for the student's knowledge level
        3. Include clear explanations of key concepts
        4. Use examples and analogies to illustrate points
        5. Include scaffolding elements appropriate for the knowledge level
        6. Be structured with clear sections and headings
        7. End with a brief summary of key points
        
        Format your content using markdown for better readability.
        """),
        HumanMessagePromptTemplate.from_template("""Create educational content for:
        
        Module Topic: {module_topic}
        Student Learning Style: {learning_style}
        Student Knowledge Level: {knowledge_level}
        
        Relevant source material:
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
    console.print("\n[bold cyan]===== Learning Style Assessment =====[/bold cyan]")
    
    survey_chain = create_learning_style_survey(chat_model)
    survey = survey_chain.invoke({})
    
    console.print(Markdown(survey))
    console.print("\n[bold yellow]Please answer each question to help determine your learning style.[/bold yellow]")
    
    # Process survey results
    console.print("\n[bold]After completing the survey, which learning style seems most accurate?[/bold]")
    console.print("1. Visual Learner")
    console.print("2. Auditory Learner")
    console.print("3. Kinesthetic Learner")
    
    style_choice = Prompt.ask("Select your primary learning style", choices=["1", "2", "3"])
    
    learning_styles = {
        "1": "visual",
        "2": "auditory",
        "3": "kinesthetic"
    }
    
    student_profile.learning_style = learning_styles[style_choice]
    
    # Save updated profile
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return student_profile

def administer_pretest(chat_model, retriever, student_profile):
    console.print("\n[bold cyan]===== Pre-Test =====[/bold cyan]")
    console.print("[yellow]Generating a pre-test based on the content...[/yellow]")
    
    pretest_chain = create_pretest_generator(chat_model, retriever)
    pretest_json = pretest_chain.invoke({})
    
    try:
        pretest = json.loads(pretest_json)
    except json.JSONDecodeError:
        console.print("[bold red]Error parsing pre-test JSON. Using fallback method.[/bold red]")
        # Extract JSON from the text if it's surrounded by additional text
        import re
        json_match = re.search(r'({[\s\S]*})', pretest_json)
        if json_match:
            try:
                pretest = json.loads(json_match.group(1))
            except:
                console.print("[bold red]Failed to parse pre-test JSON. Using default test.[/bold red]")
                pretest = {
                    "title": "Default Pre-Test",
                    "description": "This is a fallback test due to parsing issues.",
                    "questions": [
                        {
                            "question": "What is the main purpose of RAG in educational systems?",
                            "choices": ["A. To generate random questions", "B. To retrieve relevant information for generation", 
                                       "C. To replace teachers", "D. To play music during learning"],
                            "correct_answer": "B. To retrieve relevant information for generation",
                            "explanation": "RAG (Retrieval-Augmented Generation) helps with accurate information retrieval.",
                            "difficulty": "medium"
                        }
                    ]
                }
    
    console.print(f"\n[bold green]{pretest['title']}[/bold green]")
    console.print(f"[italic]{pretest['description']}[/italic]\n")
    
    # Administer the test
    score = 0
    total_questions = len(pretest['questions'])
    results = []
    
    for i, q in enumerate(pretest['questions']):
        console.print(f"\n[bold]Question {i+1}:[/bold] {q['question']}")
        for choice in q['choices']:
            console.print(f"  {choice}")
        
        answer = Prompt.ask("\nYour answer (A, B, C, or D)").upper()
        correct_letter = q['correct_answer'][0].upper()
        
        if answer == correct_letter:
            score += 1
            console.print("[bold green]Correct![/bold green]")
        else:
            console.print(f"[bold red]Incorrect. The correct answer is {q['correct_answer']}[/bold red]")
        
        console.print(f"[italic]{q['explanation']}[/italic]")
        
        results.append({
            "question": q['question'],
            "student_answer": answer,
            "correct_answer": q['correct_answer'],
            "is_correct": answer == correct_letter,
            "difficulty": q['difficulty']
        })
    
    # Calculate score
    percentage = (score / total_questions) * 100
    console.print(f"\n[bold]Test complete! Your score: {score}/{total_questions} ({percentage:.1f}%)[/bold]")
    
    # Assess knowledge level
    if percentage >= 80:
        knowledge_level = "advanced"
    elif percentage >= 50:
        knowledge_level = "intermediate"
    else:
        knowledge_level = "beginner"
    
    console.print(f"[yellow]Based on your pre-test results, your current knowledge level is: [bold]{knowledge_level}[/bold][/yellow]")
    
    # Update student profile
    student_profile.current_knowledge_level = knowledge_level
    student_profile.learning_history.append({
        "activity_type": "pre-test",
        "timestamp": datetime.datetime.now().isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "knowledge_level": knowledge_level
    })
    
    # Save updated profile
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return pretest, results, knowledge_level

def generate_learning_path(chat_model, retriever, student_profile, pretest_results):
    console.print("\n[bold cyan]===== Generating Personalized Learning Path =====[/bold cyan]")
    console.print("[yellow]Creating a learning path tailored to your profile and test results...[/yellow]")
    
    # Format the test results for the chain
    test_results_formatted = json.dumps({
        "score_percentage": (sum(1 for r in pretest_results if r["is_correct"]) / len(pretest_results)) * 100,
        "knowledge_level": student_profile.current_knowledge_level,
        "strengths": [r["question"] for r in pretest_results if r["is_correct"]],
        "weaknesses": [r["question"] for r in pretest_results if not r["is_correct"]]
    })
    
    # Format the profile for the chain
    profile_formatted = json.dumps({
        "name": student_profile.name,
        "learning_style": student_profile.learning_style,
        "current_knowledge_level": student_profile.current_knowledge_level,
        "interests": student_profile.interests
    })
    
    learning_path_chain = create_learning_path_generator(chat_model, retriever)
    learning_path_json = learning_path_chain.invoke(profile_formatted, test_results_formatted)
    
    try:
        learning_path = json.loads(learning_path_json)
    except json.JSONDecodeError:
        console.print("[bold red]Error parsing learning path JSON. Using fallback method.[/bold red]")
        # Extract JSON from the text if it's surrounded by additional text
        import re
        json_match = re.search(r'({[\s\S]*})', learning_path_json)
        if json_match:
            try:
                learning_path = json.loads(json_match.group(1))
            except:
                console.print("[bold red]Failed to parse learning path JSON. Using default path.[/bold red]")
                learning_path = {
                    "title": f"Default Learning Path for {student_profile.name}",
                    "description": f"This learning path is tailored to {student_profile.name}'s {student_profile.learning_style} learning style and {student_profile.current_knowledge_level} knowledge level",
                    "objectives": ["Learn core concepts", "Build practical skills", "Prepare for assessment"],
                    "modules": [
                        {
                            "title": "Module 1: Introduction to Key Concepts",
                            "description": "An overview of fundamental concepts in the subject area",
                            "activities": [
                                {
                                    "type": "reading",
                                    "title": "Introduction to Core Concepts",
                                    "description": "Reading material covering the basics",
                                    "estimated_time": "20 minutes",
                                    "difficulty": student_profile.current_knowledge_level
                                }
                            ],
                            "resources": ["Main course material"],
                            "assessment": "Short quiz on basic concepts"
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
    
    # Allow time for reading
    console.print("\n[yellow]Take your time to read and understand the content.[/yellow]")
    Prompt.ask("\nPress Enter when you're ready to continue")
    
    return content

def engage_peer_discussion(chat_model, retriever, topic):
    console.print(f"\n[bold cyan]===== Peer Discussion: {topic} =====[/bold cyan]")
    console.print("[yellow]Meet your AI study buddy! Ask questions or discuss the topic to deepen your understanding.[/yellow]")
    
    discussion_chain = create_peer_discussion_ai(chat_model, retriever)
    
    console.print("\n[bold green]Study Buddy:[/bold green] Hey there! I've been studying this topic too. What aspects would you like to discuss or any questions you have about it?")
    
    conversation_history = []
    
    while True:
        user_message = Prompt.ask("\n[bold blue]You[/bold blue]")
        
        if user_message.lower() in ["exit", "quit", "bye", "end"]:
            console.print("\n[bold green]Study Buddy:[/bold green] Great discussion! Let me know if you want to chat more later.")
            break
        
        response = discussion_chain.invoke(topic, user_message)
        console.print(f"\n[bold green]Study Buddy:[/bold green] {response}")
        
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": response})
        
        if len(conversation_history) >= 10:  # Limit discussion length
            console.print("\n[yellow]We've had a good discussion. Ready to move on?[/yellow]")
            if Confirm.ask("End discussion?"):
                break
    
    return conversation_history

def administer_posttest(chat_model, retriever, module, student_profile):
    console.print(f"\n[bold cyan]===== Post-Test: {module['title']} =====[/bold cyan]")
    console.print("[yellow]Testing your understanding of the module content...[/yellow]")
    
    # Get module topic
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    
    # Generate post-test questions based on module content and student level
    posttest_chain = create_posttest_generator(chat_model, retriever)
    posttest_json = posttest_chain.invoke(student_profile.current_knowledge_level, module_topic)
    
    try:
        posttest = json.loads(posttest_json)
    except json.JSONDecodeError:
        console.print("[bold red]Error parsing post-test JSON. Using fallback method.[/bold red]")
        # Extract JSON from the text if it's surrounded by additional text
        import re
        json_match = re.search(r'({[\s\S]*})', posttest_json)
        if json_match:
            try:
                posttest = json.loads(json_match.group(1))
            except:
                console.print("[bold red]Failed to parse post-test JSON. Using default test.[/bold red]")
                posttest = {
                    "title": f"Post-Test on {module_topic}",
                    "description": f"This test will assess what you've learned about {module_topic}",
                    "questions": [
                        {
                            "question": f"What is a key concept in {module_topic}?",
                            "choices": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
                            "correct_answer": "A. Option A",
                            "explanation": "This is the correct answer based on the module content.",
                            "difficulty": "medium"
                        }
                    ]
                }
    
    console.print(f"\n[bold green]{posttest['title']}[/bold green]")
    console.print(f"[italic]{posttest['description']}[/italic]\n")
    
    # Administer the test
    score = 0
    total_questions = len(posttest['questions'])
    results = []
    
    for i, q in enumerate(posttest['questions']):
        console.print(f"\n[bold]Question {i+1}:[/bold] {q['question']}")
        for choice in q['choices']:
            console.print(f"  {choice}")
        
        answer = Prompt.ask("\nYour answer (A, B, C, or D)").upper()
        correct_letter = q['correct_answer'][0].upper()
        
        if answer == correct_letter:
            score += 1
            console.print("[bold green]Correct![/bold green]")
        else:
            console.print(f"[bold red]Incorrect. The correct answer is {q['correct_answer']}[/bold red]")
        
        console.print(f"[italic]{q['explanation']}[/italic]")
        
        results.append({
            "question": q['question'],
            "student_answer": answer,
            "correct_answer": q['correct_answer'],
            "is_correct": answer == correct_letter,
            "difficulty": q['difficulty']
        })
    
    # Calculate score
    percentage = (score / total_questions) * 100
    console.print(f"\n[bold]Test complete! Your score: {score}/{total_questions} ({percentage:.1f}%)[/bold]")
    
    # Assess progress and adjust knowledge level if needed
    previous_level = student_profile.current_knowledge_level
    
    if percentage >= 80 and previous_level != "advanced":
        if previous_level == "beginner":
            new_level = "intermediate"
        else:
            new_level = "advanced"
        console.print(f"[bold green]Great progress! Your knowledge level has increased from {previous_level} to {new_level}.[/bold green]")
        student_profile.current_knowledge_level = new_level
    elif percentage < 50 and previous_level != "beginner":
        if previous_level == "advanced":
            new_level = "intermediate"
        else:
            new_level = "beginner"
        console.print(f"[yellow]You may need more practice. Your knowledge level has been adjusted from {previous_level} to {new_level}.[/yellow]")
        student_profile.current_knowledge_level = new_level
    else:
        console.print(f"[yellow]Your knowledge level remains at {previous_level}.[/yellow]")
    
    # Update student profile
    student_profile.learning_history.append({
        "activity_type": "post-test",
        "module": module['title'],
        "timestamp": datetime.datetime.now().isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "previous_level": previous_level,
        "current_level": student_profile.current_knowledge_level
    })
    
    # Save updated profile
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return posttest, results

def create_learning_log(chat_model, module, test_results, student_profile):
    console.print(f"\n[bold cyan]===== Learning Log: {module['title']} =====[/bold cyan]")
    
    # Get module topic and summary
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    module_summary = module['description']
    
    # Format test results
    test_results_str = f"Score: {sum(1 for r in test_results if r['is_correct'])}/{len(test_results)} questions correct\n"
    test_results_str += "Strengths: " + ", ".join([r['question'] for r in test_results if r['is_correct']])
    test_results_str += "\nAreas for improvement: " + ", ".join([r['question'] for r in test_results if not r['is_correct']])
    
    # Generate reflection prompts
    learning_log_chain = create_learning_log_prompter(chat_model)
    reflection_prompts = learning_log_chain.invoke({
        "module_title": module_topic,
        "module_summary": module_summary,
        "test_results": test_results_str
    })
    
    console.print(Markdown(reflection_prompts))
    
    # Get student reflections
    console.print("\n[bold yellow]Please write your learning log reflection based on the prompts above:[/bold yellow]")
    log_content = ""
    
    while True:
        line = input()
        if line.lower() == "done":
            break
        log_content += line + "\n"
    
    # Create and save learning log
    log_id = str(uuid.uuid4())[:8]
    log = LearningLog(
        id=log_id,
        student_id=student_profile.id,
        timestamp=datetime.datetime.now().isoformat(),
        topic=module_topic,
        content=log_content,
        reflections=[],  # Will be populated by analysis
        questions=[],    # Will be populated by analysis
        next_steps=[]    # Will be populated by analysis
    )
    
    # Analyze the learning log
    analyzer_chain = create_learning_log_analyzer(chat_model)
    analysis_json = analyzer_chain.invoke({
        "student_name": student_profile.name,
        "topic": module_topic,
        "log_content": log_content
    })
    
    try:
        analysis = json.loads(analysis_json)
        
        # Update student profile based on analysis
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
        
        # Display analysis summary
        console.print("\n[bold green]Learning Log Analysis:[/bold green]")
        console.print(f"[bold]Understanding Level:[/bold] {analysis.get('understanding_level', 'Not determined')}")
        
        console.print("\n[bold]Strengths:[/bold]")
        for strength in analysis.get("strengths", []):
            console.print(f"? {strength}")
        
        console.print("\n[bold]Areas for Improvement:[/bold]")
        for area in analysis.get("areas_for_improvement", []):
            console.print(f"? {area}")
        
        console.print("\n[bold]Recommended Next Steps:[/bold]")
        for step in analysis.get("recommended_next_steps", []):
            console.print(f"? {step}")
    
    except (json.JSONDecodeError, KeyError):
        console.print("[bold red]Error analyzing learning log. Saving raw log only.[/bold red]")
    
    # Save the learning log
    with open(f"learning_logs/{log_id}.json", "w") as f:
        f.write(log.json(indent=4))
    
    # Update and save student profile
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return log

# Main application function
def main():
    console.print("[bold cyan]=====================================[/bold cyan]")
    console.print("[bold cyan]== RAG Scaffolding Education System ==[/bold cyan]")
    console.print("[bold cyan]=====================================[/bold cyan]\n")
    
    # Initialize models and RAG system
    console.print("[yellow]Initializing system...[/yellow]")
    chat_model, embedding = initialize_models()
    retriever = initialize_rag_system(embedding)
    
    # Manage student profile
    student_profile = manage_student_profile()
    console.print(f"\n[bold green]Welcome, {student_profile.name}![/bold green]")
    
    # Conduct learning style survey if not already done
    if not student_profile.learning_style:
        student_profile = conduct_learning_style_survey(chat_model, student_profile)
    else:
        console.print(f"\n[yellow]Your learning style is: [bold]{student_profile.learning_style}[/bold][/yellow]")
    
    # Administer pre-test
    pretest, pretest_results, knowledge_level = administer_pretest(chat_model, retriever, student_profile)
    
    # Generate personalized learning path
    learning_path = generate_learning_path(chat_model, retriever, student_profile, pretest_results)
    
    # Learning process
    modules = learning_path['modules']
    module_count = len(modules)
    # Start from saved progress
    start_index = getattr(student_profile, 'current_module_index', 0)
    for module_index in range(start_index, module_count):
        module = modules[module_index]
        console.print(f"\n[bold cyan]===== Starting Module {module_index + 1}/{module_count} =====[/bold cyan]")
        
        # Ask if user wants to proceed with this module
        proceed = Confirm.ask(f"Ready to start module: {module['title']}?")
        if not proceed:
            break
        
        # Deliver module content
        module_content = deliver_module_content(chat_model, retriever, student_profile, module)
        
        # Engage in peer discussion
        module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
        discussion_history = engage_peer_discussion(chat_model, retriever, module_topic)
        
        # Administer post-test
        posttest, posttest_results = administer_posttest(chat_model, retriever, module, student_profile)
        
        # Create learning log
        learning_log = create_learning_log(chat_model, module, posttest_results, student_profile)
        
        # Update and save progress after each module
        student_profile.current_module_index = module_index + 1
        with open(f"student_profiles/{student_profile.id}.json", "w") as f:
            f.write(student_profile.json(indent=4))
        
        # Ask if user wants to continue to next module
        if module_index < module_count - 1:
            continue_learning = Confirm.ask("Continue to the next module?")
            if not continue_learning:
                break
    
    console.print("\n[bold green]===== Learning Path Complete =====[/bold green]")
    console.print("[yellow]Thank you for using the RAG Scaffolding Education System![/yellow]")
    console.print(f"[yellow]Your current knowledge level: [bold]{student_profile.current_knowledge_level}[/bold][/yellow]")
    
    # Summarize learning progress
    console.print("\n[bold]Your Learning Journey Summary:[/bold]")
    console.print(f"? Completed {len(student_profile.learning_history)} learning activities")
    console.print(f"? Current strengths: {', '.join(student_profile.strengths) if student_profile.strengths else 'Not yet determined'}")
    console.print(f"? Areas for improvement: {', '.join(student_profile.areas_for_improvement) if student_profile.areas_for_improvement else 'Not yet determined'}")
    
    console.print("\n[bold]Suggestions for Continued Learning:[/bold]")
    if student_profile.current_knowledge_level == "beginner":
        console.print("? Focus on mastering foundational concepts")
        console.print("? Practice with more beginner to intermediate level examples")
    elif student_profile.current_knowledge_level == "intermediate":
        console.print("? Deepen your understanding of complex topics")
        console.print("? Begin applying concepts to real-world problems")
    else:  # advanced
        console.print("? Explore specialized topics in the field")
        console.print("? Consider teaching or mentoring others to solidify your knowledge")

if __name__ == "__main__":
    main()