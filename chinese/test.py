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
        SystemMessage(content="""�z�O�@��M���ǲ߭���������Ш|�M�a�C
        �г]�p�@��²������Ī��ǲ߭�������ݨ��A�]�t 5 �Ӧh���D�C
        �C�Ӱ��D���� 3 �ӿﶵ�A�Ω�P�_�ǥͬO�_�D�n�O�G
        1. ��ı���ǲߪ�
        2. ťı���ǲߪ�
        3. ��ı���ǲߪ�
        
        �бN�z���^���榡�Ƭ��@���ݨ��A�]�t�s�������D�M�r���аO���ﶵ�C"""),
        HumanMessagePromptTemplate.from_template("�]�p�@���ǲ߭�������ݨ��C")
    ])
    
    return prompt | chat_model | StrOutputParser()

def create_pretest_generator(chat_model, retriever):
    # Define the prompt template for generating pre-test questions
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M���Ш|�����]�p���M�a�C
        �ھڴ��Ѫ����e�A�]�p�@���e���]Pre-Test�^�A�H�����ǥͦb�ӥD�D�W���{�����Ѥ����C
        
        �г]�p�[�\���P���ׯŧO�����D�G²��B�����M�x���C
        ���C�Ӱ��D�A�д��ѡG
        1. ���D�奻
        2. �|�Ӧh��ﶵ�]A, B, C, D�^
        3. ���T����
        4. �����򥿽T������
        5. ���ׯŧO
        
        �z������`�H�U��T�� JSON �榡�G
        {
          "title": "�e���G[�D�D]",
          "description": "������N�����z��[�D�D]���{������",
          "questions": [
            {
              "question": "���D�奻�H",
              "choices": ["A. �ﶵ A", "B. �ﶵ B", "C. �ﶵ C", "D. �ﶵ D"],
              "correct_answer": "A. �ﶵ A",
              "explanation": "������ A �O���T���ת�����",
              "difficulty": "²��"
            }
          ]
        }

        �Юھڴ��Ѫ����e�ͦ��`�@ 5 �Ӱ��D�A�å]�t���P���ׯŧO�����D�C
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���e�ͦ��@���e���G
        
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
        SystemMessage(content="""�z�O�@��M���ӤH�ƾǲ߸��|�]�p���Ш|�ҵ{�]�p�M�a�C
        �ھڴ��Ѫ��ǥ��ɮסB���絲�G�M���e�A�Ыؤ@���ӤH�ƪ��ǲ߸��|�C
        
        �z���ǲ߸��|���ӡG
        1. �w��ǥͪ��ǲ߭���B���Ѥ����M����i��q���w��
        2. �]�t�M�����ǲߥؼ�
        3. �N���e���Ѭ��޿�ҲաA�æ���һݮɶ�
        4. �]�A�A�X�ǥ;ǲ߭��檺�m�߬��ʩM�귽
        5. ��`�N�[��h�A�v�B�W�[���רô�֤��
        
        �z���^��������`�H�U��T�� JSON �榡�G
        {
          "title": "�w��[�D�D]���ӤH�ƾǲ߸��|",
          "description": "���ǲ߸��|�w��[name]���ǲ߭���M��e���Ѥ����i��q���w��",
          "objectives": ["�ؼ� 1", "�ؼ� 2", "�ؼ� 3"],
          "modules": [
            {
              "title": "�Ҳ� 1: [���D]",
              "description": "�Ҳմy�z",
              "activities": [
                {
                  "type": "�\Ū",
                  "title": "���ʼ��D",
                  "description": "���ʴy�z",
                  "estimated_time": "20 ����",
                  "difficulty": "��Ǫ�"
                }
              ],
              "resources": ["�귽 1", "�귽 2"],
              "assessment": "�Ҳյ����y�z"
            }
          ]
        }
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���e�ͦ��ӤH�ƾǲ߸��|�G
        
        �ǥ��ɮסG
        {profile}
        
        ���絲�G�G
        {test_results}
        
        ���e�G
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
        SystemMessage(content="""�z�O�u�ǲ߹٦�v�A�@�Ӥ͵��B�����U�� AI �P���A�P�ǥͶi�榳�س]�ʪ��Q�סC
        �z������O�G
        1. �����@��]�b�ǲ߸ӥD�D�����@�w���Ѫ��P��
        2. ���X�P�i��P�ʫ�Ҫ��`����{�����D
        3. ���ѷũM�����ɡA�Ӥ��O�������X����
        4. �H��ܪ��覡��F�Q�k�A���O�ǥͤ�������y
        5. �ϥ�Ĭ��ԩ������ݪk���U�ǥ͵o�{����
        6. ���y�ëO���n�����A��
        
        �ھڴ��Ѫ��������e�^���A�����n�u�O²��a�I�w��T�C
        �ӬO�H�۵M���覡�i��Ӧ^�Q�סA�N���@�_�ǲߤ@�ˡC
        """),
        HumanMessagePromptTemplate.from_template("""�ǥͧƱ�Q�׳o�ӥD�D�G
        
        �D�D: {topic}
        
        �������e:
        {context}
        
        �ǥͰT��:
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
        SystemMessage(content="""�z�O�@��M�~���Ш|�����]�p�v�C
        �ھڴ��Ѫ��ǲ߼Ҳդ��e�M�ǥͪ���e���Ѥ����A�]�p�@������A�]�t�h���D�H�����ǥͪ��ǲߦ��G�C
        
        �������P�ǥͪ���e�����۲šG
        - ��Ǫ̡G��h²����D�]70%�^�A�@�Ǥ������D�]30%�^
        - ���Ū̡G�@��²����D�]30%�^�A�D�n�O�������D�]50%�^�A�@�ǧx�����D�]20%�^
        - ���Ū̡G�@�Ǥ������D�]40%�^�A�D�n�O�x�����D�]60%�^
        
        �]�p�����D�����վǥ͹鷺�e���z�ѡB���ΩM���R��O�C
        
        ���C�Ӱ��D�A�д��ѡG
        1. ���D�奻
        2. �|�Ӧh��ﶵ�]A, B, C, D�^
        3. ���T����
        4. �����򥿽T������
        5. ���ׯŧO
        
        �z������`�H�U��T�� JSON �榡�G
        {
          "title": "����G[�D�D]",
          "description": "������N�����z��[�D�D]���ǲߦ��G",
          "questions": [
            {
              "question": "���D�奻�H",
              "choices": ["A. �ﶵ A", "B. �ﶵ B", "C. �ﶵ C", "D. �ﶵ D"],
              "correct_answer": "A. �ﶵ A",
              "explanation": "������ A �O���T���ת�����",
              "difficulty": "²��"
            }
          ]
        }

        �ھھǥͪ������ͦ��`�@ 5 �Ӱ��D�A�þA����t���סC
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���e�ͦ��@������G
        
        �ǥͪ���e���Ѥ���: {knowledge_level}
        �Ҳդ��e: {context}
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
        SystemMessage(content="""�z�O�@��ϫ�ǲ߱нm�A�M�����U�ǥͳЫئ��N�q���ǲߤ�x�C
        �ھھǥͧ������ǲ߼ҲթM���絲�G�A�޾ɥL�̤ϫ�ۤv���ǲߡC
        
        ���X�`����{���}�񦡰��D�H�P�i�ϫ�A�]�A�G
        1. �L�̾Ǩ�F����]���䷧���M���ѡ^
        2. �L�̹�ǲ߹L�{���P��
        3. �L��ı�o���D�Ԫ��a��
        4. �L�̤��M��������D
        5. �L�̦p�����ΩҾǪ����e
        6. �L�̱��U�ӷQ�Ǥ���
        
        �z���ؼЬO���U�ǥͳЫؤ@���״I�B���ϫ�ʪ��ǲߤ�x�A��L�̪����������ȡC
        """),
        HumanMessagePromptTemplate.from_template("""���U�ǥͰ��H�U���e�Ыؾǲߤ�x�ϫ�G
        
        �������Ҳ�: {module_title}
        
        �Ҳդ��e�K�n: {module_summary}
        
        ���絲�G: {test_results}
        """)
    ])
    
    learning_log_chain = prompt | chat_model | StrOutputParser()
    
    return learning_log_chain

def create_learning_log_analyzer(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M�~���Ш|���R�v�A�M�����R�ǥͪ��ǲߤ�x�C
        �ھھǥͪ��ǲߤ�x�A�����G
        
        1. �����䷧�����z�ѵ{��
        2. �u�թM�۫H�����
        3. �V�c�λ~�Ѫ����
        4. ����ƪ����P����
        5. �ǲ߭��檺����
        6. ��b���ǲߤU�@�B
        
        �N�z���^���榡�Ƭ��H�U��T�� JSON ���c:
        {
          "understanding_level": "��/��/�C",
          "strengths": ["�u�� 1", "�u�� 2"],
          "areas_for_improvement": ["��i��� 1", "��i��� 2"],
          "emotional_response": "�ﱡ�P�������y�z",
          "learning_style_indicators": ["���� 1", "���� 2"],
          "recommended_next_steps": ["��ĳ�B�J 1", "��ĳ�B�J 2"],
          "suggested_resources": ["�귽 1", "�귽 2"]
        }
        """),
        HumanMessagePromptTemplate.from_template("""���R�H�U�ǲߤ�x�G
        
        �ǥ�: {student_name}
        �D�D: {topic}
        �ǲߤ�x���e:
        {log_content}
        """)
    ])
    
    learning_log_analysis_chain = prompt | chat_model | StrOutputParser()
    
    return learning_log_analysis_chain

def create_knowledge_level_assessor(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��Ш|�����M�a�C
        �ھھǥͪ����絲�G�A�T�w�L�̦b���S�w�D�D�W�����Ѥ����C
        
        �Ҽ{�G
        1. ���T���ת��ƶq
        2. ���T�^�������D����
        3. ���׼Ҧ��]�@�P���z�ѻP�t�Z�^
        
        �N�ǥͪ����Ѥ����������G
        - ��Ǫ̡G�򥻼��x�A�z��²�淧��
        - ���Ū̡G�}�n���֤߷����z�ѡA�@�w�����ί�O
        - ���Ū̡G�`��z�ѡA��N�������Ω�s����
        
        ���z����������²�u���z�ѡC
        
        �N�z���^���榡�Ƭ� JSON ��H�G
        {
          "knowledge_level": "��Ǫ�/���Ū�/���Ū�",
          "justification": "�������²�u����",
          "strengths": ["�u�� 1", "�u�� 2"],
          "areas_for_improvement": ["��i��� 1", "��i��� 2"],
          "recommended_focus": "�ǥͱ��U�����ӱM�`�󤰻�"
        }
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���絲�G�����ǥͪ����Ѥ����G
        
        ����: {test_title}
        
        ���D�M����:
        {test_results}
        """)
    ])
    
    knowledge_level_chain = prompt | chat_model | StrOutputParser()
    
    return knowledge_level_chain

def create_module_content_generator(chat_model, retriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M�~���Ш|���e�Ч@�̡C
        �ھڴ��Ѫ��ҲեD�D�H�ξǥͪ��ǲ߭���M���Ѥ����A�ЫؤޤH�J�Ӫ��Ш|���e�C
        
        �z�����e���G
        1. �w��ǥͪ��ǲ߭���]��ı���Bťı���ΰ�ı���^�i��q���w��
        2. �A�X�ǥͪ����Ѥ���
        3. �]�t���䷧�����M������
        4. �ϥΥܨҩM����ӻ����n�I
        5. �]�A�A�X���Ѥ������N�[����
        6. ���c�M���A�]�t���T�������M���D
        7. �H�����I��²�u�`������
        
        �ϥ� markdown �榡�Ʊz�����e�H�����iŪ�ʡC
        """),
        HumanMessagePromptTemplate.from_template("""���H�U���e�ЫرШ|���e�G
        
        �ҲեD�D: {module_topic}
        �ǥ;ǲ߭���: {learning_style}
        �ǥͪ��Ѥ���: {knowledge_level}
        
        �����ӷ�����:
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
    console.print("\n[bold cyan]===== �ǲ߭������ =====[/bold cyan]")
    
    survey_chain = create_learning_style_survey(chat_model)
    survey = survey_chain.invoke({})
    
    console.print(Markdown(survey))
    console.print("\n[bold yellow]�Ц^���C�Ӱ��D�H���U�T�w�z���ǲ߭���C[bold yellow]")
    
    # �B�z�ݨ����G
    console.print("\n[bold]�����ݨ���A���ؾǲ߭���̲ŦX�z�H[/bold]")
    console.print("1. ��ı���ǲߪ�")
    console.print("2. ťı���ǲߪ�")
    console.print("3. ��ı���ǲߪ�")
    
    style_choice = Prompt.ask("��ܱz���D�n�ǲ߭���", choices=["1", "2", "3"])
    
    learning_styles = {
        "1": "��ı��",
        "2": "ťı��",
        "3": "��ı��"
    }
    
    student_profile.learning_style = learning_styles[style_choice]
    administer_pretest
    # �x�s��s�᪺�ɮ�
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return student_profile

def administer_pretest(chat_model, retriever, student_profile):
    console.print("\n[bold cyan]===== �e�� =====[/bold cyan]")
    console.print("[yellow]�ھڤ��e�ͦ��e��...[/yellow]")
    
    pretest_chain = create_pretest_generator(chat_model, retriever)
    pretest_json = pretest_chain.invoke("")
    
    try:
        pretest = json.loads(pretest_json)
    except json.JSONDecodeError:
        console.print("[bold red]�ѪR�e�� JSON �ɥX���C�ϥγƥΤ�k�C[bold red]")
        # �p�G JSON �Q��L��r�]��A���� JSON
        import re
        json_match = re.search(r'({[\s\S]*})', pretest_json)
        if json_match:
            try:
                pretest = json.loads(json_match.group(1))
            except:
                console.print("[bold red]�L�k�ѪR�e�� JSON�C�ϥιw�]����C[bold red]")
                pretest = {
                    "title": "�w�]�e��",
                    "description": "�ѩ�ѪR���D�A�o�O�@�ӳƥδ���C",
                    "questions": [
                        {
                            "question": "RAG �b�Ш|�t�Τ����D�n�ت��O����H",
                            "choices": ["A. �ͦ��H�����D", "B. �˯��ͦ��һݪ�������T", 
                                       "C. ���N�Юv", "D. �b�ǲ߹L�{�����񭵼�"],
                            "correct_answer": "B. �˯��ͦ��һݪ�������T",
                            "explanation": "RAG�]�˯��W�j�ͦ��^���U��ǽT����T�˯��C",
                            "difficulty": "����"
                        }
                    ]
                }
    
    console.print(f"\n[bold green]{pretest['title']}[/bold green]")
    console.print(f"[italic]{pretest['description']}[/italic]\n")
    
    # �������
    score = 0
    total_questions = len(pretest['questions'])
    results = []
    
    for i, q in enumerate(pretest['questions']):
        console.print(f"\n[bold]���D {i+1}:[/bold] {q['question']}")
        for choice in q['choices']:
            console.print(f"  {choice}")
        
        answer = Prompt.ask("\n�z������ (A, B, C �� D)").upper()
        correct_letter = q['correct_answer'][0].upper()
        
        if answer == correct_letter:
            score += 1
            console.print("[bold green]���T�I[/bold green]")
        else:
            console.print(f"[bold red]���~�C���T���׬O {q['correct_answer']}[/bold red]")
        
        console.print(f"[italic]{q['explanation']}[/italic]")
        
        results.append({
            "question": q['question'],
            "student_answer": answer,
            "correct_answer": q['correct_answer'],
            "is_correct": answer == correct_letter,
            "difficulty": q['difficulty']
        })
    
    # �p�����
    percentage = (score / total_questions) * 100
    console.print(f"\n[bold]���秹���I�z�����ơG{score}/{total_questions} ({percentage:.1f}%)[/bold]")
    
    # �������Ѥ���
    if percentage >= 80:
        knowledge_level = "����"
    elif percentage >= 50:
        knowledge_level = "����"
    else:
        knowledge_level = "��Ǫ�"
    
    console.print(f"[yellow]�ھڱz���e�����G�A�z����e���Ѥ����O�G[bold]{knowledge_level}[/bold][/yellow]")
    
    # ��s�ǥ��ɮ�
    student_profile.current_knowledge_level = knowledge_level
    student_profile.learning_history.append({
        "activity_type": "�e��",
        "timestamp": datetime.datetime.now().isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "knowledge_level": knowledge_level
    })
    
    # �x�s��s�᪺�ɮ�
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return pretest, results, knowledge_level

def generate_learning_path(chat_model, retriever, student_profile, pretest_results):
    console.print("\n[bold cyan]===== �ͦ��ӤH�ƾǲ߸��| =====[/bold cyan]")
    console.print("[yellow]�ھڱz���ɮשM���絲�G�Ыؾǲ߸��|...[/yellow]")
    
    # �榡�ƴ��絲�G�H����ϥ�
    test_results_formatted = json.dumps({
        "score_percentage": (sum(1 for r in pretest_results if r["is_correct"]) / len(pretest_results)) * 100,
        "knowledge_level": student_profile.current_knowledge_level,
        "strengths": [r["question"] for r in pretest_results if r["is_correct"]],
        "weaknesses": [r["question"] for r in pretest_results if not r["is_correct"]]
    })
    
    # �榡���ɮץH����ϥ�
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
        console.print("[bold red]�ѪR�ǲ߸��| JSON �ɥX���C�ϥγƥΤ�k�C[bold red]")
        # �p�G JSON �Q��L��r�]��A���� JSON
        import re
        json_match = re.search(r'({[\s\S]*})', learning_path_json)
        if json_match:
            try:
                learning_path = json.loads(json_match.group(1))
            except:
                console.print("[bold red]�L�k�ѪR�ǲ߸��| JSON�C�ϥιw�]���|�C[bold red]")
                learning_path = {
                    "title": f"�w�� {student_profile.name} ���w�]�ǲ߸��|",
                    "description": f"���ǲ߸��|�w�� {student_profile.name} �� {student_profile.learning_style} �ǲ߭���M {student_profile.current_knowledge_level} ���Ѥ����i��q���w��",
                    "objectives": ["�ǲ֤߮߷���", "�إ߹�Χޯ�", "�ǳƵ���"],
                    "modules": [
                        {
                            "title": "�Ҳ� 1: �֤߷�������",
                            "description": "�ӥD�D���򥻷��������z",
                            "activities": [
                                {
                                    "type": "�\Ū",
                                    "title": "�֤߷�������",
                                    "description": "�[�\��¦���Ѫ��\Ū����",
                                    "estimated_time": "20 ����",
                                    "difficulty": student_profile.current_knowledge_level
                                }
                            ],
                            "resources": ["�D�n�ҵ{����"],
                            "assessment": "��¦������²�u����"
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
    
    # �����ɶ��\Ū���e
    console.print("\n[yellow]�Ъ�ɶ��\Ū�òz�Ѥ��e�C[yellow]")
    Prompt.ask("\n�ǳƦn�~��ɽЫ� Enter")
    
    return content

def engage_peer_discussion(chat_model, retriever, topic):
    console.print(f"\n[bold cyan]===== �P���Q��: {topic} =====[/bold cyan]")
    console.print("[yellow]�P�z�� AI �ǲ߹٦񨣭��I���X���D�ΰQ�ץD�D�H�[�`�z���z�ѡC[yellow]")
    
    discussion_chain = create_peer_discussion_ai(chat_model, retriever)
    
    console.print("\n[bold green]�ǲ߹٦�:[/bold green] �١I�ڤ]�b�ǲ߳o�ӥD�D�C�z�Q�Q�׭��Ǥ譱�Φ�������D�Q�ݡH")
    
    conversation_history = []
    
    while True:
        user_message = Prompt.ask("\n[bold blue]�z[/bold blue]")
        
        if user_message.lower() in ["�h�X", "����", "�A��", "�����Q��"]:
            console.print("\n[bold green]�ǲ߹٦�:[/bold green] �ܴΪ��Q�סI�p�G�z�Q�y��A��A�Чi�D�ڡC")
            break
        
        response = discussion_chain.invoke(topic, user_message)
        console.print(f"\n[bold green]�ǲ߹٦�:[/bold green] {response}")
        
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": response})
        
        if len(conversation_history) >= 10:  # ����Q�ת���
            console.print("\n[yellow]�ڭ̤w�g�i��F�@���ܦn���Q�סC�ǳƦn�~��ܡH[yellow]")
            if Confirm.ask("�����Q�סH"):
                break
    
    return conversation_history

def administer_posttest(chat_model, retriever, module, student_profile):
    console.print(f"\n[bold cyan]===== ���: {module['title']} =====[/bold cyan]")
    console.print("[yellow]���ձz��Ҳդ��e���z��...[/yellow]")
    
    # ����ҲեD�D
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    
    # �ھڼҲդ��e�M�ǥͤ����ͦ�������D
    posttest_chain = create_posttest_generator(chat_model, retriever)
    posttest_json = posttest_chain.invoke(student_profile.current_knowledge_level, module_topic)
    
    try:
        posttest = json.loads(posttest_json)
    except json.JSONDecodeError:
        console.print("[bold red]�ѪR��� JSON �ɥX���C�ϥγƥΤ�k�C[bold red]")
        # �p�G JSON �Q��L��r�]��A���� JSON
        import re
        json_match = re.search(r'({[\s\S]*})', posttest_json)
        if json_match:
            try:
                posttest = json.loads(json_match.group(1))
            except:
                console.print("[bold red]�L�k�ѪR��� JSON�C�ϥιw�]����C[bold red]")
                posttest = {
                    "title": f"���: {module_topic}",
                    "description": f"������N�����z�� {module_topic} ���ǲߦ��G",
                    "questions": [
                        {
                            "question": f"{module_topic} ���@�����䷧���O����H",
                            "choices": ["A. �ﶵ A", "B. �ﶵ B", "C. �ﶵ C", "D. �ﶵ D"],
                            "correct_answer": "A. �ﶵ A",
                            "explanation": "�ھڼҲդ��e�A�o�O���T���סC",
                            "difficulty": "����"
                        }
                    ]
                }
    
    console.print(f"\n[bold green]{posttest['title']}[/bold green]")
    console.print(f"[italic]{posttest['description']}[/italic]\n")
    
    # �������
    score = 0
    total_questions = len(posttest['questions'])
    results = []
    
    for i, q in enumerate(posttest['questions']):
        console.print(f"\n[bold]���D {i+1}:[/bold] {q['question']}")
        for choice in q['choices']:
            console.print(f"  {choice}")
        
        answer = Prompt.ask("\n�z������ (A, B, C �� D)").upper()
        correct_letter = q['correct_answer'][0].upper()
        
        if answer == correct_letter:
            score += 1
            console.print("[bold green]���T�I[/bold green]")
        else:
            console.print(f"[bold red]���~�C���T���׬O {q['correct_answer']}[/bold red]")
        
        console.print(f"[italic]{q['explanation']}[/italic]")
        
        results.append({
            "question": q['question'],
            "student_answer": answer,
            "correct_answer": q['correct_answer'],
            "is_correct": answer == correct_letter,
            "difficulty": q['difficulty']
        })
    
    # �p�����
    percentage = (score / total_questions) * 100
    console.print(f"\n[bold]���秹���I�z�����ơG{score}/{total_questions} ({percentage:.1f}%)[/bold]")
    
    # �����i�B�îھڻݭn�վ㪾�Ѥ���
    previous_level = student_profile.current_knowledge_level
    
    if percentage >= 80 and previous_level != "����":
        if previous_level == "��Ǫ�":
            new_level = "����"
        else:
            new_level = "����"
        console.print(f"[bold green]�i�B�ܤj�I�z�����Ѥ����w�q {previous_level} ���ɨ� {new_level}�C[bold green]")
        student_profile.current_knowledge_level = new_level
    elif percentage < 50 and previous_level != "��Ǫ�":
        if previous_level == "����":
            new_level = "����"
        else:
            new_level = "��Ǫ�"
        console.print(f"[yellow]�z�i��ݭn��h�m�ߡC�z�����Ѥ����w�q {previous_level} �վ㬰 {new_level}�C[yellow]")
        student_profile.current_knowledge_level = new_level
    else:
        console.print(f"[yellow]�z�����Ѥ����O���b {previous_level}�C[yellow]")
    
    # ��s�ǥ��ɮ�
    student_profile.learning_history.append({
        "activity_type": "���",
        "module": module['title'],
        "timestamp": datetime.datetime.now().isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "previous_level": previous_level,
        "current_level": student_profile.current_knowledge_level
    })
    
    # �x�s��s�᪺�ɮ�
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return posttest, results

def create_learning_log(chat_model, module, test_results, student_profile):
    console.print(f"\n[bold cyan]===== �ǲߤ�x: {module['title']} =====[/bold cyan]")
    
    # ����ҲեD�D�M�K�n
    module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
    module_summary = module['description']
    
    # �榡�ƴ��絲�G
    test_results_str = f"����: {sum(1 for r in test_results if r['is_correct'])}/{len(test_results)} �D���T\n"
    test_results_str += "�u��: " + ", ".join([r['question'] for r in test_results if r['is_correct']])
    test_results_str += "\n�ݭn��i�����: " + ", ".join([r['question'] for r in test_results if not r['is_correct']])
    
    # �ͦ��ϫ䴣��
    learning_log_chain = create_learning_log_prompter(chat_model)
    reflection_prompts = learning_log_chain.invoke({
        "module_title": module_topic,
        "module_summary": module_summary,
        "test_results": test_results_str
    })
    
    console.print(Markdown(reflection_prompts))
    
    # ����ǥͤϫ�
    console.print("\n[bold yellow]�ЮھڤW�z���ܼ��g�z���ǲߤ�x�ϫ�G[bold yellow]")
    log_content = ""
    
    while True:
        line = input()
        if line.lower() == "done":
            break
        log_content += line + "\n"
    
    # �ЫبëO�s�ǲߤ�x
    log_id = str(uuid.uuid4())[:8]
    log = LearningLog(
        id=log_id,
        student_id=student_profile.id,
        timestamp=datetime.datetime.now().isoformat(),
        topic=module_topic,
        content=log_content,
        reflections=[],  # �N�Ѥ��R��R
        questions=[],    # �N�Ѥ��R��R
        next_steps=[]    # �N�Ѥ��R��R
    )
    
    # ���R�ǲߤ�x
    analyzer_chain = create_learning_log_analyzer(chat_model)
    analysis_json = analyzer_chain.invoke({
        "student_name": student_profile.name,
        "topic": module_topic,
        "log_content": log_content
    })
    
    try:
        analysis = json.loads(analysis_json)
        
        # �ھڤ��R��s�ǥ��ɮ�
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
        
        # ��ܤ��R�K�n
        console.print("\n[bold green]�ǲߤ�x���R���G:[/bold green]")
        console.print(f"[bold]�z�ѵ{��:[/bold] {analysis.get('understanding_level', '���T�w')}")
        
        console.print("\n[bold]�u��:[/bold]")
        for strength in analysis.get("strengths", []):
            console.print(f"? {strength}")
        
        console.print("\n[bold]�ݭn��i�����:[/bold]")
        for area in analysis.get("areas_for_improvement", []):
            console.print(f"? {area}")
        
        console.print("\n[bold]��ĳ���U�@�B���:[/bold]")
        for step in analysis.get("recommended_next_steps", []):
            console.print(f"? {step}")
    
    except (json.JSONDecodeError, KeyError):
        console.print("[bold red]���R�ǲߤ�x�ɥX���C�ȫO�s��l��x�C[bold red]")
    
    # �O�s�ǲߤ�x
    with open(f"learning_logs/{log_id}.json", "w") as f:
        f.write(log.json(indent=4))
    
    # ��s�ëO�s�ǥ��ɮ�
    with open(f"student_profiles/{student_profile.id}.json", "w") as f:
        f.write(student_profile.json(indent=4))
    
    return log

# Main application function
def main():
    console.print("[bold cyan]=====================================[/bold cyan]")
    console.print("[bold cyan]== RAG �N�[�Ш|�t�� ==[/bold cyan]")
    console.print("[bold cyan]=====================================[/bold cyan]\n")
    
    # ��l�Ƽҫ��M RAG �t��
    console.print("[yellow]���b��l�ƨt��...[/yellow]")
    chat_model, embedding = initialize_models()
    retriever = initialize_rag_system(embedding)
    
    # �޲z�ǥ��ɮ�
    student_profile = manage_student_profile()
    console.print(f"\n[bold green]�w��, {student_profile.name}![/bold green]")
    
    # �p�G�|�������ǲ߭�������A�i�����
    if not student_profile.learning_style:
        student_profile = conduct_learning_style_survey(chat_model, student_profile)
    else:
        console.print(f"\n[yellow]�z���ǲ߭���O: [bold]{student_profile.learning_style}[/bold][/yellow]")
    
    # �i��e��
    pretest, pretest_results, knowledge_level = administer_pretest(chat_model, retriever, student_profile)
    
    # �ͦ��ӤH�ƾǲ߸��|
    learning_path = generate_learning_path(chat_model, retriever, student_profile, pretest_results)
    
    # �ǲ߹L�{
    for module_index, module in enumerate(learning_path['modules']):
        console.print(f"\n[bold cyan]===== �}�l�Ҳ� {module_index + 1}/{len(learning_path['modules'])} =====[/bold cyan]")
        
        # �߰ݨϥΪ̬O�_�ǳƦn�i�榹�Ҳ�
        proceed = Confirm.ask(f"�ǳƦn�}�l�Ҳ�: {module['title']}?")
        if not proceed:
            continue
        
        # ���ѼҲդ��e
        module_content = deliver_module_content(chat_model, retriever, student_profile, module)
        
        # �i��P���Q��
        module_topic = module['title'].split(": ", 1)[1] if ": " in module['title'] else module['title']
        discussion_history = engage_peer_discussion(chat_model, retriever, module_topic)
        
        # �i����
        posttest, posttest_results = administer_posttest(chat_model, retriever, module, student_profile)
        
        # �Ыؾǲߤ�x
        learning_log = create_learning_log(chat_model, module, posttest_results, student_profile)
        
        # �߰ݨϥΪ̬O�_�Q�~��i��U�@�ӼҲ�
        if module_index < len(learning_path['modules']) - 1:
            continue_learning = Confirm.ask("�O�_�~��i��U�@�ӼҲ�?")
            if not continue_learning:
                break
    
    console.print("\n[bold green]===== �ǲ߸��|���� =====[/bold green]")
    console.print("[yellow]�P�±z�ϥ� RAG �N�[�Ш|�t�ΡI[/yellow]")
    console.print(f"[yellow]�z����e���Ѥ���: [bold]{student_profile.current_knowledge_level}[/bold][/yellow]")
    
    # �`���ǲ߶i��
    console.print("\n[bold]�z���ǲ߮ȵ{�K�n:[/bold]")
    console.print(f"? �����F {len(student_profile.learning_history)} ���ǲ߬���")
    console.print(f"? ��e�u��: {', '.join(student_profile.strengths) if student_profile.strengths else '�|���T�w'}")
    console.print(f"? �ݭn��i�����: {', '.join(student_profile.areas_for_improvement) if student_profile.areas_for_improvement else '�|���T�w'}")
    
    console.print("\n[bold]����ǲߪ���ĳ:[/bold]")
    if student_profile.current_knowledge_level == "��Ǫ�":
        console.print("? �M�`��x����¦����")
        console.print("? �h�m�ߪ�Ǫ̨줤�Ū��d��")
    elif student_profile.current_knowledge_level == "����":
        console.print("? �[�`������D�D���z��")
        console.print("? �}�l�N�������Ω��ڰ��D")
    else:  # ����
        console.print("? �����ӻ�쪺�M�~�D�D")
        console.print("? �Ҽ{�оǩΫ��ɥL�H�H�d�T�z������")

if __name__ == "__main__":
    main()