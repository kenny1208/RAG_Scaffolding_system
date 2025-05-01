# -*- coding: big5 -*-

import datetime
import json
import uuid
from typing import List, Dict, Any, Tuple, Optional
import os

from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStoreRetriever

# Import necessary components from other modules
from config import console, LOGS_DIR, CHAT_MODEL # Use initialized model
from models import StudentProfile, LearningLog, Test, Question # Import models
from profile_manager import save_profile # Import save function
from chains import ( # Import all chain creation functions
    create_learning_style_survey,
    create_pretest_generator,
    create_learning_path_generator,
    create_module_content_generator,
    create_peer_discussion_ai,
    create_posttest_generator,
    create_learning_log_prompter,
    create_learning_log_analyzer,
    # create_knowledge_level_assessor # Not explicitly used in original main flow, but available
)

# --- Core Workflow Functions ---

def conduct_learning_style_survey(
    chat_model: ChatGoogleGenerativeAI,
    student_profile: StudentProfile
) -> StudentProfile:
    """Conducts the learning style survey and updates the profile."""
    console.print("\n[bold cyan]===== �ǲ߭������ =====[/bold cyan]")

    # Check if already done
    if student_profile.learning_style:
        console.print(f"[yellow]�ǲ߭���w�]�w��: [bold]{student_profile.learning_style}[/bold][/yellow]")
        return student_profile

    survey_chain = create_learning_style_survey(chat_model)
    console.print("[yellow]���b�ͦ��ǲ߭���ݨ�...[/yellow]")
    try:
        survey = survey_chain.invoke({})
        console.print(Markdown(survey))
    except Exception as e:
        console.print(f"[bold red]�L�k�ͦ��ݨ�: {e}[/bold red]")
        # Default or ask manually
        console.print("[yellow]�L�k�۰ʥͦ��ݨ��A�Ф�ʿ�ܡC[/yellow]")
        style_choice = Prompt.ask(
            "�п�ܱz���D�n�ǲ߭��� (Select your primary learning style)",
            choices=["1", "2", "3"],
            default="1"
        )
    else:
        # Ask based on the generated survey
        console.print("\n[bold yellow]�Ц^���C�Ӱ��D�H���U�T�w�z���ǲ߭���C[bold yellow]")
        # (Optional: Add logic to actually process answers if the survey was structured for it)
        # For now, just ask the final choice:
        console.print("\n[bold]�����ݨ���A���ؾǲ߭���̲ŦX�z�H (After completing the survey, which style best describes you?)[/bold]")
        console.print("1. ��ı�� (Visual)")
        console.print("2. ťı�� (Auditory)")
        console.print("3. ��ı�� (Kinesthetic/Reading-Writing)") # Combine K and R/W for simplicity maybe
        style_choice = Prompt.ask(
            "��ܱz���D�n�ǲ߭��� (Choose your primary style)",
            choices=["1", "2", "3"],
            default="1"
        )

    learning_styles = {"1": "��ı��", "2": "ťı��", "3": "��ı��"}
    student_profile.learning_style = learning_styles[style_choice]
    console.print(f"[green]�ǲ߭���w�]�w��: [bold]{student_profile.learning_style}[/bold][/green]")

    save_profile(student_profile) # Save updated profile
    return student_profile


def administer_pretest(
    chat_model: ChatGoogleGenerativeAI,
    retriever: VectorStoreRetriever,
    student_profile: StudentProfile
) -> Tuple[Optional[Dict], List[Dict], Optional[str]]:
    """Administers the pre-test, assesses knowledge level, and updates profile."""
    console.print("\n[bold cyan]===== �e�� (Pre-Test) =====[/bold cyan]")
    console.print("[yellow]���b�ھڤ��e�ͦ��e��... (Generating pre-test based on content...)[/yellow]")

    pretest_chain = create_pretest_generator(chat_model, retriever)
    pretest_data: Optional[Dict] = None
    try:
        # Provide a general topic for context retrieval if needed
        pretest_data = pretest_chain.invoke({"topic": "��¦����"}) # Or derive topic
        # Validate structure (basic check)
        if not pretest_data or "questions" not in pretest_data or not isinstance(pretest_data["questions"], list):
            console.print("[bold red]�ͦ�������榡�L�ġC[/bold red]")
            return None, [], student_profile.current_knowledge_level # Return existing level
    except Exception as e:
        console.print(f"[bold red]�ͦ��e���ɵo�Ϳ��~: {e}[/bold red]")
        return None, [], student_profile.current_knowledge_level # Return existing level

    console.print(f"\n[bold green]{pretest_data.get('title', '�e��')}[/bold green]")
    console.print(f"[italic]{pretest_data.get('description', '�����z����¦����')}[/italic]\n")

    # Execute the test
    score = 0
    total_questions = len(pretest_data['questions'])
    results = [] # To store detailed results

    if total_questions == 0:
        console.print("[yellow]���礤�S�����D�C[/yellow]")
        return pretest_data, [], student_profile.current_knowledge_level

    for i, q_data in enumerate(pretest_data['questions']):
        # Validate question structure
        if not all(k in q_data for k in ["question", "choices", "correct_answer", "explanation", "difficulty"]):
             console.print(f"[yellow]���L�榡�����T�����D {i+1}�C[/yellow]")
             total_questions -= 1 # Adjust total count
             continue

        q = Question.model_validate(q_data) # Validate with Pydantic model

        console.print(f"\n[bold]���D {i+1}:[/bold] {q.question}")
        choice_map = {}
        for choice in q.choices:
            letter = choice.split('.')[0].strip().upper()
            console.print(f"  {choice}")
            choice_map[letter] = choice

        if not choice_map:
            console.print(f"[yellow]���D {i+1} �S�����Ī��ﶵ�A���L�C[/yellow]")
            total_questions -= 1
            continue

        valid_answer_keys = list(choice_map.keys())
        answer = Prompt.ask(
            f"\n�z������ ({'/'.join(valid_answer_keys)})",
            choices=valid_answer_keys,
            show_choices=False # Don't repeat choices
        ).upper()

        correct_letter = q.correct_answer.split('.')[0].strip().upper()
        is_correct = (answer == correct_letter)

        if is_correct:
            score += 1
            console.print("[bold green]���T�I[/bold green]")
        else:
            console.print(f"[bold red]���~�C���T���׬O {q.correct_answer}[/bold red]")

        console.print(f"[italic]����: {q.explanation}[/italic]")
        console.print(f"[dim]����: {q.difficulty}[/dim]")

        results.append({
            "question": q.question,
            "student_answer": choice_map.get(answer, answer), # Store full answer text if possible
            "correct_answer": q.correct_answer,
            "is_correct": is_correct,
            "difficulty": q.difficulty
        })

    # Calculate score and assess level
    if total_questions > 0:
        percentage = (score / total_questions) * 100
        console.print(f"\n[bold]���秹���I�z�����ơG{score}/{total_questions} ({percentage:.1f}%)[/bold]")

        # Assess knowledge level (simple thresholding)
        if percentage >= 75: # Adjusted thresholds
            knowledge_level = "����"
        elif percentage >= 40:
            knowledge_level = "����"
        else:
            knowledge_level = "��Ǫ�"
    else:
        console.print("[yellow]�L�k�p����ơA�]���S�����Ī����D�C[/yellow]")
        percentage = 0.0
        knowledge_level = student_profile.current_knowledge_level or "��Ǫ�" # Keep old or default

    console.print(f"[yellow]�ھڱz���e�����G�A�z����e���Ѥ����������G[bold]{knowledge_level}[/bold][/yellow]")

    # Update student profile
    student_profile.current_knowledge_level = knowledge_level
    student_profile.learning_history.append({
        "activity_type": "�e��",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "assessed_knowledge_level": knowledge_level # Changed key name
    })

    save_profile(student_profile) # Save updated profile
    return pretest_data, results, knowledge_level


def generate_learning_path(
    chat_model: ChatGoogleGenerativeAI,
    retriever: VectorStoreRetriever,
    student_profile: StudentProfile,
    pretest_results: List[Dict]
) -> Optional[Dict]:
    """Generates a personalized learning path based on profile and pre-test."""
    console.print("\n[bold cyan]===== �ͦ��ӤH�ƾǲ߸��| (Generating Personalized Learning Path) =====[/bold cyan]")
    console.print("[yellow]���b�ھڱz���ɮשM���絲�G�Ыؾǲ߸��|...[/yellow]")

    # Format pre-test results for the prompt
    score = sum(1 for r in pretest_results if r["is_correct"])
    total = len(pretest_results)
    percentage = (score / total * 100) if total > 0 else 0
    test_results_summary = json.dumps({
        "score_percentage": round(percentage, 1),
        "knowledge_level": student_profile.current_knowledge_level,
        "strengths_topics": [r["question"] for r in pretest_results if r["is_correct"]],
        "weakness_topics": [r["question"] for r in pretest_results if not r["is_correct"]]
    }, ensure_ascii=False) # Use ensure_ascii=False for Chinese characters

    # Format profile for the prompt
    profile_summary = json.dumps({
        "name": student_profile.name,
        "learning_style": student_profile.learning_style,
        "current_knowledge_level": student_profile.current_knowledge_level,
        "interests": student_profile.interests
    }, ensure_ascii=False)

    learning_path_chain = create_learning_path_generator(chat_model, retriever)
    learning_path_data: Optional[Dict] = None
    try:
        # Provide context for retriever if needed, e.g., based on weaknesses
        weak_topics = ", ".join([r["question"] for r in pretest_results if not r["is_correct"]])
        retrieval_topic = f"�ǲߥD�D�w��: {weak_topics}" if weak_topics else "�@��ǲߥD�D"

        learning_path_data = learning_path_chain.invoke({
            "profile": profile_summary,
            "test_results": test_results_summary,
            "topic": retrieval_topic # Pass specific topic for context
        })
         # Basic validation
        if not learning_path_data or "modules" not in learning_path_data or not isinstance(learning_path_data["modules"], list):
            console.print("[bold red]�ͦ����ǲ߸��|�榡�L�ġC[/bold red]")
            return None

    except Exception as e:
        console.print(f"[bold red]�ͦ��ǲ߸��|�ɵo�Ϳ��~: {e}[/bold red]")
        return None

    # Display the learning path
    console.print(f"\n[bold green]{learning_path_data.get('title', '�ӤH�ƾǲ߸��|')}[/bold green]")
    console.print(f"[italic]{learning_path_data.get('description', '���z�q���w��ǲ߭p�e')}[/italic]\n")

    console.print("[bold]�ǲߥؼ� (Learning Objectives):[/bold]")
    for i, objective in enumerate(learning_path_data.get('objectives', []), 1):
        console.print(f" {i}. {objective}")

    console.print("\n[bold]�ǲ߳��` (Learning Modules):[/bold]")
    for i, module in enumerate(learning_path_data.get('modules', []), 1):
        console.print(f"\n[bold cyan]���` {i}: {module.get('title', '���R�W���`')}[/bold cyan]")
        console.print(f"  [italic]�y�z: {module.get('description', '�L')}[/italic]")

        console.print("  [bold]���� (Activities):[/bold]")
        for j, activity in enumerate(module.get('activities', []), 1):
            act_type = activity.get('type', '����')
            act_title = activity.get('title', '���R�W����')
            act_desc = activity.get('description', '�L')
            act_diff = activity.get('difficulty', '���w')
            console.print(f"    {j}. {act_title} ({act_type}) - {act_desc} [����: {act_diff}]")

        if 'resources' in module and module['resources']:
            console.print("  [bold]�귽 (Resources):[/bold]")
            for resource in module['resources']:
                console.print(f"    - {resource}")

    # Add learning path generation to history (optional)
    student_profile.learning_history.append({
        "activity_type": "�ǲ߸��|�ͦ�",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "path_title": learning_path_data.get('title', 'N/A'),
        "num_modules": len(learning_path_data.get('modules', []))
    })
    save_profile(student_profile)

    return learning_path_data


def deliver_module_content(
    chat_model: ChatGoogleGenerativeAI,
    retriever: VectorStoreRetriever,
    student_profile: StudentProfile,
    module: Dict[str, Any]
) -> Optional[str]:
    """Delivers the content for a specific learning module."""
    module_title = module.get('title', '���R�W���`')
    module_desc = module.get('description', '�L')
    console.print(f"\n[bold cyan]===== ���`���e: {module_title} =====[/bold cyan]")
    console.print(f"[italic]{module_desc}[/italic]\n")

    # Extract topic for content generation and retrieval
    # Handle cases like "Module 1: Introduction" -> "Introduction"
    module_topic = module_title.split(":", 1)[-1].strip() if ":" in module_title else module_title

    content_chain = create_module_content_generator(chat_model, retriever)
    content: Optional[str] = None
    try:
        content = content_chain.invoke({
            "module_topic": module_topic,
            "learning_style": student_profile.learning_style or "�V�X��", # Default if not set
            "knowledge_level": student_profile.current_knowledge_level or "��Ǫ�", # Default if not set
            # context is handled inside the chain via module_topic -> retriever
        })
    except Exception as e:
        console.print(f"[bold red]�ͦ����`���e�ɵo�Ϳ��~: {e}[/bold red]")
        return None # Indicate failure

    if content:
        console.print(Markdown(content))
        # Give time to read
        console.print("\n[yellow]�Ъ�ɶ��\Ū�òz�Ѥ��e�C[/yellow]")
        Prompt.ask("\n�ǳƦn�~��ɽЫ� Enter ")
    else:
        console.print("[red]�L�k�ͦ������`�����e�C[/red]")

    # Log content delivery (optional)
    student_profile.learning_history.append({
        "activity_type": "���e�ǲ�",
        "module": module_title,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "content_generated": bool(content)
    })
    save_profile(student_profile)

    return content # Return the generated content


def engage_peer_discussion(
    chat_model: ChatGoogleGenerativeAI,
    retriever: VectorStoreRetriever,
    topic: str,
    student_profile: StudentProfile # Pass profile for context/logging
) -> List[Dict]:
    """Engages the student in a peer discussion simulation."""
    console.print(f"\n[bold cyan]===== �P���Q�� : {topic} =====[/bold cyan]")
    console.print("[yellow]�P�z�� AI �ǲ߹٦񨣭��I���X���D�ΰQ�ץD�D�H�[�`�z���z�ѡC[/yellow]")
    console.print("[yellow]��J '����'�B'�h�X' �� '�A��' �ӵ����Q�סC[/yellow]")

    discussion_chain = create_peer_discussion_ai(chat_model, retriever)
    conversation_history = [] # Store user/AI messages

    # Initial message from AI partner
    ai_intro = "�١I�ڤ]�b�ǲ߳o�ӥD�D�C�z�Q�Q�׭��Ǥ譱�Φ�������D�Q�ݡH"
    console.print(f"\n[bold green]�ǲ߹٦�:[/bold green] {ai_intro}")
    conversation_history.append({"role": "assistant", "content": ai_intro})


    turn_limit = 5 # Limit number of turns
    turn_count = 0
    while turn_count < turn_limit:
        user_message = Prompt.ask("\n[bold blue]�z (You)[/bold blue]")

        if user_message.lower() in ["�h�X", "����", "�A��", "�����Q��", "exit", "quit", "bye"]:
            console.print("\n[bold green]�ǲ߹٦�:[/bold green] �ܴΪ��Q�סI�p�G�z�Q�y��A��A�Чi�D�ڡC")
            break

        conversation_history.append({"role": "user", "content": user_message})

        console.print("[yellow]�ǲ߹٦񥿦b���...[/yellow]")
        try:
            # Invoke requires a dict. Pass necessary info.
            response = discussion_chain.invoke({
                "topic": topic,
                "message": user_message
                # Context is handled inside the chain by retrieving based on topic
            })
            console.print(f"\n[bold green]�ǲ߹٦�:[/bold green] {response}")
            conversation_history.append({"role": "assistant", "content": response})
        except Exception as e:
            console.print(f"[bold red]�P�ǲ߹٦񷾳q�ɵo�Ϳ��~: {e}[/bold red]")
            # Allow user to try again or exit
            if not Confirm.ask("�O�_����? (Try again?)", default=True):
                 break
            continue # Skip turn increment if retrying


        turn_count += 1
        if turn_count == turn_limit:
             console.print("\n[yellow]�ڭ̤w�g�i��F�@���ܦn���Q�סC[/yellow]")
             if not Confirm.ask("�O�_�~��Q��?", default=False):
                 break
             else:
                 turn_limit += 3 # Extend discussion if requested


    # Log discussion activity
    student_profile.learning_history.append({
        "activity_type": "�P���Q��",
        "module_topic": topic,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "conversation_turns": len(conversation_history) // 2 # Approx user turns
    })
    save_profile(student_profile)

    return conversation_history


def administer_posttest(
    chat_model: ChatGoogleGenerativeAI,
    retriever: VectorStoreRetriever,
    module: Dict[str, Any],
    student_profile: StudentProfile
) -> Tuple[Optional[Dict], List[Dict]]:
    """Administers the post-test for a module and updates profile knowledge level."""
    module_title = module.get('title', '���R�W���`')
    console.print(f"\n[bold cyan]===== ��� (Post-Test): {module_title} =====[/bold cyan]")
    console.print("[yellow]���b���ձz�ﳹ�`���e���z��...[/yellow]")

    # Extract topic
    module_topic = module_title.split(":", 1)[-1].strip() if ":" in module_title else module_title
    current_level = student_profile.current_knowledge_level or "��Ǫ�"

    posttest_chain = create_posttest_generator(chat_model, retriever)
    posttest_data: Optional[Dict] = None
    try:
        posttest_data = posttest_chain.invoke({
            "knowledge_level": current_level,
            "module_topic": module_topic
            # Context handled inside chain
        })
        # Validate
        if not posttest_data or "questions" not in posttest_data or not isinstance(posttest_data["questions"], list):
            console.print("[bold red]�ͦ�������榡�L�ġC[/bold red]")
            return None, []
    except Exception as e:
        console.print(f"[bold red]�ͦ�����ɵo�Ϳ��~: {e}[/bold red]")
        return None, []

    console.print(f"\n[bold green]{posttest_data.get('title', '���')}[/bold green]")
    console.print(f"[italic]{posttest_data.get('description', '�����z���ǲߦ��G')}[/italic]\n")

    # Execute the test
    score = 0
    total_questions = len(posttest_data['questions'])
    results = []

    if total_questions == 0:
        console.print("[yellow]���礤�S�����D�C[/yellow]")
        return posttest_data, []

    for i, q_data in enumerate(posttest_data['questions']):
         # Validate question structure
        if not all(k in q_data for k in ["question", "choices", "correct_answer", "explanation", "difficulty"]):
             console.print(f"[yellow]���L�榡�����T�����D {i+1}�C[/yellow]")
             total_questions -= 1
             continue

        q = Question.model_validate(q_data) # Validate with Pydantic

        console.print(f"\n[bold]���D {i+1}:[/bold] {q.question}")
        choice_map = {}
        for choice in q.choices:
            letter = choice.split('.')[0].strip().upper()
            console.print(f"  {choice}")
            choice_map[letter] = choice

        if not choice_map:
            console.print(f"[yellow]���D {i+1} �S�����Ī��ﶵ�A���L�C[/yellow]")
            total_questions -= 1
            continue

        valid_answer_keys = list(choice_map.keys())
        answer = Prompt.ask(
            f"\n�z������ ({'/'.join(valid_answer_keys)})",
            choices=valid_answer_keys,
            show_choices=False
        ).upper()

        correct_letter = q.correct_answer.split('.')[0].strip().upper()
        is_correct = (answer == correct_letter)

        if is_correct:
            score += 1
            console.print("[bold green]���T�I[/bold green]")
        else:
            console.print(f"[bold red]���~�C���T���׬O {q.correct_answer}[/bold red]")

        console.print(f"[italic]����: {q.explanation}[/italic]")
        console.print(f"[dim]����: {q.difficulty}[/dim]")

        results.append({
            "question": q.question,
            "student_answer": choice_map.get(answer, answer),
            "correct_answer": q.correct_answer,
            "is_correct": is_correct,
            "difficulty": q.difficulty
        })

    # Calculate score and potentially update knowledge level
    new_level = current_level
    if total_questions > 0:
        percentage = (score / total_questions) * 100
        console.print(f"\n[bold]���秹���I�z�����ơG{score}/{total_questions} ({percentage:.1f}%)[/bold]")

        # Update level based on performance
        if percentage >= 80 and current_level != "����":
            new_level = "����" if current_level == "��Ǫ�" else "����"
            console.print(f"[bold green]��{�u���I�z�����Ѥ����w�q {current_level} ���ɨ� {new_level}�C[/bold green]")
        elif percentage < 50 and current_level != "��Ǫ�":
             new_level = "����" if current_level == "����" else "��Ǫ�"
             console.print(f"[yellow]�z�i��ݭn��h�m�ߡC�z�����Ѥ����w�q {current_level} �վ㬰 {new_level}�C[/yellow]")
        else:
            console.print(f"[yellow]�z�����Ѥ����O���b {current_level}�C[/yellow]")

    else:
        console.print("[yellow]�L�k�p����ƩΧ�s���Ѥ����C[/yellow]")
        percentage = 0.0

    # Update student profile
    student_profile.current_knowledge_level = new_level # Update with the potentially new level
    student_profile.learning_history.append({
        "activity_type": "���",
        "module": module_title,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "score": f"{score}/{total_questions}",
        "percentage": percentage,
        "previous_level": current_level,
        "current_level": new_level # Record the final level for this step
    })
    save_profile(student_profile)

    return posttest_data, results


def create_learning_log(
    chat_model: ChatGoogleGenerativeAI,
    module: Dict[str, Any],
    posttest_results: List[Dict],
    student_profile: StudentProfile
) -> Optional[LearningLog]:
    """Guides the student to create a learning log and analyzes it."""
    module_title = module.get('title', '���R�W���`')
    console.print(f"\n[bold cyan]===== �ǲߤ�x (Learning Log): {module_title} =====[/bold cyan]")

    module_topic = module_title.split(": ", 1)[-1].strip() if ": " in module_title else module_title
    module_summary = module.get('description', '�L') # Use module description as summary

    # Format test results for the prompter chain
    score = sum(1 for r in posttest_results if r["is_correct"])
    total = len(posttest_results)
    test_results_str = f"����: {score}/{total} ���T ({(score/total*100):.1f}% if total > 0 else 0.0)\n"
    strengths = [r['question'] for r in posttest_results if r['is_correct']]
    weaknesses = [r['question'] for r in posttest_results if not r['is_correct']]
    test_results_str += "��{�n�����D: " + (", ".join(strengths) if strengths else "�L")
    test_results_str += "\n�ݭn�[�j�����D: " + (", ".join(weaknesses) if weaknesses else "�L")

    # Generate reflection prompts
    prompter_chain = create_learning_log_prompter(chat_model)
    reflection_prompts: Optional[str] = None
    console.print("[yellow]���b�ͦ��ǲߤ�x����...[/yellow]")
    try:
        reflection_prompts = prompter_chain.invoke({
            "module_title": module_topic,
            "module_summary": module_summary,
            "test_results": test_results_str
        })
        console.print(Markdown(reflection_prompts))
    except Exception as e:
        console.print(f"[bold red]�ͦ���x���ܮɵo�Ϳ��~: {e}[/bold red]")
        console.print("[yellow]�Цۦ�ϫ�z�Ǩ�F����B�J�쪺�x���H�α��U�Ӫ��p�e�C[/yellow]")


    # Get student reflections
    console.print("\n[bold yellow]�Юھڴ��ܩΦۦ�ϫ�A���g�z���ǲߤ�x�]��J 'done' �����^�G[/bold yellow]")
    log_content_lines = []
    while True:
        try:
            line = input("> ")
            if line.strip().lower() == "done":
                break
            log_content_lines.append(line)
        except EOFError: # Handle Ctrl+D
             break
    log_content = "\n".join(log_content_lines).strip()

    if not log_content:
        console.print("[yellow]����J�ǲߤ�x���e�C[/yellow]")
        return None

    # Create initial log object
    log_id = str(uuid.uuid4())[:8]
    log = LearningLog(
        id=log_id,
        student_id=student_profile.id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        topic=module_topic,
        content=log_content,
        reflections=[], # To be filled by analysis
        questions=[],   # To be filled by analysis
        next_steps=[]   # To be filled by analysis
    )

    # Analyze the learning log
    analyzer_chain = create_learning_log_analyzer(chat_model)
    analysis: Optional[Dict] = None
    console.print("[yellow]���b���R�z���ǲߤ�x...[/yellow]")
    try:
        analysis = analyzer_chain.invoke({
            "student_name": student_profile.name,
            "topic": module_topic,
            "log_content": log_content
        })
    except Exception as e:
        console.print(f"[bold red]���R�ǲߤ�x�ɵo�Ϳ��~: {e}[/bold red]")
        # Save the log without analysis results
        analysis = {} # Empty analysis dict

    # Update log and profile based on analysis (if successful)
    if analysis:
        log.reflections = analysis.get("learning_style_indicators", []) # Use indicators as reflections?
        log.questions = analysis.get("areas_for_improvement", []) # Treat improvement areas as questions/topics
        log.next_steps = analysis.get("recommended_next_steps", [])

        # Update student profile strengths/weaknesses (append unique)
        current_strengths = set(student_profile.strengths)
        current_weaknesses = set(student_profile.areas_for_improvement)
        new_strengths = analysis.get("strengths", [])
        new_weaknesses = analysis.get("areas_for_improvement", [])

        updated_strengths = list(current_strengths.union(new_strengths))
        updated_weaknesses = list(current_weaknesses.union(new_weaknesses))

        student_profile.strengths = updated_strengths
        student_profile.areas_for_improvement = updated_weaknesses

        # Display analysis summary
        console.print("\n[bold green]�ǲߤ�x���R�K�n:[/bold green]")
        console.print(f"  [bold]�z�ѵ{��:[/bold] {analysis.get('understanding_level', '������')}")
        console.print(f"  [bold]�����쪺�u��:[/bold] {', '.join(new_strengths) if new_strengths else '�L'}")
        console.print(f"  [bold]��ĳ��i�����:[/bold] {', '.join(new_weaknesses) if new_weaknesses else '�L'}")
        console.print(f"  [bold]��ĳ���U�@�B:[/bold] {', '.join(log.next_steps) if log.next_steps else '�L'}")

    # Save the learning log
    log_path = os.path.join(LOGS_DIR, f"{log_id}.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log.model_dump_json(indent=4))
        console.print(f"[green]�ǲߤ�x�w�x�s�� {log_path}[/green]")
    except IOError as e:
        console.print(f"[bold red]�L�k�x�s�ǲߤ�x {log_path}: {e}[/bold red]")

    # Save the updated student profile
    save_profile(student_profile)

    return log