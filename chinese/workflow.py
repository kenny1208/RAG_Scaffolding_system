# -*- coding: UTF-8 -*-

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
    console.print("\n[bold cyan]===== 學習風格評估 =====[/bold cyan]")

    # Check if already done
    if student_profile.learning_style:
        console.print(f"[yellow]學習風格已設定為: [bold]{student_profile.learning_style}[/bold][/yellow]")
        return student_profile

    survey_chain = create_learning_style_survey(chat_model)
    console.print("[yellow]正在生成學習風格問卷...[/yellow]")
    try:
        survey = survey_chain.invoke({})
        console.print(Markdown(survey))
    except Exception as e:
        console.print(f"[bold red]無法生成問卷: {e}[/bold red]")
        # Default or ask manually
        console.print("[yellow]無法自動生成問卷，請手動選擇。[/yellow]")
        style_choice = Prompt.ask(
            "請選擇您的主要學習風格 (Select your primary learning style)",
            choices=["1", "2", "3"],
            default="1"
        )
    else:
        # Ask based on the generated survey
        console.print("\n[bold yellow]請回答每個問題以幫助確定您的學習風格。[bold yellow]")
        # (Optional: Add logic to actually process answers if the survey was structured for it)
        # For now, just ask the final choice:
        console.print("\n[bold]完成問卷後，哪種學習風格最符合您？ (After completing the survey, which style best describes you?)[/bold]")
        console.print("1. 視覺型 (Visual)")
        console.print("2. 聽覺型 (Auditory)")
        console.print("3. 動覺型 (Kinesthetic/Reading-Writing)") # Combine K and R/W for simplicity maybe
        style_choice = Prompt.ask(
            "選擇您的主要學習風格 (Choose your primary style)",
            choices=["1", "2", "3"],
            default="1"
        )

    learning_styles = {"1": "視覺型", "2": "聽覺型", "3": "動覺型"}
    student_profile.learning_style = learning_styles[style_choice]
    console.print(f"[green]學習風格已設定為: [bold]{student_profile.learning_style}[/bold][/green]")

    save_profile(student_profile) # Save updated profile
    return student_profile


def administer_pretest(
    chat_model: ChatGoogleGenerativeAI,
    retriever: VectorStoreRetriever,
    student_profile: StudentProfile
) -> Tuple[Optional[Dict], List[Dict], Optional[str]]:
    """Administers the pre-test, assesses knowledge level, and updates profile."""
    console.print("\n[bold cyan]===== 前測 (Pre-Test) =====[/bold cyan]")
    console.print("[yellow]正在根據內容生成前測... (Generating pre-test based on content...)[/yellow]")

    pretest_chain = create_pretest_generator(chat_model, retriever)
    pretest_data: Optional[Dict] = None
    try:
        # Provide a general topic for context retrieval if needed
        pretest_data = pretest_chain.invoke({"topic": "基礎知識"}) # Or derive topic
        # Validate structure (basic check)
        if not pretest_data or "questions" not in pretest_data or not isinstance(pretest_data["questions"], list):
            console.print("[bold red]生成的測驗格式無效。[/bold red]")
            return None, [], student_profile.current_knowledge_level # Return existing level
    except Exception as e:
        console.print(f"[bold red]生成前測時發生錯誤: {e}[/bold red]")
        return None, [], student_profile.current_knowledge_level # Return existing level

    console.print(f"\n[bold green]{pretest_data.get('title', '前測')}[/bold green]")
    console.print(f"[italic]{pretest_data.get('description', '評估您的基礎知識')}[/italic]\n")

    # Execute the test
    score = 0
    total_questions = len(pretest_data['questions'])
    results = [] # To store detailed results

    if total_questions == 0:
        console.print("[yellow]測驗中沒有問題。[/yellow]")
        return pretest_data, [], student_profile.current_knowledge_level

    for i, q_data in enumerate(pretest_data['questions']):
        # Validate question structure
        if not all(k in q_data for k in ["question", "choices", "correct_answer", "explanation", "difficulty"]):
             console.print(f"[yellow]跳過格式不正確的問題 {i+1}。[/yellow]")
             total_questions -= 1 # Adjust total count
             continue

        q = Question.model_validate(q_data) # Validate with Pydantic model

        console.print(f"\n[bold]問題 {i+1}:[/bold] {q.question}")
        choice_map = {}
        for choice in q.choices:
            letter = choice.split('.')[0].strip().upper()
            console.print(f"  {choice}")
            choice_map[letter] = choice

        if not choice_map:
            console.print(f"[yellow]問題 {i+1} 沒有有效的選項，跳過。[/yellow]")
            total_questions -= 1
            continue

        valid_answer_keys = list(choice_map.keys())
        answer = Prompt.ask(
            f"\n您的答案 ({'/'.join(valid_answer_keys)})",
            choices=valid_answer_keys,
            show_choices=False # Don't repeat choices
        ).upper()

        correct_letter = q.correct_answer.split('.')[0].strip().upper()
        is_correct = (answer == correct_letter)

        if is_correct:
            score += 1
            console.print("[bold green]正確！[/bold green]")
        else:
            console.print(f"[bold red]錯誤。正確答案是 {q.correct_answer}[/bold red]")

        console.print(f"[italic]解釋: {q.explanation}[/italic]")
        console.print(f"[dim]難度: {q.difficulty}[/dim]")

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
        console.print(f"\n[bold]測驗完成！您的分數：{score}/{total_questions} ({percentage:.1f}%)[/bold]")

        # Assess knowledge level (simple thresholding)
        if percentage >= 75: # Adjusted thresholds
            knowledge_level = "高級"
        elif percentage >= 40:
            knowledge_level = "中級"
        else:
            knowledge_level = "初學者"
    else:
        console.print("[yellow]無法計算分數，因為沒有有效的問題。[/yellow]")
        percentage = 0.0
        knowledge_level = student_profile.current_knowledge_level or "初學者" # Keep old or default

    console.print(f"[yellow]根據您的前測結果，您的當前知識水平評估為：[bold]{knowledge_level}[/bold][/yellow]")

    # Update student profile
    student_profile.current_knowledge_level = knowledge_level
    student_profile.learning_history.append({
        "activity_type": "前測",
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
    console.print("\n[bold cyan]===== 生成個人化學習路徑 (Generating Personalized Learning Path) =====[/bold cyan]")
    console.print("[yellow]正在根據您的檔案和測驗結果創建學習路徑...[/yellow]")

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
        retrieval_topic = f"學習主題針對: {weak_topics}" if weak_topics else "一般學習主題"

        learning_path_data = learning_path_chain.invoke({
            "profile": profile_summary,
            "test_results": test_results_summary,
            "topic": retrieval_topic # Pass specific topic for context
        })
         # Basic validation
        if not learning_path_data or "modules" not in learning_path_data or not isinstance(learning_path_data["modules"], list):
            console.print("[bold red]生成的學習路徑格式無效。[/bold red]")
            return None

    except Exception as e:
        console.print(f"[bold red]生成學習路徑時發生錯誤: {e}[/bold red]")
        return None

    # Display the learning path
    console.print(f"\n[bold green]{learning_path_data.get('title', '個人化學習路徑')}[/bold green]")
    console.print(f"[italic]{learning_path_data.get('description', '為您量身定制的學習計畫')}[/italic]\n")

    console.print("[bold]學習目標 (Learning Objectives):[/bold]")
    for i, objective in enumerate(learning_path_data.get('objectives', []), 1):
        console.print(f" {i}. {objective}")

    console.print("\n[bold]學習章節 (Learning Modules):[/bold]")
    for i, module in enumerate(learning_path_data.get('modules', []), 1):
        console.print(f"\n[bold cyan]章節 {i}: {module.get('title', '未命名章節')}[/bold cyan]")
        console.print(f"  [italic]描述: {module.get('description', '無')}[/italic]")

        console.print("  [bold]活動 (Activities):[/bold]")
        for j, activity in enumerate(module.get('activities', []), 1):
            act_type = activity.get('type', '未知')
            act_title = activity.get('title', '未命名活動')
            act_desc = activity.get('description', '無')
            act_diff = activity.get('difficulty', '未定')
            console.print(f"    {j}. {act_title} ({act_type}) - {act_desc} [難度: {act_diff}]")

        if 'resources' in module and module['resources']:
            console.print("  [bold]資源 (Resources):[/bold]")
            for resource in module['resources']:
                console.print(f"    - {resource}")

    # Add learning path generation to history (optional)
    student_profile.learning_history.append({
        "activity_type": "學習路徑生成",
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
    module_title = module.get('title', '未命名章節')
    module_desc = module.get('description', '無')
    console.print(f"\n[bold cyan]===== 章節內容: {module_title} =====[/bold cyan]")
    console.print(f"[italic]{module_desc}[/italic]\n")

    # Extract topic for content generation and retrieval
    # Handle cases like "Module 1: Introduction" -> "Introduction"
    module_topic = module_title.split(":", 1)[-1].strip() if ":" in module_title else module_title

    content_chain = create_module_content_generator(chat_model, retriever)
    content: Optional[str] = None
    try:
        content = content_chain.invoke({
            "module_topic": module_topic,
            "learning_style": student_profile.learning_style or "混合型", # Default if not set
            "knowledge_level": student_profile.current_knowledge_level or "初學者", # Default if not set
            # context is handled inside the chain via module_topic -> retriever
        })
    except Exception as e:
        console.print(f"[bold red]生成章節內容時發生錯誤: {e}[/bold red]")
        return None # Indicate failure

    if content:
        console.print(Markdown(content))
        # Give time to read
        console.print("\n[yellow]請花時間閱讀並理解內容。[/yellow]")
        Prompt.ask("\n準備好繼續時請按 Enter ")
    else:
        console.print("[red]無法生成此章節的內容。[/red]")

    # Log content delivery (optional)
    student_profile.learning_history.append({
        "activity_type": "內容學習",
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
    console.print(f"\n[bold cyan]===== 同儕討論 : {topic} =====[/bold cyan]")
    console.print("[yellow]與您的 AI 學習夥伴見面！提出問題或討論主題以加深您的理解。[/yellow]")
    console.print("[yellow]輸入 '結束'、'退出' 或 '再見' 來結束討論。[/yellow]")

    discussion_chain = create_peer_discussion_ai(chat_model, retriever)
    conversation_history = [] # Store user/AI messages

    # Initial message from AI partner
    ai_intro = "嗨！我也在學習這個主題。您想討論哪些方面或有什麼問題想問？"
    console.print(f"\n[bold green]學習夥伴:[/bold green] {ai_intro}")
    conversation_history.append({"role": "assistant", "content": ai_intro})


    turn_limit = 5 # Limit number of turns
    turn_count = 0
    while turn_count < turn_limit:
        user_message = Prompt.ask("\n[bold blue]您 (You)[/bold blue]")

        if user_message.lower() in ["退出", "結束", "再見", "結束討論", "exit", "quit", "bye"]:
            console.print("\n[bold green]學習夥伴:[/bold green] 很棒的討論！如果您想稍後再聊，請告訴我。")
            break

        conversation_history.append({"role": "user", "content": user_message})

        console.print("[yellow]學習夥伴正在思考...[/yellow]")
        try:
            # Invoke requires a dict. Pass necessary info.
            response = discussion_chain.invoke({
                "topic": topic,
                "message": user_message
                # Context is handled inside the chain by retrieving based on topic
            })
            console.print(f"\n[bold green]學習夥伴:[/bold green] {response}")
            conversation_history.append({"role": "assistant", "content": response})
        except Exception as e:
            console.print(f"[bold red]與學習夥伴溝通時發生錯誤: {e}[/bold red]")
            # Allow user to try again or exit
            if not Confirm.ask("是否重試? (Try again?)", default=True):
                 break
            continue # Skip turn increment if retrying


        turn_count += 1
        if turn_count == turn_limit:
             console.print("\n[yellow]我們已經進行了一次很好的討論。[/yellow]")
             if not Confirm.ask("是否繼續討論?", default=False):
                 break
             else:
                 turn_limit += 3 # Extend discussion if requested


    # Log discussion activity
    student_profile.learning_history.append({
        "activity_type": "同儕討論",
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
    module_title = module.get('title', '未命名章節')
    console.print(f"\n[bold cyan]===== 後測 (Post-Test): {module_title} =====[/bold cyan]")
    console.print("[yellow]正在測試您對章節內容的理解...[/yellow]")

    # Extract topic
    module_topic = module_title.split(":", 1)[-1].strip() if ":" in module_title else module_title
    current_level = student_profile.current_knowledge_level or "初學者"

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
            console.print("[bold red]生成的後測格式無效。[/bold red]")
            return None, []
    except Exception as e:
        console.print(f"[bold red]生成後測時發生錯誤: {e}[/bold red]")
        return None, []

    console.print(f"\n[bold green]{posttest_data.get('title', '後測')}[/bold green]")
    console.print(f"[italic]{posttest_data.get('description', '評估您的學習成果')}[/italic]\n")

    # Execute the test
    score = 0
    total_questions = len(posttest_data['questions'])
    results = []

    if total_questions == 0:
        console.print("[yellow]測驗中沒有問題。[/yellow]")
        return posttest_data, []

    for i, q_data in enumerate(posttest_data['questions']):
         # Validate question structure
        if not all(k in q_data for k in ["question", "choices", "correct_answer", "explanation", "difficulty"]):
             console.print(f"[yellow]跳過格式不正確的問題 {i+1}。[/yellow]")
             total_questions -= 1
             continue

        q = Question.model_validate(q_data) # Validate with Pydantic

        console.print(f"\n[bold]問題 {i+1}:[/bold] {q.question}")
        choice_map = {}
        for choice in q.choices:
            letter = choice.split('.')[0].strip().upper()
            console.print(f"  {choice}")
            choice_map[letter] = choice

        if not choice_map:
            console.print(f"[yellow]問題 {i+1} 沒有有效的選項，跳過。[/yellow]")
            total_questions -= 1
            continue

        valid_answer_keys = list(choice_map.keys())
        answer = Prompt.ask(
            f"\n您的答案 ({'/'.join(valid_answer_keys)})",
            choices=valid_answer_keys,
            show_choices=False
        ).upper()

        correct_letter = q.correct_answer.split('.')[0].strip().upper()
        is_correct = (answer == correct_letter)

        if is_correct:
            score += 1
            console.print("[bold green]正確！[/bold green]")
        else:
            console.print(f"[bold red]錯誤。正確答案是 {q.correct_answer}[/bold red]")

        console.print(f"[italic]解釋: {q.explanation}[/italic]")
        console.print(f"[dim]難度: {q.difficulty}[/dim]")

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
        console.print(f"\n[bold]測驗完成！您的分數：{score}/{total_questions} ({percentage:.1f}%)[/bold]")

        # Update level based on performance
        if percentage >= 80 and current_level != "高級":
            new_level = "中級" if current_level == "初學者" else "高級"
            console.print(f"[bold green]表現優異！您的知識水平已從 {current_level} 提升到 {new_level}。[/bold green]")
        elif percentage < 50 and current_level != "初學者":
             new_level = "中級" if current_level == "高級" else "初學者"
             console.print(f"[yellow]您可能需要更多練習。您的知識水平已從 {current_level} 調整為 {new_level}。[/yellow]")
        else:
            console.print(f"[yellow]您的知識水平保持在 {current_level}。[/yellow]")

    else:
        console.print("[yellow]無法計算分數或更新知識水平。[/yellow]")
        percentage = 0.0

    # Update student profile
    student_profile.current_knowledge_level = new_level # Update with the potentially new level
    student_profile.learning_history.append({
        "activity_type": "後測",
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
    module_title = module.get('title', '未命名章節')
    console.print(f"\n[bold cyan]===== 學習日誌 (Learning Log): {module_title} =====[/bold cyan]")

    module_topic = module_title.split(": ", 1)[-1].strip() if ": " in module_title else module_title
    module_summary = module.get('description', '無') # Use module description as summary

    # Format test results for the prompter chain
    score = sum(1 for r in posttest_results if r["is_correct"])
    total = len(posttest_results)
    test_results_str = f"分數: {score}/{total} 正確 ({(score/total*100):.1f}% if total > 0 else 0.0)\n"
    strengths = [r['question'] for r in posttest_results if r['is_correct']]
    weaknesses = [r['question'] for r in posttest_results if not r['is_correct']]
    test_results_str += "表現好的問題: " + (", ".join(strengths) if strengths else "無")
    test_results_str += "\n需要加強的問題: " + (", ".join(weaknesses) if weaknesses else "無")

    # Generate reflection prompts
    prompter_chain = create_learning_log_prompter(chat_model)
    reflection_prompts: Optional[str] = None
    console.print("[yellow]正在生成學習日誌提示...[/yellow]")
    try:
        reflection_prompts = prompter_chain.invoke({
            "module_title": module_topic,
            "module_summary": module_summary,
            "test_results": test_results_str
        })
        console.print(Markdown(reflection_prompts))
    except Exception as e:
        console.print(f"[bold red]生成日誌提示時發生錯誤: {e}[/bold red]")
        console.print("[yellow]請自行反思您學到了什麼、遇到的困難以及接下來的計畫。[/yellow]")


    # Get student reflections
    console.print("\n[bold yellow]請根據提示或自行反思，撰寫您的學習日誌（輸入 'done' 結束）：[/bold yellow]")
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
        console.print("[yellow]未輸入學習日誌內容。[/yellow]")
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
    console.print("[yellow]正在分析您的學習日誌...[/yellow]")
    try:
        analysis = analyzer_chain.invoke({
            "student_name": student_profile.name,
            "topic": module_topic,
            "log_content": log_content
        })
    except Exception as e:
        console.print(f"[bold red]分析學習日誌時發生錯誤: {e}[/bold red]")
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
        console.print("\n[bold green]學習日誌分析摘要:[/bold green]")
        console.print(f"  [bold]理解程度:[/bold] {analysis.get('understanding_level', '未評估')}")
        console.print(f"  [bold]偵測到的優勢:[/bold] {', '.join(new_strengths) if new_strengths else '無'}")
        console.print(f"  [bold]建議改進的領域:[/bold] {', '.join(new_weaknesses) if new_weaknesses else '無'}")
        console.print(f"  [bold]建議的下一步:[/bold] {', '.join(log.next_steps) if log.next_steps else '無'}")

    # Save the learning log
    log_path = os.path.join(LOGS_DIR, f"{log_id}.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log.model_dump_json(indent=4))
        console.print(f"[green]學習日誌已儲存至 {log_path}[/green]")
    except IOError as e:
        console.print(f"[bold red]無法儲存學習日誌 {log_path}: {e}[/bold red]")

    # Save the updated student profile
    save_profile(student_profile)

    return log