# -*- coding: UTF-8 -*-

# --- Standard Library Imports ---
import sys

# --- Third-party Library Imports ---
from rich.prompt import Confirm
from typing import Optional
# --- Local Application Imports ---
# Import necessary components from our modules
from config import console, CHAT_MODEL, EMBEDDING # Core objects
from rag_setup import RETRIEVER # Initialized retriever
from models import StudentProfile # Data model for type hinting/access
from profile_manager import manage_student_profile # Function to get profile
from workflow import ( # Import all workflow steps
    conduct_learning_style_survey,
    administer_pretest,
    generate_learning_path,
    deliver_module_content,
    engage_peer_discussion,
    administer_posttest,
    create_learning_log
)

# --- Main Application Function ---
def main():
    """Main function to run the RAG Scaffolding Education System."""
    console.print("[bold cyan]=====================================[/bold cyan]")
    console.print("[bold cyan]== RAG 鷹架教育系統 ==[/bold cyan]")
    console.print("[bold cyan]=====================================[/bold cyan]\n")


    student_profile: Optional[StudentProfile] = manage_student_profile()
    if not student_profile:
        console.print("[bold red]無法載入或建立學生檔案，程式即將結束。[/bold red]")
        sys.exit(1)
    console.print(f"\n[bold green]歡迎回來, {student_profile.name}! [/bold green]")


    student_profile = conduct_learning_style_survey(CHAT_MODEL, student_profile)
    # Profile is saved within the function


    # 3. Administer Pre-test

    pretest_data, pretest_results, _ = administer_pretest(CHAT_MODEL, RETRIEVER, student_profile)
    # Profile and knowledge level updated and saved within the function
    if pretest_data is None:
         console.print("[bold red]無法執行前測。[/bold red]")
         # Decide how to proceed: exit, use default level, etc.
         if not Confirm.ask("前測失敗，是否仍要嘗試生成學習路徑? ", default=False):
              sys.exit(1)
         # If continuing, ensure profile has a default level
         if not student_profile.current_knowledge_level:
             student_profile.current_knowledge_level = "初學者"


    # 4. Generate Learning Path
    learning_path = generate_learning_path(CHAT_MODEL, RETRIEVER, student_profile, pretest_results)


    # 5. Learning Loop (Modules)
    num_modules = len(learning_path['modules'])
    for module_index, module in enumerate(learning_path['modules']):
        module_title = module.get('title', f'章節 {module_index + 1}')
        console.print(f"\n[bold #FFD700]===== 開始章節 {module_index + 1}/{num_modules}: {module_title} =====[/bold #FFD700]") # Gold color

        # Check if module structure is valid enough
        if not isinstance(module, dict) or not module_title:
             console.print(f"[yellow]章節 {module_index + 1} 格式無效，跳過。[/yellow]")
             continue

        # Ask user to proceed
        proceed = Confirm.ask(f"準備好開始此章節嗎? ", default=True)
        if not proceed:
            console.print("[yellow]跳過此章節。[/yellow]")
            continue

        # 5a. Deliver Module Content
        console.print(f"\n[bold]-- 章節內容 ({module_title}) --[/bold]")
        _ = deliver_module_content(CHAT_MODEL, RETRIEVER, student_profile, module)
        # Profile history updated within function

        # 5b. Engage in Peer Discussion
        console.print(f"\n[bold]-- 同儕討論 ({module_title}) --[/bold]")
        module_topic = module_title.split(": ", 1)[-1].strip() if ": " in module_title else module_title
        if Confirm.ask("是否要進行同儕討論? ", default=True):
             _ = engage_peer_discussion(CHAT_MODEL, RETRIEVER, module_topic, student_profile)
             # Profile history updated within function
        else:
             console.print("[yellow]跳過同儕討論。[/yellow]")


        # 5c. Administer Post-test
        console.print(f"\n[bold]-- 後測 ({module_title}) --[/bold]")
        posttest_data, posttest_results = administer_posttest(CHAT_MODEL, RETRIEVER, module, student_profile)
        # Profile knowledge level and history updated within function
        if posttest_data is None:
             console.print("[yellow]無法執行後測。[/yellow]")
             # Continue to log without test results, or handle differently


        # 5d. Create Learning Log
        console.print(f"\n[bold]-- 學習日誌 ({module_title}) --[/bold]")
        if Confirm.ask("是否要建立學習日誌?", default=True):
            _ = create_learning_log(CHAT_MODEL, module, posttest_results, student_profile)
            # Profile strengths/weaknesses and history updated within function
        else:
             console.print("[yellow]跳過學習日誌。[/yellow]")


        # Ask to continue to next module
        if module_index < num_modules - 1:
            continue_learning = Confirm.ask("是否繼續進行下一個章節?", default=True)
            if not continue_learning:
                console.print("[yellow]學習暫停。[/yellow]")
                break
        else:
            console.print("[green]已完成所有章節！[/green]")


    # 6. Final Summary
    console.print("\n[bold green]===== 學習路徑完成  =====[/bold green]")
    console.print("[yellow]感謝您使用 RAG 鷹架教育系統！[/yellow]")
    console.print(f"[yellow]您目前的知識水平評估為: [bold]{student_profile.current_knowledge_level}[/bold][/yellow]")

    console.print("\n[bold]您的學習旅程摘要:[/bold]")
    total_activities = len(student_profile.learning_history)
    console.print(f" - 完成了 {total_activities} 項學習活動 (Completed {total_activities} learning activities)")
    # Display consolidated strengths/weaknesses from the profile
    strengths_str = ', '.join(student_profile.strengths) if student_profile.strengths else '尚未確定'
    weaknesses_str = ', '.join(student_profile.areas_for_improvement) if student_profile.areas_for_improvement else '尚未確定'
    console.print(f" - 目前評估的優勢: {strengths_str}")
    console.print(f" - 建議加強的領域: {weaknesses_str}")

    console.print("\n[bold]持續學習的建議 ")
    level = student_profile.current_knowledge_level
    if level == "初學者":
        console.print(" - 專注於掌握基礎概念 ")
        console.print(" - 多練習初學者到中級的範例 ")
    elif level == "中級":
        console.print(" - 加深對複雜主題的理解 ")
        console.print(" - 開始將概念應用於實際問題")
    else: # 高級
        console.print(" - 探索該領域的專業主題 ")
        console.print(" - 考慮教學或指導他人以鞏固您的知識 ")

# --- Entry Point Check ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print("\n[bold red]程式執行時發生未預期的錯誤:[/bold red]")
        console.print_exception(show_locals=False) # Print traceback
    finally:
        console.print("\n[bold]程式結束。(Program finished.)[/bold]")