# -*- coding: big5 -*-

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
    console.print("[bold cyan]== RAG �N�[�Ш|�t�� ==[/bold cyan]")
    console.print("[bold cyan]=====================================[/bold cyan]\n")


    student_profile: Optional[StudentProfile] = manage_student_profile()
    if not student_profile:
        console.print("[bold red]�L�k���J�Ϋإ߾ǥ��ɮסA�{���Y�N�����C[/bold red]")
        sys.exit(1)
    console.print(f"\n[bold green]�w��^��, {student_profile.name}! [/bold green]")


    console.print("\n[bold]-- �ǲ߭������ --[/bold]")
    student_profile = conduct_learning_style_survey(CHAT_MODEL, student_profile)
    # Profile is saved within the function


    # 3. Administer Pre-test
    console.print("\n[bold]-- �e�� --[/bold]")
    pretest_data, pretest_results, _ = administer_pretest(CHAT_MODEL, RETRIEVER, student_profile)
    # Profile and knowledge level updated and saved within the function
    if pretest_data is None:
         console.print("[bold red]�L�k����e���C[/bold red]")
         # Decide how to proceed: exit, use default level, etc.
         if not Confirm.ask("�e�����ѡA�O�_���n���եͦ��ǲ߸��|? ", default=False):
              sys.exit(1)
         # If continuing, ensure profile has a default level
         if not student_profile.current_knowledge_level:
             student_profile.current_knowledge_level = "��Ǫ�"


    # 4. Generate Learning Path
    console.print("\n[bold]-- �ͦ��ǲ߸��| --[/bold]")
    learning_path = generate_learning_path(CHAT_MODEL, RETRIEVER, student_profile, pretest_results)


    # 5. Learning Loop (Modules)
    console.print("\n[bold]-- ���`�ǲ� --[/bold]")
    num_modules = len(learning_path['modules'])
    for module_index, module in enumerate(learning_path['modules']):
        module_title = module.get('title', f'���` {module_index + 1}')
        console.print(f"\n[bold #FFD700]===== �}�l���` {module_index + 1}/{num_modules}: {module_title} =====[/bold #FFD700]") # Gold color

        # Check if module structure is valid enough
        if not isinstance(module, dict) or not module_title:
             console.print(f"[yellow]���` {module_index + 1} �榡�L�ġA���L�C[/yellow]")
             continue

        # Ask user to proceed
        proceed = Confirm.ask(f"�ǳƦn�}�l�����`��? ", default=True)
        if not proceed:
            console.print("[yellow]���L�����`�C[/yellow]")
            continue

        # 5a. Deliver Module Content
        console.print(f"\n[bold]-- ���`���e ({module_title}) --[/bold]")
        _ = deliver_module_content(CHAT_MODEL, RETRIEVER, student_profile, module)
        # Profile history updated within function

        # 5b. Engage in Peer Discussion
        console.print(f"\n[bold]-- �P���Q�� ({module_title}) --[/bold]")
        module_topic = module_title.split(": ", 1)[-1].strip() if ": " in module_title else module_title
        if Confirm.ask("�O�_�n�i��P���Q��? ", default=True):
             _ = engage_peer_discussion(CHAT_MODEL, RETRIEVER, module_topic, student_profile)
             # Profile history updated within function
        else:
             console.print("[yellow]���L�P���Q�סC[/yellow]")


        # 5c. Administer Post-test
        console.print(f"\n[bold]-- ��� ({module_title}) --[/bold]")
        posttest_data, posttest_results = administer_posttest(CHAT_MODEL, RETRIEVER, module, student_profile)
        # Profile knowledge level and history updated within function
        if posttest_data is None:
             console.print("[yellow]�L�k�������C[/yellow]")
             # Continue to log without test results, or handle differently


        # 5d. Create Learning Log
        console.print(f"\n[bold]-- �ǲߤ�x ({module_title}) --[/bold]")
        if Confirm.ask("�O�_�n�إ߾ǲߤ�x?", default=True):
            _ = create_learning_log(CHAT_MODEL, module, posttest_results, student_profile)
            # Profile strengths/weaknesses and history updated within function
        else:
             console.print("[yellow]���L�ǲߤ�x�C[/yellow]")


        # Ask to continue to next module
        if module_index < num_modules - 1:
            continue_learning = Confirm.ask("�O�_�~��i��U�@�ӳ��`?", default=True)
            if not continue_learning:
                console.print("[yellow]�ǲ߼Ȱ��C[/yellow]")
                break
        else:
            console.print("[green]�w�����Ҧ����`�I[/green]")


    # 6. Final Summary
    console.print("\n[bold green]===== �ǲ߸��|���� (Learning Path Complete) =====[/bold green]")
    console.print("[yellow]�P�±z�ϥ� RAG �N�[�Ш|�t�ΡI (Thank you for using the system!)[/yellow]")
    console.print(f"[yellow]�z�ثe�����Ѥ���������: [bold]{student_profile.current_knowledge_level}[/bold][/yellow]")

    console.print("\n[bold]�z���ǲ߮ȵ{�K�n (Summary of your learning journey):[/bold]")
    total_activities = len(student_profile.learning_history)
    console.print(f" - �����F {total_activities} ���ǲ߬��� (Completed {total_activities} learning activities)")
    # Display consolidated strengths/weaknesses from the profile
    strengths_str = ', '.join(student_profile.strengths) if student_profile.strengths else '�|���T�w'
    weaknesses_str = ', '.join(student_profile.areas_for_improvement) if student_profile.areas_for_improvement else '�|���T�w'
    console.print(f" - �ثe�������u��: {strengths_str}")
    console.print(f" - ��ĳ�[�j�����: {weaknesses_str}")

    console.print("\n[bold]����ǲߪ���ĳ (Suggestions for continued learning):[/bold]")
    level = student_profile.current_knowledge_level
    if level == "��Ǫ�":
        console.print(" - �M�`��x����¦���� (Focus on mastering fundamental concepts)")
        console.print(" - �h�m�ߪ�Ǫ̨줤�Ū��d�� (Practice more beginner-to-intermediate examples)")
    elif level == "����":
        console.print(" - �[�`������D�D���z�� (Deepen understanding of complex topics)")
        console.print(" - �}�l�N�������Ω��ڰ��D (Start applying concepts to practical problems)")
    else: # ����
        console.print(" - �����ӻ�쪺�M�~�D�D (Explore advanced topics in the field)")
        console.print(" - �Ҽ{�оǩΫ��ɥL�H�H�d�T�z������ (Consider teaching or mentoring others)")

# --- Entry Point Check ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print("\n[bold red]�{������ɵo�ͥ��w�������~:[/bold red]")
        console.print_exception(show_locals=False) # Print traceback
    finally:
        console.print("\n[bold]�{�������C(Program finished.)[/bold]")