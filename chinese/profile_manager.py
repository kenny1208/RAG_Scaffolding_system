# -*- coding: big5 -*-

import os
import json
import uuid
import pathlib
from glob import glob
from rich.prompt import Prompt
from typing import Optional
# Import necessary components
from models import StudentProfile
from config import PROFILES_DIR, console

def create_new_profile() -> StudentProfile:
    """Creates a new, basic student profile."""
    name = Prompt.ask("請輸入學生姓名 (Enter student name)")
    student_id = str(uuid.uuid4())[:8] # Ensure unique enough ID

    profile = StudentProfile(
        id=student_id,
        name=name,
        learning_style="", # Initialize as empty
        current_knowledge_level="", # Initialize as empty
        strengths=[],
        areas_for_improvement=[],
        interests=[],
        learning_history=[]
    )

    profile_path = os.path.join(PROFILES_DIR, f"{student_id}.json")
    try:
        with open(profile_path, "w", encoding="cp950") as f: 
            f.write(profile.model_dump_json(indent=4))
        console.print(f"[green]新學生檔案已建立並儲存至 {profile_path}[/green]")
    except IOError as e:
        console.print(f"[bold red]無法儲存學生檔案 {profile_path}: {e}[/bold red]")
        # Handle error appropriately, maybe return None or raise exception

    return profile

def load_profile(profile_path: str) -> Optional[StudentProfile]:
    """Loads a student profile from a JSON file."""
    if not os.path.exists(profile_path) or os.path.getsize(profile_path) == 0:
        console.print(f"[red]檔案不存在或為空: {profile_path}[/red]")
        return None

    try:
        with open(profile_path, "r", encoding="cp950") as f: 
            data = json.load(f)
        profile = StudentProfile.model_validate(data)
        return profile
    except json.JSONDecodeError:
        console.print(f"[red]無法解析 JSON 檔案: {profile_path}[/red]")
        return None
    except Exception as e: # Catch Pydantic validation errors too
        console.print(f"[red]載入或驗證檔案時發生錯誤 {profile_path}: {e}[/red]")
        return None

def save_profile(student_profile: StudentProfile):
    """Saves the student profile to its JSON file."""
    if not student_profile or not student_profile.id:
        console.print("[red]無效的學生檔案，無法儲存。[/red]")
        return

    profile_path = os.path.join(PROFILES_DIR, f"{student_profile.id}.json")
    try:
        with open(profile_path, "w", encoding="cp950") as f:
            f.write(student_profile.model_dump_json(indent=4))
        # console.print(f"[blue]學生檔案已更新: {profile_path}[/blue]") # Optional: reduce verbosity
    except IOError as e:
        console.print(f"[bold red]無法儲存學生檔案 {profile_path}: {e}[/bold red]")


def manage_student_profile() -> Optional[StudentProfile]:
    """Manages student profile selection or creation."""
    profiles = glob(os.path.join(PROFILES_DIR, "*.json")) # Use os.path.join

    if profiles:
        console.print("[bold]選擇現有檔案或建立新檔案:[/bold]")
        console.print("0. 建立新學生檔案")

        profile_options = {}
        for i, profile_path in enumerate(profiles, 1):
            # Attempt to load name from JSON for display
            profile_name = f"檔案: {pathlib.Path(profile_path).stem}" # Default
            temp_profile = load_profile(profile_path) # Temporarily load to get name
            if temp_profile:
                profile_name = temp_profile.name
            console.print(f"{i}. {profile_name}")
            profile_options[str(i)] = profile_path # Store path with index

        # Generate valid choices including '0'
        valid_choices = [str(i) for i in range(len(profiles) + 1)]

        choice_str = Prompt.ask(
            "請輸入您的選擇",
            choices=valid_choices,
            default="0" # Make creating new the default maybe?
        )
        choice = int(choice_str)

        if choice == 0:
            return create_new_profile()
        else:
            selected_path = profile_options[choice_str]
            profile = load_profile(selected_path)
            # If loading failed (e.g., corrupted file), offer to create new
            if not profile:
                 console.print("[yellow]載入所選檔案時發生錯誤。[/yellow]")
                 if Prompt.ask("是否要建立新檔案 ?", choices=["y", "n"], default="y") == "y":
                     return create_new_profile()
                 else:
                     return None # Or exit, or retry selection
            return profile # Return the successfully loaded profile
    else:
        console.print("[yellow]找不到現有學生檔案。[/yellow]")
        return create_new_profile()