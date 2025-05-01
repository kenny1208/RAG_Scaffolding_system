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
    name = Prompt.ask("�п�J�ǥͩm�W (Enter student name)")
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
        console.print(f"[green]�s�ǥ��ɮפw�إߨ��x�s�� {profile_path}[/green]")
    except IOError as e:
        console.print(f"[bold red]�L�k�x�s�ǥ��ɮ� {profile_path}: {e}[/bold red]")
        # Handle error appropriately, maybe return None or raise exception

    return profile

def load_profile(profile_path: str) -> Optional[StudentProfile]:
    """Loads a student profile from a JSON file."""
    if not os.path.exists(profile_path) or os.path.getsize(profile_path) == 0:
        console.print(f"[red]�ɮפ��s�b�ά���: {profile_path}[/red]")
        return None

    try:
        with open(profile_path, "r", encoding="cp950") as f: 
            data = json.load(f)
        profile = StudentProfile.model_validate(data)
        return profile
    except json.JSONDecodeError:
        console.print(f"[red]�L�k�ѪR JSON �ɮ�: {profile_path}[/red]")
        return None
    except Exception as e: # Catch Pydantic validation errors too
        console.print(f"[red]���J�������ɮ׮ɵo�Ϳ��~ {profile_path}: {e}[/red]")
        return None

def save_profile(student_profile: StudentProfile):
    """Saves the student profile to its JSON file."""
    if not student_profile or not student_profile.id:
        console.print("[red]�L�Ī��ǥ��ɮסA�L�k�x�s�C[/red]")
        return

    profile_path = os.path.join(PROFILES_DIR, f"{student_profile.id}.json")
    try:
        with open(profile_path, "w", encoding="cp950") as f:
            f.write(student_profile.model_dump_json(indent=4))
        # console.print(f"[blue]�ǥ��ɮפw��s: {profile_path}[/blue]") # Optional: reduce verbosity
    except IOError as e:
        console.print(f"[bold red]�L�k�x�s�ǥ��ɮ� {profile_path}: {e}[/bold red]")


def manage_student_profile() -> Optional[StudentProfile]:
    """Manages student profile selection or creation."""
    profiles = glob(os.path.join(PROFILES_DIR, "*.json")) # Use os.path.join

    if profiles:
        console.print("[bold]��ܲ{���ɮשΫإ߷s�ɮ�:[/bold]")
        console.print("0. �إ߷s�ǥ��ɮ�")

        profile_options = {}
        for i, profile_path in enumerate(profiles, 1):
            # Attempt to load name from JSON for display
            profile_name = f"�ɮ�: {pathlib.Path(profile_path).stem}" # Default
            temp_profile = load_profile(profile_path) # Temporarily load to get name
            if temp_profile:
                profile_name = temp_profile.name
            console.print(f"{i}. {profile_name}")
            profile_options[str(i)] = profile_path # Store path with index

        # Generate valid choices including '0'
        valid_choices = [str(i) for i in range(len(profiles) + 1)]

        choice_str = Prompt.ask(
            "�п�J�z�����",
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
                 console.print("[yellow]���J�ҿ��ɮ׮ɵo�Ϳ��~�C[/yellow]")
                 if Prompt.ask("�O�_�n�إ߷s�ɮ� ?", choices=["y", "n"], default="y") == "y":
                     return create_new_profile()
                 else:
                     return None # Or exit, or retry selection
            return profile # Return the successfully loaded profile
    else:
        console.print("[yellow]�䤣��{���ǥ��ɮסC[/yellow]")
        return create_new_profile()