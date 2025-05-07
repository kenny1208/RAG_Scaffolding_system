#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import json
import logging

logger = logging.getLogger(__name__)

def ensure_dir_exists(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def save_json(data, filepath):
    """Save data as JSON to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        return False

def load_json(filepath):
    """Load JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {str(e)}")
        return None

def save_text(text, filepath):
    """Save text to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved text to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving text to {filepath}: {str(e)}")
        return False

def load_text(filepath):
    """Load text from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Loaded text from {filepath}")
        return text
    except Exception as e:
        logger.error(f"Error loading text from {filepath}: {str(e)}")
        return None

def list_files(directory, extension=None):
    """List files in a directory, optionally filtered by extension."""
    try:
        files = os.listdir(directory)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        logger.info(f"Listed {len(files)} files in {directory}")
        return files
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {str(e)}")
        return []

def file_exists(filepath):
    """Check if a file exists."""
    return os.path.isfile(filepath)

def remove_file(filepath):
    """Remove a file."""
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
            logger.info(f"Removed file: {filepath}")
            return True
        else:
            logger.warning(f"File does not exist: {filepath}")
            return False
    except Exception as e:
        logger.error(f"Error removing file {filepath}: {str(e)}")
        return False
