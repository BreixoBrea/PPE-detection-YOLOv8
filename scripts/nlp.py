"""
NLP Module for Safety Regulation Extraction

This module processes PDF documents containing safety regulations and extracts
PPE (Personal Protective Equipment) requirements using Google's Gemini API.
It provides both PDF-to-JSON conversion and loading of existing JSON files.

Author: Breixo Brea
Date: 2025
"""

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Dict, Any, Optional
from pathlib import Path
from PyPDF2 import PdfReader
from google import genai


# ============================================================================
# CONFIGURATION
# ============================================================================

class NLPConfig:
    """Configuration for NLP processing and API settings."""
    
    # Output directory for generated JSON files
    JSON_OUTPUT_DIR: str = "../data/json"
    
    # Gemini API configuration
    GEMINI_MODEL: str = "gemini-2.5-flash"
    RESPONSE_MIME_TYPE: str = "application/json"
    
    # Valid PPE names (must match exactly in output)
    VALID_PPE_NAMES: tuple = (
        "casco de seguridad",
        "chaleco reflectante",
        "guantes de proteccion",
        "botas"
    )
    
    # UI configuration
    WINDOW_TOPMOST: bool = True


# ============================================================================
# API CLIENT INITIALIZATION
# ============================================================================

def initialize_gemini_client() -> genai.Client:
    """
    Initialize the Gemini API client.
    
    Requires GEMINI_API_KEY environment variable to be set.
    
    Returns:
        Initialized Gemini client
        
    Raises:
        RuntimeError: If client initialization fails or API key is missing
    """
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not found. "
            "Please set it before running this script."
        )
    
    try:
        client = genai.Client()
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")


# Initialize global client
try:
    client = initialize_gemini_client()
except RuntimeError as e:
    print(f"ERROR: {str(e)}")
    sys.exit(1)


# ============================================================================
# USER INTERFACE FUNCTIONS
# ============================================================================

def ask_user_preference() -> bool:
    """
    Ask user if they want to load a new regulation from PDF.
    
    Returns:
        True if user wants to process PDF, False to use existing JSON
    """
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Aggressive approach to prevent window manifestation
        root.update_idletasks()
        root.overrideredirect(True)  # Hide title bar
        root.resizable(False, False)
        root.wm_attributes("-topmost", NLPConfig.WINDOW_TOPMOST)
        
        response = messagebox.askyesno(
            "Safety Regulations",
            "Do you want to load a new regulation from a PDF file?"
        )
        
    finally:
        root.destroy()
        
    return response


def select_pdf_file() -> Optional[str]:
    """
    Open file dialog to select a PDF file.
    
    Returns:
        Path to selected PDF file, or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    
    try:
        pdf_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        return pdf_path if pdf_path else None
        
    finally:
        root.destroy()


def select_json_file() -> Optional[str]:
    """
    Open file dialog to select an existing JSON file.
    
    Returns:
        Path to selected JSON file, or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    
    try:
        json_path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        return json_path if json_path else None
        
    finally:
        root.destroy()


# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If text extraction fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        reader = PdfReader(pdf_path)
        text_content = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"
        
        if not text_content.strip():
            raise RuntimeError("PDF contains no extractable text")
        
        return text_content
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")


# ============================================================================
# LLM PROCESSING
# ============================================================================

def build_extraction_prompt(pdf_text: str, valid_ppe_names: tuple) -> str:
    """
    Build the prompt for Gemini to extract PPE requirements.
    
    Args:
        pdf_text: Extracted text from PDF
        valid_ppe_names: Tuple of valid PPE names
        
    Returns:
        Formatted prompt string
    """
    ppe_list = "\n".join(f'- "{name}"' for name in valid_ppe_names)
    
    prompt = f"""
The following text has been extracted from a PDF document containing safety regulations.

**Task:** Extract the PPE (Personal Protective Equipment) requirements for each area and task in JSON format.

**JSON Structure:**
The JSON must include:
1. `document_id`: Extract the document ID from the text (or create a placeholder if missing).
2. `titulo`: Set a title for the document, starting with "Normativa de Seguridad Industrial - ".
3. `fecha`: Extract the date from the text in "DD-MM-YYYY" format.
4. `areas`: A list of areas, each with:
   - `nombre_area`: Name of the area.
   - `requisitos_epi`: A list of required PPEs for the area, each with:
     - `nombre_epi`: Name of the PPE.
     - `uso_obligatorio`: Boolean value indicating mandatory usage.
5. `tareas`: A list of tasks, each with:
   - `nombre_tarea`: Name of the task.
   - `requisitos_epi`: A list of required PPEs for the task, with the same structure as above.

Replace all placeholders with appropriate values in Spanish.

**CRITICAL - VALID PPE NAMES:**
You must use ONLY the following valid PPE names (`nombre_epi`):

{ppe_list}

**Conversion Rules:**
- Any type of helmet → "casco de seguridad"
- Any type of gloves → "guantes de proteccion"
- Any protective footwear → "botas"
- Any reflective vest → "chaleco reflectante"

If a PPE does not match any category, IGNORE it and do NOT include it in the JSON.

**Additional Rules:**
- The output must be in **Spanish**.
- For each `requisitos_epi`, the maximum number of PPEs with the same `nombre_epi` is 1.
- Consider `uso_obligatorio`: false ONLY if explicitly mentioned as optional or not mandatory.

**Input Text:**
---
{pdf_text}
---

Return ONLY the JSON object, without any markdown formatting or explanations.
"""
    
    return prompt


def extract_requirements_with_llm(pdf_text: str, config: NLPConfig) -> str:
    """
    Send text to Gemini API to extract PPE requirements in JSON format.
    
    Args:
        pdf_text: Extracted text from PDF document
        config: NLP configuration object
        
    Returns:
        JSON string with extracted requirements
        
    Raises:
        RuntimeError: If API call fails
    """
    prompt = build_extraction_prompt(pdf_text, config.VALID_PPE_NAMES)
    
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config={
                'response_mime_type': config.RESPONSE_MIME_TYPE
            }
        )
        
        return response.text
        
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {str(e)}")


def clean_json_response(json_string: str) -> str:
    """
    Remove markdown code blocks from JSON response if present.
    
    Args:
        json_string: Raw JSON string from API
        
    Returns:
        Cleaned JSON string
    """
    cleaned = json_string.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    return cleaned.strip()


def parse_json_response(json_string: str) -> Dict[str, Any]:
    """
    Parse and validate JSON response from API.
    
    Args:
        json_string: JSON string to parse
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If JSON is invalid or malformed
    """
    cleaned_json = clean_json_response(json_string)
    
    try:
        json_data = json.loads(cleaned_json)
        
        # Sometimes Gemini wraps response in a list
        if isinstance(json_data, list) and len(json_data) == 1:
            json_data = json_data[0]
        
        if not isinstance(json_data, dict):
            raise ValueError("JSON response is not a dictionary")
        
        return json_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}\nResponse: {json_string}")


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def save_json_file(json_data: Dict[str, Any], pdf_path: str, output_dir: str) -> str:
    """
    Save JSON data to file with same name as source PDF.
    
    Uses relative path from script location to ensure portability.
    
    Args:
        json_data: Dictionary to save as JSON
        pdf_path: Path to original PDF file
        output_dir: Relative directory path to save JSON file
        
    Returns:
        Absolute path to saved JSON file
    """
    # Get absolute path of the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Join script path with desired output directory
    full_output_dir = os.path.join(script_dir, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Generate output filename from PDF name
    pdf_name = Path(pdf_path).stem
    output_path = os.path.join(full_output_dir, f"{pdf_name}.json")
    
    # Save JSON with proper formatting
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)
    
    # Display absolute path for user reference
    absolute_path = os.path.abspath(output_path)
    print(f"✓ JSON saved to: {absolute_path}")
    
    return output_path


def load_json_file(json_path: str) -> Dict[str, Any]:
    """
    Load and parse JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON is invalid
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        # Handle wrapped list format
        if isinstance(json_data, list) and len(json_data) == 1:
            json_data = json_data[0]
        
        if not isinstance(json_data, dict):
            raise ValueError("JSON file does not contain a valid dictionary")
        
        return json_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {str(e)}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def get_regulation_data() -> Dict[str, Any]:
    """
    Main function to obtain regulation data from PDF or existing JSON.
    
    Workflow:
    1. Ask user if they want to process PDF or load existing JSON
    2. If PDF: extract text → send to LLM → parse JSON → save
    3. If JSON: load existing file
    
    Returns:
        Dictionary containing regulation data
        
    Raises:
        SystemExit: If user cancels or critical error occurs
    """
    config = NLPConfig()
    
    # Ask user preference
    use_pdf = ask_user_preference()
    
    if use_pdf:
        # === PDF WORKFLOW ===
        print("\n" + "="*60)
        print("PDF TO JSON WORKFLOW")
        print("="*60)
        
        # Select PDF file
        pdf_path = select_pdf_file()
        if not pdf_path:
            print("ERROR: No PDF file selected.")
            sys.exit(1)
        
        print(f"\n📄 Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            print("⏳ Extracting text from PDF...")
            pdf_text = extract_text_from_pdf(pdf_path)
            print(f"✓ Extracted {len(pdf_text)} characters")
            
            # Send to LLM for processing
            print("⏳ Sending to Gemini API for analysis...")
            json_response = extract_requirements_with_llm(pdf_text, config)
            print("✓ Received response from API")
            
            # Parse JSON response
            print("⏳ Parsing JSON response...")
            json_data = parse_json_response(json_response)
            print("✓ JSON parsed successfully")
            
            # Save JSON file
            print("⏳ Saving JSON file...")
            save_json_file(json_data, pdf_path, config.JSON_OUTPUT_DIR)
            
            print("\n" + "="*60)
            print("PDF PROCESSING COMPLETED SUCCESSFULLY ✓")
            print("="*60)
            
            return json_data
            
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"\nERROR: {str(e)}")
            sys.exit(1)
    
    else:
        # === JSON WORKFLOW ===
        print("\n" + "="*60)
        print("LOAD EXISTING JSON WORKFLOW")
        print("="*60)
        
        # Select JSON file
        json_path = select_json_file()
        if not json_path:
            print("ERROR: No JSON file selected.")
            sys.exit(1)
        
        print(f"\n📁 Loading JSON: {json_path}")
        
        try:
            json_data = load_json_file(json_path)
            print("✓ JSON loaded successfully")
            
            print("\n" + "="*60)
            print("JSON LOADING COMPLETED SUCCESSFULLY ✓")
            print("="*60)
            
            return json_data
            
        except (FileNotFoundError, ValueError) as e:
            print(f"\nERROR: {str(e)}")
            sys.exit(1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main entry point for standalone execution."""
    try:
        regulation_data = get_regulation_data()
        
        # Optional: Display summary
        print("\n" + "="*60)
        print("REGULATION DATA SUMMARY")
        print("="*60)
        
        if "titulo" in regulation_data:
            print(f"Title: {regulation_data['titulo']}")
        if "fecha" in regulation_data:
            print(f"Date: {regulation_data['fecha']}")
        if "areas" in regulation_data:
            print(f"Areas: {len(regulation_data['areas'])}")
        if "tareas" in regulation_data:
            print(f"Tasks: {len(regulation_data['tareas'])}")
        
        print("\n✓ Process completed successfully")
        
    except Exception as e:
        print(f"\nERROR: Unexpected error - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()