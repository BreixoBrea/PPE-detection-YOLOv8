"""
Web Dashboard for Safety Regulation Extraction and PPE Detection

Flask-based web interface for processing PDF documents, extracting
PPE requirements using NLP, and running PPE detection on images using CV.

Author: Breixo Brea
Date: 2025
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import json
import csv
from pathlib import Path
from werkzeug.utils import secure_filename
import shutil

# Import the NLP module functions
try:
    from scripts.nlp import (
        extract_text_from_pdf,
        extract_requirements_with_llm,
        parse_json_response,
        NLPConfig,
        initialize_gemini_client
    )
except ImportError:
    print("ERROR: Could not import NLP module. Make sure nlp_module_improved.py is in the same directory.")
    sys.exit(1)

# Import the CV module functions
try:
    from scripts.cv import (
        load_detection_model,
        run_ppe_detection,
        get_detection_summary,
        CVConfig
    )
except ImportError:
    print("ERROR: Could not import CV module. Make sure cv_module_improved.py is in the same directory.")
    sys.exit(1)


# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)

# Use absolute paths to avoid issues with relative paths in different environments
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['JSON_FOLDER'] = os.path.join(BASE_DIR, 'data', 'json')
app.config['IMAGE_FOLDER'] = os.path.join(BASE_DIR, 'images')
app.config['DETECTION_FOLDER'] = os.path.join(BASE_DIR, 'detections')

# Allowed file extensions
ALLOWED_PDF_EXTENSIONS = {'pdf'}
ALLOWED_JSON_EXTENSIONS = {'json'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['JSON_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)

# Initialize NLP config
nlp_config = NLPConfig()

# Initialize CV config
cv_config = CVConfig()
cv_config.OUTPUT_DIR = app.config['DETECTION_FOLDER']

# Initialize Gemini client
try:
    client = initialize_gemini_client()
except RuntimeError as e:
    print(f"ERROR: {str(e)}")
    print("Please set GEMINI_API_KEY environment variable before running the dashboard.")
    sys.exit(1)

# Load YOLO model (lazy loading - only when needed)
yolo_model = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename, file_type='pdf'):
    """Check if file extension is allowed based on file type."""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'pdf':
        return ext in ALLOWED_PDF_EXTENSIONS
    elif file_type == 'json':
        return ext in ALLOWED_JSON_EXTENSIONS
    elif file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    
    return False


def get_yolo_model():
    """Lazy load YOLO model."""
    global yolo_model
    if yolo_model is None:
        try:
            yolo_model = load_detection_model(cv_config.MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    return yolo_model


def get_saved_json_files():
    """Get list of all saved JSON files."""
    json_files = []
    json_dir = app.config['JSON_FOLDER']
    
    if os.path.exists(json_dir):
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(json_dir, filename)
                file_size = os.path.getsize(filepath)
                modified_time = os.path.getmtime(filepath)
                
                json_files.append({
                    'name': filename,
                    'size': f"{file_size / 1024:.1f} KB",
                    'modified': modified_time
                })
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: x['modified'], reverse=True)
    
    return json_files


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render main dashboard page."""
    return render_template('index.html', saved_files=get_saved_json_files())


@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and processing."""
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename, 'pdf'):
        return jsonify({'success': False, 'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Process with LLM
        json_response = extract_requirements_with_llm(pdf_text, nlp_config)
        
        # Parse JSON
        json_data = parse_json_response(json_response)
        
        # Save JSON file
        pdf_name = Path(pdf_path).stem
        json_filename = f"{pdf_name}.json"
        json_path = os.path.join(app.config['JSON_FOLDER'], json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        # Clean up uploaded PDF
        os.remove(pdf_path)
        
        return jsonify({
            'success': True,
            'message': 'PDF processed successfully',
            'filename': json_filename,
            'data': json_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload-json', methods=['POST'])
def upload_json():
    """Handle JSON file upload."""
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename, 'json'):
        return jsonify({'success': False, 'error': 'Only JSON files are allowed'}), 400
    
    try:
        # Read and parse JSON
        json_data = json.load(file)
        
        # Save to JSON folder
        filename = secure_filename(file.filename)
        json_path = os.path.join(app.config['JSON_FOLDER'], filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        return jsonify({
            'success': True,
            'message': 'JSON uploaded successfully',
            'filename': filename,
            'data': json_data
        })
        
    except json.JSONDecodeError:
        return jsonify({'success': False, 'error': 'Invalid JSON format'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get-json/<filename>')
def get_json(filename):
    """Retrieve JSON file content."""
    try:
        json_path = os.path.join(app.config['JSON_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(json_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        return jsonify({'success': True, 'data': json_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download-json/<filename>')
def download_json(filename):
    """Download JSON file."""
    try:
        json_path = os.path.join(app.config['JSON_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(json_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        return send_file(json_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete-json/<filename>', methods=['DELETE'])
def delete_json(filename):
    """Delete JSON file."""
    try:
        json_path = os.path.join(app.config['JSON_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(json_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        os.remove(json_path)
        
        return jsonify({'success': True, 'message': 'File deleted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/list-files')
def list_files():
    """Get list of saved JSON files."""
    return jsonify({'success': True, 'files': get_saved_json_files()})


@app.route('/upload-images', methods=['POST'])
@app.route('/upload-images', methods=['POST'])
def upload_images():
    """Handle multiple image uploads for PPE detection and compliance check."""
    
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    
    # Get reference file and area name from the form
    # This allows us to know which regulation to compare against
    reference_json = request.form.get('reference_json')
    selected_area = request.form.get('area_name') # User must specify "Laboratorio", "Almacen", etc.
    
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    try:
        # Create unique batch ID and secure absolute paths
        import time
        batch_id = str(int(time.time()))
        batch_dir = os.path.abspath(os.path.join(app.config['IMAGE_FOLDER'], batch_id))
        os.makedirs(batch_dir, exist_ok=True)
        
        # Load required PPE from the reference JSON if provided
        required_ppe_names = []
        if reference_json and selected_area:
            json_path = os.path.join(app.config['JSON_FOLDER'], secure_filename(reference_json))
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    reg_data = json.load(f)
                    
                    # Search within the 'areas' list
                    areas_list = reg_data.get('areas', [])
                    for area in areas_list:
                        if area.get('nombre_area') == selected_area:
                            # Extract names of PPEs where uso_obligatorio is True
                            epis = area.get('requisitos_epi', [])
                            required_ppe_names = [epi['nombre_epi'] for epi in epis if epi.get('uso_obligatorio')]
                            break
        
        # Mapping Spanish PPE names to YOLO Class Names
        # YOLO detects: 'Helmet', 'Gloves', 'Safety Boot', 'Safety Vest'
        # PDF has: 'casco de seguridad', 'guantes de proteccion', 'botas', 'chaleco reflectante'
        synonyms = {
            "casco de seguridad": "Helmet",
            "guantes de proteccion": "Gloves",
            "botas": "Safety Boot",
            "chaleco reflectante": "Safety Vest"
        }
        
        # Translate the requirements so run_ppe_detection can match them
        translated_requirements = []
        for name in required_ppe_names:
            translated_requirements.append(synonyms.get(name.lower().strip(), name))

        # Save images (Only once)
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename, 'image'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(batch_dir, filename)
                file.save(filepath)
                saved_files.append(filename)
        
        if not saved_files:
            shutil.rmtree(batch_dir)
            return jsonify({'success': False, 'error': 'No valid image files'}), 400
        
        # Load YOLO model
        try:
            model = get_yolo_model()
        except RuntimeError as e:
            shutil.rmtree(batch_dir)
            return jsonify({'success': False, 'error': str(e)}), 500
        
        # Run PPE detection
        # This ensures the CSV column 'access_granted' is populated correctly
        output_csv = os.path.abspath(os.path.join(app.config['DETECTION_FOLDER'], f'detection_{batch_id}.csv'))
        csv_path, total_detections = run_ppe_detection(
            model=model,
            image_path=batch_dir,
            config=cv_config,
            output_csv=output_csv,
            save_images=True,
            required_ppe=translated_requirements
        )
        
        # PERFORM COMPLIANCE MATCHING (Re-integrated for the Frontend Summary)
        image_compliance = {}
        processed_images = set()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['image_name']
                processed_images.add(img_name)
                # We store the status from the CSV row directly
                if img_name not in image_compliance:
                    image_compliance[img_name] = {
                        'status': 'ALLOWED' if "AUTHORIZED" in row['access_granted'] else 'DENIED',
                        'detail': row['access_granted']
                    }

        # Final Summary construction to ensure frontend displays correct counts
        summary = get_detection_summary(csv_path)
        summary['compliance_check'] = image_compliance
        summary['images_processed'] = len(processed_images)
        summary['total_detections'] = total_detections
        
        shutil.rmtree(batch_dir)
        
        return jsonify({
            'success': True,
            'message': f'Area: {selected_area} processed successfully',
            'batch_id': batch_id,
            'summary': summary,
            'csv_filename': os.path.basename(csv_path),
            'translated_requirements': translated_requirements
        })
        
    except Exception as e:
        # In case of error, try to clean up
        if 'batch_dir' in locals() and os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get-detection-results/<batch_id>')
def get_detection_results(batch_id):
    """Retrieve detection results for a specific batch."""
    try:
        csv_filename = f'detection_{batch_id}.csv'
        # Use absolute path for lookups
        csv_path = os.path.normpath(os.path.join(app.config['DETECTION_FOLDER'], csv_filename))
        
        print(f"DEBUG: Looking for CSV at: {csv_path}") # This will show in your terminal
        
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': f'File not found at {csv_path}'}), 404
        
        # Read CSV and return as JSON
        detections = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                detections.append(row)
        
        summary = get_detection_summary(csv_path)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download-detection-csv/<batch_id>')
def download_detection_csv(batch_id):
    """Download detection results CSV."""
    try:
        csv_filename = f'detection_{batch_id}.csv'
        csv_path = os.path.join(app.config['DETECTION_FOLDER'], csv_filename)
        
        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        return send_file(csv_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get-annotated-image/<batch_id>/<filename>')
def get_annotated_image(batch_id, filename):
    """Retrieve annotated image from detection results."""
    try:
        image_dir = os.path.join(app.config['DETECTION_FOLDER'], cv_config.RUN_NAME)
        image_path = os.path.join(image_dir, secure_filename(filename))
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get-areas/<filename>')
def get_areas(filename):
    """Extract area names from a specific JSON file for the dropdown."""
    try:
        json_path = os.path.join(app.config['JSON_FOLDER'], secure_filename(filename))
        if not os.path.exists(json_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extraemos los nombres de las áreas según tu formato JSON
            areas = [area.get('nombre_area') for area in data.get('areas', [])]
            
        return jsonify({'success': True, 'areas': areas})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500    


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SAFETY REGULATION EXTRACTION DASHBOARD")
    print("="*60)
    print("\n🌐 Starting web server...")
    print("📍 Access the dashboard at: http://localhost:5000")
    print("\n⚠️  Press CTRL+C to stop the server\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)