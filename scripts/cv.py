"""
Computer Vision Module for PPE Detection

This module handles PPE (Personal Protective Equipment) detection in images
using a fine-tuned YOLOv8 model. It processes single images or directories
and generates detection results in CSV format.

Author: Breixo Brea
Date: 2025
"""

import os
import csv
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

class CVConfig:
    """Configuration for computer vision PPE detection."""
    
    BASE_DIR = Path(__file__).resolve().parent.parent 
    MODEL_PATH: str = str(BASE_DIR / "finetuning" / "detect" / "ppe_yolov8_finetuned" / "weights" / "best.pt")
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Output configuration
    # Ensure output directory is absolute relative to project root
    OUTPUT_DIR: str = str(BASE_DIR / "detections")
    RUN_NAME: str = "ppe_detections"
    SAVE_IMAGES: bool = True
    
    # CSV output
    CSV_FILENAME: str = "detection_results.csv"
    
    # Supported image formats
    SUPPORTED_FORMATS: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_detection_model(model_path: str) -> YOLO:
    """
    Load YOLOv8 model for PPE detection.
    
    Args:
        model_path: Path to the trained model weights
        
    Returns:
        Loaded YOLO model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = YOLO(model_path)
        print(f"✓ PPE detection model loaded successfully: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {str(e)}")


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def get_image_files(image_path: str, supported_formats: tuple) -> List[str]:
    """
    Get list of image files from a path (file or directory).
    
    Args:
        image_path: Path to image file or directory
        supported_formats: Tuple of supported file extensions
        
    Returns:
        List of image file paths
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If no valid images found
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Path not found: {image_path}")
    
    image_files = []
    
    if os.path.isdir(image_path):
        # Process directory
        for filename in os.listdir(image_path):
            file_path = os.path.join(image_path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in supported_formats:
                    image_files.append(file_path)
    else:
        # Process single file
        _, ext = os.path.splitext(image_path.lower())
        if ext not in supported_formats:
            raise ValueError(f"Unsupported image format: {ext}")
        image_files.append(image_path)
    
    if not image_files:
        raise ValueError(f"No valid image files found in: {image_path}")
    
    return image_files


def process_detection_results(results: Any) -> List[Dict[str, Any]]:
    """
    Process YOLO detection results and extract relevant information.
    
    Args:
        results: YOLO prediction results
        
    Returns:
        List of detection dictionaries with class_id, class_name, confidence, and bbox
    """
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            
            # Get bounding box coordinates (xyxy format)
            bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
            
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
    
    return detections


# ============================================================================
# DETECTION EXECUTION
# ============================================================================

def run_ppe_detection(
    model: YOLO,
    image_path: str,
    config: CVConfig,
    output_csv: Optional[str] = None,
    save_images: bool = True,
    required_ppe: Optional[List[str]] = None  # parameter for compliance check
) -> Tuple[str, int]:
    """
    Run PPE detection on images and save results.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to image file or directory
        config: CV configuration object
        output_csv: Path for CSV output (optional, uses config default if None)
        save_images: Whether to save annotated images
        required_ppe: List of required PPE items for compliance check (optional)
    Returns:
        Tuple of (csv_path, total_detections)
        
    Raises:
        FileNotFoundError: If image path doesn't exist
        ValueError: If no valid images found
    """
    # Get list of image files
    image_files = get_image_files(image_path, config.SUPPORTED_FORMATS)
    
    print(f"\n{'='*60}")
    print(f"PPE DETECTION - Processing {len(image_files)} images")
    print(f"{'='*60}\n")
    
    # Determine CSV output path
    if output_csv is None:
        output_csv = os.path.join(config.OUTPUT_DIR, config.CSV_FILENAME)
    else:
        # Ensure the output path is absolute to avoid WinError issues
        output_csv = os.path.abspath(output_csv)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # We ensure it's a set even if required_ppe is None or empty
    if required_ppe is not None:
        required_set = {item.lower().strip() for item in required_ppe if item}
    else:
        required_set = set()
    
    total_detections = 0
    
    # Open CSV file for writing results
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header
        writer.writerow([
            'image_name',
            'class_id',
            'class_name',
            'confidence',
            'bbox_x1',
            'bbox_y1',
            'bbox_x2',
            'bbox_y2',
            'access_granted'
        ])
        
        # Process each image
        for idx, img_path in enumerate(image_files, 1):
            img_name = os.path.basename(img_path)
            
            print(f"⏳ [{idx}/{len(image_files)}] Processing: {img_name}")
            
            try:
                # Run YOLO prediction
                results = model.predict(
                    source=img_path,
                    conf=config.CONFIDENCE_THRESHOLD,
                    save=save_images,
                    project=config.OUTPUT_DIR,
                    name=config.RUN_NAME,
                    exist_ok=True,
                    verbose=False
                )
                
                # Extract detection information
                detections = process_detection_results(results)
                
                # --- LOGIC FOR ACCESS GRANTED ---
                detected_names = {d['class_name'].lower().strip() for d in detections}
                
                # Check if all required items are present in detected items
                missing_items = required_set - detected_names
                is_compliant = len(missing_items) == 0
                
                # Formulate the status message
                if not required_set:
                    # If no PPE is required, we allow entry even with 0 detections
                    access_status = "AUTHORIZED (No requirements)"
                elif not detections:
                    # If PPE is required but NOTHING was detected, access is strictly denied
                    access_status = f"DENIED (No PPE detected. Missing: {', '.join(required_set)})"
                elif is_compliant:
                    access_status = "AUTHORIZED"
                else:
                    access_status = f"DENIED (Missing: {', '.join(missing_items)})"
                # --------------------------------

               # Write detections to CSV
                if not detections:
                    # even if the AI found 0 objects.
                    writer.writerow([
                        img_name, 
                        '-1',        # ID -1 to indicate no objects
                        'None',      # Class name
                        '0.0000',    # Confidence
                        '0.00', '0.00', '0.00', '0.00', # Empty BBox
                        access_status
                    ])
                else:
                    for detection in detections:
                        bbox = detection['bbox'] if detection['bbox'] else [0, 0, 0, 0]
                        writer.writerow([
                            img_name,
                            detection['class_id'],
                            detection['class_name'],
                            f"{detection['confidence']:.4f}",
                            f"{bbox[0]:.2f}",
                            f"{bbox[1]:.2f}",
                            f"{bbox[2]:.2f}",
                            f"{bbox[3]:.2f}",
                            access_status
                        ])
                        total_detections += 1
                
                print(f"  ✓ Found {len(detections)} PPE items. Status: {access_status}")
                
            except Exception as e:
                print(f"  ✗ Error processing {img_name}: {str(e)}")
                continue
    
    print(f"\n{'='*60}")
    print(f"DETECTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total PPE items detected: {total_detections}")
    print(f"Results saved to: {output_csv}")
    
    if save_images:
        output_images_dir = os.path.join(config.OUTPUT_DIR, config.RUN_NAME)
        print(f"Annotated images saved to: {output_images_dir}")
    
    print(f"{'='*60}\n")
    
    return output_csv, total_detections


def get_detection_summary(csv_path: str) -> Dict[str, Any]:
    """
    Generate summary statistics from detection CSV.
    
    Args:
        csv_path: Path to detection results CSV
        
    Returns:
        Dictionary with detection statistics
    """
    if not os.path.exists(csv_path):
        return {}
    
    class_counts = {}
    total_detections = 0
    images_processed = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            class_name = row['class_name']
            images_processed.add(row['image_name'])
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_detections += 1
    
    return {
        'total_detections': total_detections,
        'images_processed': len(images_processed),
        'class_distribution': class_counts
    }

def validate_ppe_compliance(detected_items: List[str], required_items: List[str]) -> Tuple[bool, List[str]]:
    """
    Compare detected PPE against required PPE from the regulation document.
    
    Args:
        detected_items: List of labels found by YOLO
        required_items: List of mandatory PPE from the PDF
        
    Returns:
        Tuple of (is_compliant, missing_items)
    """
    # Normalize strings for comparison (lowercase and strip)
    detected_set = {item.lower().strip() for item in detected_items}
    required_set = {item.lower().strip() for item in required_items}
    
    missing_items = list(required_set - detected_set)
    is_compliant = len(missing_items) == 0
    
    return is_compliant, missing_items


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main entry point for standalone execution."""
    config = CVConfig()
    
    print("\n" + "="*60)
    print("PPE DETECTION SYSTEM")
    print("="*60 + "\n")
    
    try:
        # Load model
        print("Loading model...")
        model = load_detection_model(config.MODEL_PATH)
        
        # Example: Process a directory of images
        # Modify this path to your image directory
        image_path = "test_images"
        
        if not os.path.exists(image_path):
            print(f"\nERROR: Image path not found: {image_path}")
            print("Please create a 'test_images' directory with images to process.")
            return
        
        # Run detection
        csv_path, total_detections = run_ppe_detection(
            model=model,
            image_path=image_path,
            config=config
        )
        
        # Display summary
        summary = get_detection_summary(csv_path)
        
        if summary:
            print("\n" + "="*60)
            print("DETECTION SUMMARY")
            print("="*60)
            print(f"Images processed: {summary['images_processed']}")
            print(f"Total detections: {summary['total_detections']}")
            print("\nClass distribution:")
            for class_name, count in summary['class_distribution'].items():
                print(f"  - {class_name}: {count}")
            print("="*60 + "\n")
        
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\nERROR: {str(e)}")
    except Exception as e:
        print(f"\nERROR: Unexpected error - {str(e)}")


if __name__ == "__main__":
    main()