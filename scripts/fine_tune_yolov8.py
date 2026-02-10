"""
YOLOv8 Training Script for PPE Detection

This module handles the training, evaluation, and export of a YOLOv8 model
for Personal Protective Equipment (PPE) detection. It includes dataset validation,
GPU detection, model training with fine-tuning, and model export to ONNX format.

Author: Breixo Brea
Date: 2025
"""

from ultralytics import YOLO
import os
import torch
import sys
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration class for YOLOv8 training parameters."""
    
    # Dataset paths
    BASE_DIR: str = "../datasets" # You may need to adjust this path based on your project structure
    DATA_YAML: str = os.path.join(BASE_DIR, "data.yaml")
    
    # Model configuration
    MODEL_VERSION: str = "yolov8m.pt"
    
    # Training hyperparameters
    EPOCHS: int = 500
    PATIENCE: int = 20  # Early stopping patience
    IMAGE_SIZE: int = 640
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.0005
    OPTIMIZER: str = "AdamW"
    
    # Inference configuration
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Training options
    USE_PRETRAINED: bool = True
    USE_AUGMENTATION: bool = True
    NUM_WORKERS: int = 4
    
    # Output configuration
    EXPERIMENT_NAME: str = "ppe_yolov8_finetuned"
    EXPORT_FORMAT: str = "onnx"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_dataset_structure(base_dir: str) -> bool:
    """
    Validate the dataset directory structure.
    
    Checks if all required directories exist and contain files:
    - train/images, train/labels
    - val/images, val/labels
    - test/images, test/labels
    
    Args:
        base_dir: Path to the dataset root directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    expected_dirs = [
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
        "test/images",
        "test/labels"
    ]
    
    all_valid = True
    
    for dir_path in expected_dirs:
        full_path = os.path.join(base_dir, dir_path)
        
        if not os.path.exists(full_path):
            print(f"ERROR: Directory does not exist: {full_path}")
            all_valid = False
            continue
            
        if not os.listdir(full_path):
            print(f"WARNING: Directory is empty: {full_path}")
    
    if all_valid:
        print("✓ Dataset structure validated successfully.")
    
    return all_valid


def detect_device() -> str:
    """
    Detect available computing device (GPU or CPU).
    
    Returns:
        'cuda' if GPU is available, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
        return 'cuda'
    else:
        print("⚠ No GPU detected. Training will use CPU (slower).")
        return 'cpu'


def load_model(model_path: str) -> YOLO:
    """
    Load a YOLOv8 model from file or download if not exists.
    
    Args:
        model_path: Path to the model weights file
        
    Returns:
        Loaded YOLO model instance
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        # YOLO will automatically download the model if it doesn't exist
        model = YOLO(model_path)
        print(f"✓ YOLOv8 model loaded successfully: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv8 model: {str(e)}")


def validate_config(config: TrainingConfig) -> None:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration object
        
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If configuration values are invalid
    """
    # Check if data.yaml exists
    if not os.path.exists(config.DATA_YAML):
        raise FileNotFoundError(f"Data YAML file not found: {config.DATA_YAML}")
    
    # Check if base directory exists
    if not os.path.exists(config.BASE_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {config.BASE_DIR}")
    
    # Validate hyperparameters
    if config.EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    if config.BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    if not (0 < config.LEARNING_RATE < 1):
        raise ValueError("LEARNING_RATE must be between 0 and 1")
    if not (0 < config.CONFIDENCE_THRESHOLD < 1):
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    print("✓ Configuration validated successfully.")


def train_model(model: YOLO, config: TrainingConfig, device: str) -> Any:
    """
    Train the YOLOv8 model with specified configuration.
    
    Args:
        model: YOLO model instance
        config: Training configuration
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Training results object
    """
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    results = model.train(
        data=config.DATA_YAML,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,  # Early stopping if no improvement
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        lr0=config.LEARNING_RATE,
        optimizer=config.OPTIMIZER,
        device=device,
        name=config.EXPERIMENT_NAME,
        pretrained=config.USE_PRETRAINED,
        augment=config.USE_AUGMENTATION,
        workers=config.NUM_WORKERS,
    )
    
    print("\n✓ Training completed successfully.")
    return results


def evaluate_model(model: YOLO, config: TrainingConfig) -> Any:
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained YOLO model
        config: Training configuration
        
    Returns:
        Prediction results
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)
    
    test_images_path = os.path.join(config.BASE_DIR, "test/images")
    
    if not os.path.exists(test_images_path):
        print(f"WARNING: Test images directory not found: {test_images_path}")
        return None
    
    results = model.predict(
        source=test_images_path,
        save=True,
        conf=config.CONFIDENCE_THRESHOLD
    )
    
    print(f"✓ Predictions saved to: {model.predictor.save_dir}")
    return results


def export_model(model: YOLO, export_format: str) -> None:
    """
    Export the trained model to specified format.
    
    Args:
        model: Trained YOLO model
        export_format: Export format (e.g., 'onnx', 'torchscript')
    """
    print("\n" + "="*60)
    print(f"EXPORTING MODEL TO {export_format.upper()} FORMAT")
    print("="*60)
    
    try:
        model.export(format=export_format)
        print(f"✓ Model exported successfully to {export_format.upper()} format.")
    except Exception as e:
        print(f"ERROR: Failed to export model: {str(e)}")
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution function for YOLOv8 training pipeline.
    
    Pipeline steps:
    1. Validate configuration
    2. Check dataset structure
    3. Detect computing device
    4. Load model
    5. Train model
    6. Evaluate on test set
    7. Export model
    """
    config = TrainingConfig()
    
    print("="*60)
    print("YOLOV8 TRAINING PIPELINE - PPE DETECTION")
    print("="*60)
    print(f"Dataset path: {config.BASE_DIR}")
    print(f"Data YAML: {config.DATA_YAML}")
    print(f"Model: {config.MODEL_VERSION}")
    print(f"Experiment: {config.EXPERIMENT_NAME}")
    print("="*60 + "\n")
    
    try:
        # Step 1: Validate configuration
        validate_config(config)
        
        # Step 2: Check dataset structure
        if not check_dataset_structure(config.BASE_DIR):
            print("\nERROR: Dataset structure is incomplete or invalid.")
            print("Please verify that all required directories exist and contain data.")
            sys.exit(1)
        
        # Step 3: Detect device
        device = detect_device()
        
        # Step 4: Load model
        model = load_model(config.MODEL_VERSION)
        
        # Step 5: Train model
        train_results = train_model(model, config, device)
        
        # Step 6: Evaluate model
        eval_results = evaluate_model(model, config)
        
        # Step 7: Export model
        export_model(model, config.EXPORT_FORMAT)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY ✓")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: Invalid configuration - {str(e)}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nERROR: Runtime error - {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()