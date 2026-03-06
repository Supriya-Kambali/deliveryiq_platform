"""
IBM DeliveryIQ — Module 4: QLoRA Fine-Tuning
=============================================
WHY QLORA FINE-TUNING?
    We want IBM DeliveryIQ to speak IBM's language — using IBM's exact
    report formats, terminology, and communication style. Fine-tuning
    achieves this by training the model on IBM-specific examples.

    QLoRA = Quantized Low-Rank Adaptation:

    Q (Quantization):
    - Compresses model weights from 32-bit → 4-bit numbers
    - Reduces memory from ~16GB → ~4GB for a 7B model
    - Makes fine-tuning possible on your Mac M4 Pro!
    - Slight quality loss (~5%) but massive memory savings

    LoRA (Low-Rank Adaptation):
    - Instead of updating ALL 7 billion parameters (expensive!)
    - We add small "adapter" matrices to specific layers
    - Only train the adapters (~0.1% of parameters)
    - The original model stays frozen — we just add IBM knowledge on top
    - After training: merge adapters back into model

    PEFT (Parameter-Efficient Fine-Tuning):
    - The HuggingFace library that implements LoRA/QLoRA
    - Manages adapter creation, training, and merging
    - Works with any HuggingFace model

    This is Week 4 in action:
    - HuggingFace Transformers: AutoTokenizer, AutoModelForCausalLM
    - PEFT: LoraConfig, get_peft_model, TaskType
    - BitsAndBytes: BitsAndBytesConfig for 4-bit quantization
    - Trainer: SFTTrainer for supervised fine-tuning
    - Mac M4 Pro: Uses Apple MPS (Metal Performance Shaders) as GPU

HARDWARE REQUIREMENTS:
    - Mac M4 Pro: ✅ Perfect (Apple Silicon MPS)
    - RAM: 16GB minimum (M4 Pro has 24-48GB) ✅
    - Storage: ~10GB for model + dataset
    - Training time: ~30-60 minutes for small dataset

MODEL CHOICE:
    We fine-tune on the model already in Day1-2/outputs/ (GPT-2 based)
    OR use Mistral-7B / Llama-3-8B for production quality.
    For demo purposes, we use the existing checkpoint.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# WHY THESE HYPERPARAMETERS?
# These are carefully chosen for Mac M4 Pro fine-tuning:
# - Small batch size (1-2): Mac has limited GPU memory vs NVIDIA
# - Gradient accumulation (8): Simulates larger batch size
# - Low learning rate (2e-4): Prevents catastrophic forgetting
# - Few epochs (3): Enough to learn IBM style without overfitting
# ─────────────────────────────────────────────────────────────────
FINETUNE_CONFIG = {
    # Model settings
    "base_model": "microsoft/phi-2",      # Small but capable, runs on Mac
    "output_dir": "./ibm_deliveryiq_model",

    # QLoRA settings
    "load_in_4bit": True,                  # 4-bit quantization
    "bnb_4bit_compute_dtype": "float16",   # Compute in float16 for speed
    "bnb_4bit_quant_type": "nf4",          # NF4 quantization (best quality)
    "use_nested_quant": False,

    # LoRA settings
    "lora_r": 16,           # Rank of LoRA matrices (higher = more capacity)
    "lora_alpha": 32,       # LoRA scaling factor (usually 2x rank)
    "lora_dropout": 0.05,   # Dropout for regularization
    "target_modules": ["q_proj", "v_proj"],  # Which layers to add LoRA to

    # Training settings
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,      # Small for Mac memory
    "gradient_accumulation_steps": 8,      # Effective batch size = 8
    "learning_rate": 2e-4,
    "max_seq_length": 512,                 # Max token length per example
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "save_steps": 50,
    "logging_steps": 10,
    "fp16": False,                         # Mac uses MPS, not CUDA fp16
    "bf16": False,
}


def check_environment() -> dict:
    """
    Check if the fine-tuning environment is properly set up.

    WHY THIS CHECK?
    Fine-tuning requires specific libraries and hardware.
    Better to fail fast with a clear message than crash mid-training.
    """
    env_status = {
        "python_version": None,
        "torch_available": False,
        "mps_available": False,  # Apple Silicon GPU
        "cuda_available": False,
        "transformers_available": False,
        "peft_available": False,
        "bitsandbytes_available": False,
        "recommended_device": "cpu"
    }

    import sys
    env_status["python_version"] = sys.version

    # Check PyTorch
    env_status["torch_available"] = True
    env_status["torch_version"] = torch.__version__

    # Check Apple Silicon MPS (Mac M4 Pro)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        env_status["mps_available"] = True
        env_status["recommended_device"] = "mps"
        print("✅ Apple Silicon MPS detected — Mac M4 Pro GPU will be used!")

    # Check CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        env_status["cuda_available"] = True
        env_status["recommended_device"] = "cuda"
        env_status["gpu_name"] = torch.cuda.get_device_name(0)

    # Check HuggingFace libraries
    try:
        import transformers
        env_status["transformers_available"] = True
        env_status["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        env_status["peft_available"] = True
        env_status["peft_version"] = peft.__version__
    except ImportError:
        pass

    try:
        import bitsandbytes
        env_status["bitsandbytes_available"] = True
    except ImportError:
        pass

    return env_status


def setup_qlora_model(model_name: str, config: dict):
    """
    Load and configure the model for QLoRA fine-tuning.

    STEP BY STEP:
    1. Load tokenizer — converts text to token IDs
    2. Configure 4-bit quantization — compress model to 4-bit
    3. Load model with quantization — loads compressed model
    4. Configure LoRA — add trainable adapter layers
    5. Return model ready for training

    WHY THESE SPECIFIC SETTINGS?
    - nf4 quantization: Best quality/compression tradeoff
    - q_proj + v_proj: The attention layers that benefit most from LoRA
    - r=16: Good balance of capacity vs memory for IBM delivery tasks
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        print(f"📥 Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        # Add padding token if missing (required for batch training)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("⚙️  Configuring 4-bit quantization (QLoRA)...")
        # WHY BitsAndBytesConfig?
        # This tells HuggingFace to load the model in 4-bit precision.
        # Without this, a 7B model needs ~14GB RAM. With 4-bit: ~4GB.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["load_in_4bit"],
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=config["use_nested_quant"]
        )

        print(f"📥 Loading model: {model_name} (4-bit quantized)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare model for k-bit training
        # WHY? Enables gradient checkpointing and casts layer norms to float32
        model = prepare_model_for_kbit_training(model)

        print("🔧 Adding LoRA adapters...")
        # WHY LORACONFIG?
        # This defines WHERE and HOW to add the trainable adapters.
        # r=16: Each adapter is a 16-rank matrix (small but effective)
        # target_modules: Only add adapters to attention layers (most impactful)
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM  # We're fine-tuning a language model
        )

        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ LoRA adapters added:")
        print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   Total parameters: {total_params:,}")
        print(f"   WHY SO FEW? LoRA only trains the adapter matrices, not the full model!")

        return model, tokenizer

    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("Install with: pip install transformers peft bitsandbytes accelerate")
        return None, None


def load_training_data(data_dir: str) -> tuple:
    """
    Load the prepared IBM delivery dataset for training.

    WHY JSONL FORMAT?
    JSONL (JSON Lines) is the standard format for HuggingFace datasets.
    Each line is one training example — easy to stream for large datasets.
    """
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "validation.jsonl")

    train_data = []
    val_data = []

    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        print(f"✅ Loaded {len(train_data)} training examples")
    else:
        print(f"⚠️  Training data not found at {train_path}")
        print("   Run prepare_dataset.py first!")

    if os.path.exists(val_path):
        with open(val_path, 'r') as f:
            for line in f:
                val_data.append(json.loads(line.strip()))
        print(f"✅ Loaded {len(val_data)} validation examples")

    return train_data, val_data


def run_finetuning(config: dict = None) -> dict:
    """
    Execute the full QLoRA fine-tuning pipeline.

    This is the main training function that:
    1. Checks environment
    2. Loads and prepares data
    3. Sets up QLoRA model
    4. Runs training
    5. Saves the fine-tuned model
    """
    if config is None:
        config = FINETUNE_CONFIG

    print("=" * 60)
    print("IBM DeliveryIQ — QLoRA Fine-Tuning")
    print("=" * 60)

    # Step 1: Check environment
    print("\n🔍 Step 1: Checking environment...")
    env = check_environment()
    print(f"   Device: {env['recommended_device'].upper()}")
    print(f"   PyTorch: {env.get('torch_version', 'N/A')}")
    print(f"   Transformers: {'✅' if env['transformers_available'] else '❌ Install: pip install transformers'}")
    print(f"   PEFT: {'✅' if env['peft_available'] else '❌ Install: pip install peft'}")
    print(f"   BitsAndBytes: {'✅' if env['bitsandbytes_available'] else '⚠️  Optional on Mac'}")

    if not env['transformers_available'] or not env['peft_available']:
        return {"success": False, "error": "Missing required libraries. Run: pip install -r requirements.txt"}

    # Step 2: Load training data
    print("\n📊 Step 2: Loading IBM delivery training data...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'kaggle_data')
    train_data, val_data = load_training_data(data_dir)

    if not train_data:
        print("   Generating dataset first...")
        from prepare_dataset import IBM_DELIVERY_QA, create_alpaca_format, augment_dataset, save_dataset
        alpaca_data = create_alpaca_format(IBM_DELIVERY_QA)
        augmented = augment_dataset(alpaca_data, multiplier=3)
        save_dataset(augmented, data_dir)
        train_data, val_data = load_training_data(data_dir)

    # Step 3: Setup QLoRA model
    print(f"\n🤖 Step 3: Setting up QLoRA model ({config['base_model']})...")
    model, tokenizer = setup_qlora_model(config["base_model"], config)

    if model is None:
        return {"success": False, "error": "Model setup failed"}

    # Step 4: Configure training
    print("\n⚙️  Step 4: Configuring training...")
    try:
        from transformers import TrainingArguments
        from trl import SFTTrainer  # Supervised Fine-Tuning Trainer

        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type=config["lr_scheduler_type"],
            save_steps=config["save_steps"],
            logging_steps=config["logging_steps"],
            fp16=config["fp16"],
            bf16=config["bf16"],
            report_to="none",  # Disable wandb for local training
            push_to_hub=False,
        )

        # Step 5: Create SFTTrainer
        # WHY SFTTrainer?
        # SFTTrainer (Supervised Fine-Tuning Trainer) from TRL library
        # handles the Alpaca format automatically — no custom data collator needed
        from datasets import Dataset

        train_dataset = Dataset.from_list([
            {"text": item["text"]} for item in train_data
        ])

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=config["max_seq_length"],
            args=training_args,
        )

        # Step 6: Train!
        print(f"\n🚀 Step 5: Starting QLoRA fine-tuning...")
        print(f"   Model: {config['base_model']}")
        print(f"   Training examples: {len(train_data)}")
        print(f"   Epochs: {config['num_train_epochs']}")
        print(f"   Estimated time: 30-60 minutes on Mac M4 Pro")
        print(f"   Device: {env['recommended_device'].upper()}")
        print("\n   Training in progress... (check logs below)")

        trainer.train()

        # Step 7: Save the fine-tuned model
        print(f"\n💾 Step 6: Saving fine-tuned model...")
        output_path = config["output_dir"]
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)

        print(f"✅ Fine-tuning complete!")
        print(f"   Model saved to: {output_path}")
        print(f"   Use inference.py to test the model")

        return {
            "success": True,
            "model_path": output_path,
            "training_examples": len(train_data),
            "epochs": config["num_train_epochs"]
        }

    except ImportError as e:
        return {"success": False, "error": f"Missing library: {e}. Run: pip install trl datasets"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# MAIN: Run fine-tuning
# python qlora_finetune.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("IBM DeliveryIQ — QLoRA Fine-Tuning Script")
    print("=" * 60)
    print("\n⚠️  BEFORE RUNNING:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Prepare dataset: python prepare_dataset.py")
    print("3. Ensure 10GB+ free disk space")
    print("4. Close other apps to free RAM")
    print("\nStarting fine-tuning in 5 seconds...")

    import time
    time.sleep(5)

    result = run_finetuning()

    if result["success"]:
        print(f"\n🎉 SUCCESS! Model fine-tuned and saved.")
        print(f"   Next: python inference.py to test your IBM-tuned model")
    else:
        print(f"\n❌ Fine-tuning failed: {result['error']}")
        print("\nTROUBLESHOOTING:")
        print("• Install missing libraries: pip install transformers peft trl bitsandbytes datasets")
        print("• For Mac: bitsandbytes may not support 4-bit on MPS — use cpu offloading")
        print("• Alternative: Use the existing model in Day1-2/outputs/final_model/")


