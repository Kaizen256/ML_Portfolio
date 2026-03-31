import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# CONFIG
# ============================================================

@dataclass
class QuizConfig:
    n_questions: int = 25
    multiple_choice: bool = True
    n_choices: int = 7
    seed: int | None = None
    allow_skip: bool = True

# ============================================================
# CATEGORIES (FILTERED)
# ============================================================

CATEGORIES: List[str] = [
    "Computer Vision / Backbones",
    "Computer Vision / Object Detection",
    "Computer Vision / Segmentation",
    "NLP / Text",
    "Generative Models / Image Generation",
    "Multimodal (Vision + Language)",
    "Reinforcement Learning",
]

# ============================================================
# MODEL -> CATEGORY
# (FILTERED to your requested fields)
# ============================================================

MODEL_TO_CATEGORY: Dict[str, str] = {
    # ----------------------------
    # Computer Vision / Backbones
    # ----------------------------
    "ResNet": "Computer Vision / Backbones",
    "EfficientNet": "Computer Vision / Backbones",
    "ConvNeXt": "Computer Vision / Backbones",
    "ViT": "Computer Vision / Backbones",
    "Swin Transformer": "Computer Vision / Backbones",

    # ----------------------------
    # Computer Vision / Detection
    # ----------------------------
    "Faster R-CNN": "Computer Vision / Object Detection",
    "DETR": "Computer Vision / Object Detection",
    "RT-DETR": "Computer Vision / Object Detection",
    "YOLOv10": "Computer Vision / Object Detection",
    "Mask R-CNN": "Computer Vision / Object Detection",

    # ----------------------------
    # Computer Vision / Segmentation
    # ----------------------------
    "U-Net": "Computer Vision / Segmentation",
    "DeepLabv3+": "Computer Vision / Segmentation",
    "Mask2Former": "Computer Vision / Segmentation",
    "SAM 2": "Computer Vision / Segmentation",

    # ----------------------------
    # NLP / Text
    # ----------------------------
    "BERT-family (Encoder-only)": "NLP / Text",
    "T5-family (Encoder-decoder)": "NLP / Text",
    "GPT-style (Decoder-only)": "NLP / Text",
    "RAG": "NLP / Text",
    "Mamba": "NLP / Text",
    "LoRA": "NLP / Text",
    "QLoRA": "NLP / Text",
    "Llama 3.1": "NLP / Text",
    "Mistral Large 2": "NLP / Text",
    "Qwen2.5": "NLP / Text",

    # ----------------------------
    # Generative / Image
    # ----------------------------
    "DDPM": "Generative Models / Image Generation",
    "Latent Diffusion (Stable Diffusion)": "Generative Models / Image Generation",
    "DiT": "Generative Models / Image Generation",
    "SDXL": "Generative Models / Image Generation",
    "Stable Diffusion 3": "Generative Models / Image Generation",
    "DALL-E 3": "Generative Models / Image Generation",
    "FLUX": "Generative Models / Image Generation",

    # ----------------------------
    # Multimodal
    # ----------------------------
    "CLIP": "Multimodal (Vision + Language)",
    "SigLIP": "Multimodal (Vision + Language)",
    "BLIP-2": "Multimodal (Vision + Language)",
    "Flamingo": "Multimodal (Vision + Language)",
    "LLaVA": "Multimodal (Vision + Language)",

    # ----------------------------
    # Reinforcement Learning
    # ----------------------------
    "DQN": "Reinforcement Learning",
    "PPO": "Reinforcement Learning",
    "SAC": "Reinforcement Learning",
    "DDPG": "Reinforcement Learning",
    "TD3": "Reinforcement Learning",
    "MuZero": "Reinforcement Learning",
    "Decision Transformer": "Reinforcement Learning",
}

# ============================================================
# MODEL "CARDS": definition + what makes it different
# Keep them short enough to read after each question.
# ============================================================

MODEL_CARDS: Dict[str, Dict[str, str]] = {
    # ----------------------------
    # CV Backbones
    # ----------------------------
    "ResNet": {
        "definition": "A CNN backbone built around residual (skip) connections to enable very deep networks without vanishing-gradient collapse.",
        "distinctive": "Skip connections that learn residuals; extremely stable baseline; backbone for countless detectors/segmenters.",
        "features": "Residual blocks, identity shortcuts, deep stacks; strong pretrained ecosystem."
    },
    "EfficientNet": {
        "definition": "A CNN family that scales depth/width/resolution in a principled way (compound scaling) for strong accuracy per compute.",
        "distinctive": "Excellent accuracy per FLOP; strong choice when compute or latency is constrained.",
        "features": "Compound scaling, MBConv-style blocks, squeeze-excitation in many variants."
    },
    "ConvNeXt": {
        "definition": "A modernized CNN updated with design choices inspired by transformers, achieving transformer-level performance with convs.",
        "distinctive": "CNN that competes with ViTs while staying conv-native; strong backbone in modern pipelines.",
        "features": "Large kernels, inverted bottlenecks, simplified stage design, modern training recipe."
    },
    "ViT": {
        "definition": "Vision Transformer that tokenizes images into patches and applies standard transformer blocks for vision representation learning.",
        "distinctive": "Strong with large-scale pretraining; clean transformer design; widely used in 2020s vision stacks.",
        "features": "Patch embeddings, multi-head self-attention, positional embeddings, transformer encoder blocks."
    },
    "Swin Transformer": {
        "definition": "Hierarchical vision transformer using shifted-window attention to scale efficiently and work well as a dense-prediction backbone.",
        "distinctive": "Great for detection/segmentation backbones due to hierarchical features and local attention windows.",
        "features": "Windowed self-attention, shifted windows, multiscale pyramid-like stages."
    },

    # ----------------------------
    # CV Detection
    # ----------------------------
    "Faster R-CNN": {
        "definition": "A two-stage detector: first proposes regions likely to contain objects, then classifies/refines boxes for those regions.",
        "distinctive": "Accuracy-first classic; strong on small objects/clutter; more complex and slower than one-stage detectors.",
        "features": "RPN region proposals, RoI pooling/align, anchor boxes (often), NMS postprocessing."
    },
    "DETR": {
        "definition": "A transformer-based detector that predicts a set of objects directly as set prediction, removing proposals and NMS.",
        "distinctive": "Clean end-to-end formulation; avoids duplicate detections via matching; vanilla can be slow to train and weaker on small objects.",
        "features": "Object queries, transformer encoder-decoder, Hungarian matching loss, set-based outputs."
    },
    "RT-DETR": {
        "definition": "A real-time variant of DETR designed to keep set prediction while making inference/training faster and more practical.",
        "distinctive": "Bridges transformer detection with real-time needs; competitive speed/accuracy tradeoffs vs YOLO-scale models.",
        "features": "Optimized query selection/initialization, adjustable decoder depth at inference in some variants."
    },
    "YOLOv10": {
        "definition": "A modern one-stage detector that predicts boxes and classes in a single pass over multi-scale feature maps.",
        "distinctive": "Real-time king in practice; strong accuracy per millisecond; huge deployment ecosystem.",
        "features": "One-stage dense predictions, multi-scale heads, optimized assignment/losses, streamlined postprocessing."
    },
    "Mask R-CNN": {
        "definition": "An extension of Faster R-CNN that adds an instance mask prediction head alongside box classification/regression.",
        "distinctive": "When you need per-object masks (instance segmentation), not just boxes; heavier than box-only detection.",
        "features": "Two-stage detection + parallel mask head, RoIAlign, typically FPN backbone."
    },

    # ----------------------------
    # CV Segmentation
    # ----------------------------
    "U-Net": {
        "definition": "An encoder-decoder segmentation architecture with skip connections to preserve spatial detail while capturing context.",
        "distinctive": "Small-data champion; excellent for medical/scientific imaging; sharp boundaries with limited labels.",
        "features": "Downsample/upsample path, skip connections, symmetric ‘U’ shape, many variants (2D/3D/Attention/nnU-Net)."
    },
    "DeepLabv3+": {
        "definition": "A semantic segmentation model using atrous (dilated) convolutions and multi-scale context aggregation with a decoder for refinement.",
        "distinctive": "Dependable semantic baseline with strong boundary quality; great when you want robust semantic segmentation.",
        "features": "Atrous conv, ASPP (Atrous Spatial Pyramid Pooling), decoder refinement head."
    },
    "Mask2Former": {
        "definition": "A transformer-based universal segmentation model for semantic, instance, and panoptic segmentation using mask-aware attention.",
        "distinctive": "Modern do-everything segmentation; strong panoptic performance; heavier than classic CNN baselines.",
        "features": "Query-based mask prediction, masked attention, set prediction style optimized for masks."
    },
    "SAM 2": {
        "definition": "A promptable segmentation foundation model that produces masks from prompts (points/boxes/masks), extended to video with memory.",
        "distinctive": "Best for interactive/zero-shot segmentation and video mask propagation; not always the best fixed-class semantic model.",
        "features": "Prompt-to-mask interface, foundation model behavior, streaming memory for video consistency."
    },

    # ----------------------------
    # NLP / Text
    # ----------------------------
    "BERT-family (Encoder-only)": {
        "definition": "Transformer encoder models that produce contextual embeddings using bidirectional attention for understanding-focused tasks.",
        "distinctive": "Best for classification, tagging, extraction, embeddings, reranking; not designed for long-form generation.",
        "features": "Bidirectional self-attention, encoder stack, masked-language-model style pretraining common."
    },
    "T5-family (Encoder-decoder)": {
        "definition": "Transformer encoder-decoder models framing tasks as text-to-text: encode input, decode output autoregressively with cross-attention.",
        "distinctive": "Strong for translation/summarization/structured generation; more flexible than encoder-only for generation tasks.",
        "features": "Encoder + decoder, cross-attention, denoising/span corruption pretraining commonly used."
    },
    "GPT-style (Decoder-only)": {
        "definition": "Transformer decoder models trained for next-token prediction with causal attention, optimized for generation.",
        "distinctive": "Best for open-ended generation, instruction following, tool use; less efficient than encoder-only for pure classification.",
        "features": "Causal self-attention, autoregressive decoding, next-token pretraining + instruction tuning in practice."
    },
    "RAG": {
        "definition": "Retrieval-Augmented Generation: combines a retriever with a generator so responses can use external documents and be up-to-date.",
        "distinctive": "Use when knowledge freshness, citations, or long-tail facts matter; reduces hallucination when retrieval is good.",
        "features": "Retriever + index, retrieved context injection, generation conditioned on retrieved passages."
    },
    "Mamba": {
        "definition": "A state-space sequence model designed to scale efficiently to long contexts, offering an alternative to attention-heavy transformers.",
        "distinctive": "Strong when transformers are too expensive for long sequences; different inductive bias vs attention.",
        "features": "State-space dynamics, efficient long-context handling, hardware-friendly design."
    },
    "LoRA": {
        "definition": "Low-Rank Adaptation: fine-tunes large models by training small low-rank adapter matrices while keeping base weights frozen.",
        "distinctive": "Cheap fine-tuning with minimal extra parameters; widely used for LLM adaptation.",
        "features": "Low-rank adapters injected into linear layers; base model mostly frozen."
    },
    "QLoRA": {
        "definition": "Quantized LoRA: combines quantization of base weights with LoRA adapters to fine-tune large LLMs on limited hardware.",
        "distinctive": "Lets you fine-tune big models on smaller GPUs while keeping quality surprisingly high.",
        "features": "Weight quantization + LoRA adapters; memory-efficient training."
    },
    "Llama 3.1": {
        "definition": "A widely used open-weight LLM ecosystem used as a base for instruction-tuned and fine-tuned models.",
        "distinctive": "Strong general backbone and community tooling; common starting point for open deployments.",
        "features": "Decoder-only LLM family with many sizes and fine-tune variants in the ecosystem."
    },
    "Mistral Large 2": {
        "definition": "A high-performance LLM ecosystem (often used for strong reasoning/coding) with a production-focused model lineup.",
        "distinctive": "Popular for capability-per-cost in some deployments; ecosystem emphasis on efficiency and practicality.",
        "features": "Decoder-only LLM family; often paired with strong inference tooling and model variants."
    },
    "Qwen2.5": {
        "definition": "An LLM ecosystem with strong general and coding performance, widely used in open deployments and variants.",
        "distinctive": "Broad lineup and strong multilingual/coding performance depending on variant.",
        "features": "Decoder-only LLM family with multiple sizes; common in open-source stacks."
    },

    # ----------------------------
    # Generative Models (Image)
    # ----------------------------
    "DDPM": {
        "definition": "Denoising Diffusion Probabilistic Models: generate images by iteratively denoising from noise using a learned reverse process.",
        "distinctive": "Core diffusion formulation powering the modern wave; high quality but historically slow sampling without accelerations.",
        "features": "Forward noise schedule, reverse denoising model, iterative sampling."
    },
    "Latent Diffusion (Stable Diffusion)": {
        "definition": "Diffusion performed in a learned latent space (instead of pixel space) for much faster, more compute-efficient generation.",
        "distinctive": "Practical open ecosystem (Stable Diffusion); strong balance of quality and speed; huge tooling community.",
        "features": "VAE latent space, UNet denoiser in latent, text conditioning (often CLIP-like text encoder)."
    },
    "DiT": {
        "definition": "Diffusion Transformer: replaces the typical UNet denoiser with a transformer backbone for diffusion modeling.",
        "distinctive": "Transformer-native diffusion; strong scaling behavior; popular research direction.",
        "features": "Transformer blocks as denoiser, tokenized image/latent representations, diffusion training objective."
    },
    "SDXL": {
        "definition": "A higher-capacity latent diffusion model family designed for improved high-resolution generation and prompt adherence.",
        "distinctive": "Stronger high-res output vs earlier Stable Diffusion versions; commonly used in open workflows.",
        "features": "Larger architecture, improved conditioning, better high-res fidelity."
    },
    "Stable Diffusion 3": {
        "definition": "A newer Stable Diffusion family with updated architecture choices aimed at better prompt fidelity and quality.",
        "distinctive": "Modernized SD line; generally better prompt adherence and quality vs earlier generations (varies by release/variant).",
        "features": "Updated diffusion backbone/conditioning pipeline; newer training recipe."
    },
    "DALL-E 3": {
        "definition": "A text-to-image generation system optimized for high prompt fidelity and reliable production-grade outputs.",
        "distinctive": "Very strong prompt following; less about tinkering with open weights, more about turnkey quality.",
        "features": "Text-to-image generation with strong alignment to prompts and safety/quality systems."
    },
    "FLUX": {
        "definition": "A modern text-to-image and editing-oriented model ecosystem known for high quality and practical workflows.",
        "distinctive": "Strong modern open tooling vibe; often used for generation plus editing/control workflows.",
        "features": "Modern generation backbone + ecosystem emphasis on editing and practical usage."
    },

    # ----------------------------
    # Multimodal
    # ----------------------------
    "CLIP": {
        "definition": "Contrastive Language–Image Pretraining: learns aligned image and text embeddings for retrieval and zero-shot classification.",
        "distinctive": "Turns images+text into a shared embedding space; great for zero-shot and search/retrieval pipelines.",
        "features": "Contrastive loss, dual encoders (image + text), cosine similarity embedding space."
    },
    "SigLIP": {
        "definition": "A CLIP-style vision-language embedding model using a sigmoid-based loss variant for training.",
        "distinctive": "Often competitive alternative to CLIP for embedding quality and training dynamics.",
        "features": "Vision+text encoders, sigmoid loss formulation, shared embedding space."
    },
    "BLIP-2": {
        "definition": "A vision-language model that connects a frozen vision encoder to a large language model through a lightweight bridging module.",
        "distinctive": "Practical way to get strong captioning/VQA using powerful pretrained components without full joint training.",
        "features": "Vision encoder + Q-Former/bridge + LLM, cross-modal alignment."
    },
    "Flamingo": {
        "definition": "A few-shot vision-language model designed to condition an LLM on visual inputs for multimodal reasoning and generation.",
        "distinctive": "Notable for few-shot multimodal capabilities; conceptually influential in VLM design.",
        "features": "Visual conditioning modules + LLM backbone, few-shot prompting behavior."
    },
    "LLaVA": {
        "definition": "A popular open vision-language assistant model that pairs an image encoder with an instruction-tuned language model.",
        "distinctive": "Common open baseline for “chat with images”; strong community and many variants.",
        "features": "Image encoder + projection + LLM, instruction tuning for multimodal dialogue."
    },

    # ----------------------------
    # RL
    # ----------------------------
    "DQN": {
        "definition": "Learns a Q-score for each discrete action and picks the biggest; uses replay + target network so training doesn’t implode.",
        "distinctive": "Classic discrete-action baseline; needs stability tricks; struggles with continuous actions.",
        "features": "Replay buffer, target network, epsilon-greedy exploration, Q-learning objective."
    },
    "PPO": {
        "definition": "Learns a policy directly but limits how much it can change per update; stable and scalable but needs lots of fresh data.",
        "distinctive": "Stable, widely used workhorse; less sample efficient because it’s on-policy.",
        "features": "Clipped objective / trust-region-like constraint, rollouts, advantage estimation."
    },
    "SAC": {
        "definition": "Off-policy actor-critic for continuous actions that intentionally stays random via entropy; robust and sample efficient.",
        "distinctive": "Strong continuous-control default; sample efficient from replay; robust due to stochastic policies.",
        "features": "Replay buffer, stochastic actor, entropy temperature, typically two critics."
    },
    "DDPG": {
        "definition": "Continuous control actor-critic with a deterministic actor; off-policy with replay; can be unstable.",
        "distinctive": "Can work well but is brittle; sensitive to hyperparameters and exploration noise; TD3 improves it.",
        "features": "Deterministic actor, critic Q-network, target networks, action noise for exploration."
    },
    "TD3": {
        "definition": "DDPG but fixed: two critics, delayed actor updates, and target smoothing to stop overestimation and jitter.",
        "distinctive": "More stable than DDPG; strong continuous-control baseline; often less robust than SAC in some settings.",
        "features": "Twin Q networks, delayed actor updates, target action noise smoothing."
    },
    "MuZero": {
        "definition": "Learns its own internal model of the environment in latent space and uses search/planning to pick actions.",
        "distinctive": "Elite results in planning-heavy domains (games); more complex and compute-intensive than PPO/SAC.",
        "features": "Representation + dynamics + prediction networks, planning/search in latent space."
    },
    "Decision Transformer": {
        "definition": "Doesn’t learn Q or policy gradients; it predicts actions like language modeling conditioned on desired return.",
        "distinctive": "Great framing for offline datasets; avoids explicit value iteration; depends on dataset quality/coverage.",
        "features": "Trajectory tokens (states/actions/returns), transformer sequence model, return-conditioning."
    },
}

# ============================================================
# Quiz Engine
# ============================================================

def build_choices(correct: str, all_categories: List[str], k: int) -> List[str]:
    pool = [c for c in all_categories if c != correct]
    distractors = random.sample(pool, k=min(k - 1, len(pool)))
    choices = distractors + [correct]
    random.shuffle(choices)
    return choices

def print_model_card(model: str) -> None:
    card = MODEL_CARDS.get(model)
    if not card:
        print("Model card not found.\n")
        return
    print("---- Model Card ----")
    print(f"Definition:   {card['definition']}")
    print(f"Difference:   {card['distinctive']}")
    print(f"Key features: {card['features']}")
    print("--------------------\n")

def run_quiz(cfg: QuizConfig) -> None:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    models = list(MODEL_TO_CATEGORY.keys())
    random.shuffle(models)

    picked = [models[i % len(models)] for i in range(cfg.n_questions)]

    correct_count = 0
    total_answered = 0
    wrong: List[Tuple[str, str, str]] = []

    print("\nModel → Category Drill (with model cards)")
    print("-" * 44)
    print("Type the number of the category.")
    print("Commands: 'q' quit" + (" | 's' skip" if cfg.allow_skip else ""))
    print()

    for i, model in enumerate(picked, start=1):
        correct_cat = MODEL_TO_CATEGORY[model]
        print(f"[{i}/{cfg.n_questions}] Model: {model}")

        choices = build_choices(correct_cat, CATEGORIES, cfg.n_choices)
        for idx, cat in enumerate(choices, start=1):
            print(f"  {idx}) {cat}")

        raw = input("Your answer: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if cfg.allow_skip and raw in {"s", "skip"}:
            print(f"Skipped. Correct: {correct_cat}")
            print_model_card(model)
            continue

        total_answered += 1

        if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
            guess = "(invalid)"
            wrong.append((model, guess, correct_cat))
            print(f"Invalid. Correct: {correct_cat}")
            print_model_card(model)
            continue

        guess = choices[int(raw) - 1]
        if guess == correct_cat:
            correct_count += 1
            print("✅ Correct.")
        else:
            wrong.append((model, guess, correct_cat))
            print(f"❌ Wrong. Correct: {correct_cat}")

        print_model_card(model)

    # Summary
    print("\nResults")
    print("-" * 44)
    if total_answered == 0:
        print("You answered nothing. Deep commitment to chaos.")
        return

    acc = correct_count / total_answered
    print(f"Answered: {total_answered}")
    print(f"Correct:  {correct_count}")
    print(f"Accuracy: {acc:.1%}")

    if wrong:
        print("\nMissed:")
        for model, guess, correct in wrong:
            print(f" - {model}: guessed '{guess}' | correct '{correct}'")
    print()

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    cfg = QuizConfig(
        n_questions=30,
        multiple_choice=True,
        n_choices=6,
        seed=None,   # set to an int for repeatable quizzes
        allow_skip=True,
    )
    run_quiz(cfg)