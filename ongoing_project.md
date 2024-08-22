# Handover Document for Multi-Modal Continual Learning Research

Date: 2024.08.23

Prepared by: Edward Tang

Handover to: CRUISE Team

Project Title: LLMs within Continual Learning for Open-vocabulary Challenges

## Project Overview

### Project Objectives

LLMs training requires the vast amount of data, but LLMs still face catastrophic forgetting challenge with hallucinations, particularly in realistic scenarios such as recent unseen data, open-vocabulary queries, and long-term videos. Fortunately, continual learning presents a more efficient solution to these issues. However, most prior work in this field has focused on basic classification tasks, whereas real-world application data is difficult to collect. Our goal is to maintain the generalization capabilities of existing LLMs while enhancing their ability to continually learn and integrate new knowledge.

## Technical Details

### Data and Datasets

The data is stored in ```/mnt/data728/tianqi/openeqa/data```

### Experimental Setup

We prefer few-shot continual learning setting for limited samples or new tasks.

#### Task definition:

Few-shot continual learning aims to develop models that can continuously learn from a sequence of tasks or experiences, where each task provides only a few labelled examples (few-shot learning).

**Upstream tasks:**
Tasks used to fine-tune a model. Here we plan to leverage VQA tasks from dataset activitynet, msr-vtt, ego4d

**Downstream tasks:**
Given few-shot samples, we evaluate models on openeqa

**Algorithms:**

Fine-tuning; LoRA; Instruction tuning; Retrieval-augmented Generation; Adapter


### Model Architecture and Implementation

The framework consists of four main components:

1. Visual Encoder
2. Text Encoder
3. Long-Term Memory Bank
4. Multi-Modal LLM (e.g., LLaMA, LLaVA, VideoLLaVA)

We incorporate large language models (LLMs) to benefit from their strong generalization capabilities in open-vocabulary scenarios. However, due to the high computational cost of training/fine-tuning LLMs and GPU memory constraints, additional encoders with fewer parameters are recommended. We employ ViT/Transformer architectures as visual and text encoders. For continual learning, particularly in long-term video understanding, three dedicated memory banks are implemented to store raw visual features, learned queries, and responses with high confidence at each timestep. The LLM generates textual outputs to support various downstream video understanding tasks.

### Training and Evaluation Protocols

[TBD] Training script

For evaluation,
run ```python openeqa/baselines/llava-video.py --num-frames 50```. Decrease the number of frames if GPU memory is limited.

Then,
run ```python evaluate-predictions.py <path/to/results/file.json>```

## Research Progress and Results

See in shared slides: https://docs.google.com/presentation/d/1227x8zam2n7sMsnYoYer5Sck5q1QBD1R/edit?usp=drive_link&ouid=114050544859960499104&rtpof=true&sd=true

### Completed Tasks

1. Evaluation protocols for different multi-modal LLMs, especially video-llava
2. Model archiecture design

### Ongoing Work

1. Fine-tuning script with different algorithms (LoRA; Instruction tuning; retrieval augmented generation; adapter tuning)

## References and Appendices

[1] Touvron, Hugo, et al. "Llama: Open and efficient foundation language models." arXiv preprint arXiv:2302.13971 (2023).
[2] Majumdar, Arjun, et al. "Openeqa: Embodied question answering in the era of foundation models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[3] Islam, Md Mohaiminul, et al. "Video ReCap: Recursive Captioning of Hour-Long Videos." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[4] Lin, Bin, et al. "Video-llava: Learning united visual representation by alignment before projection." arXiv preprint arXiv:2311.10122 (2023).
[5] Liu, Haotian, et al. "Visual instruction tuning." Advances in neural information processing systems 36 (2024).
[6] Wu, Tongtong, et al. "Continual learning for large language models: A survey." arXiv preprint arXiv:2402.01364 (2024).
[7] Continual learning of large language models: A comprehensive survey
[8] He, Bo, et al. "Ma-lmm: Memory-augmented large multimodal model for long-term video understanding." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[9] Weng, Yuetian, et al. "Longvlm: Efficient long video understanding via large language models." arXiv preprint arXiv:2404.03384 (2024).