# Handover Document for Multi-Modal Continual Learning Research

Date: 2024.08.23

Prepared by: Edward Tang

Handover to: CRUISE Team

Project Title: ViLCo-Bench: VIdeo Language COntinual learning Benchmark

## Project Overview

This project presents ViLCo, the first video-language continual learning benchmark designed to evaluate continual learning models across diverse video-text tasks. The benchmark primarily utilizes data curated from the largest egocentric video dataset, Ego4D, and introduces a novel continual learning scenario called query-incremental learning. We also assess the performance of state-of-the-art continual learning methods within our benchmark. Additionally, we propose a new framework tailored specifically for multi-modal continual learning tasks.

More details coule be found in:

[Paper](https://arxiv.org/abs/2406.13123)

[Code](https://github.com/cruiseresearchgroup/ViLCo?tab=readme-ov-file)

### Project Objectives

1. We propose the first standardized benchmark in multimodal continual learning for video data, defining protocols for training and metrics for evaluation. This standardized framework allows researchers to effectively compare models, driving advancements in AI systems that can continuously learn from diverse data sources.

2. We define the setup for three recent multimodal tasks in a continual learning setup: Moment Query (MQ), Natural Language Query (NLQ), and Visual Query (VQ). We provide systematic insights into the challenges, gaps, and limitations of each video-text CL tasks.

3. We provide a comparison against state-of-the-art (SOTA) models in video and text continual learning for each benchmark setup. We prepared a curated dataset suitable for multimodal continual learning tasks using the well-known Ego4D dataset.

### Background

Recently, different benchmarks have been introduced for continual learning purposes in different tasks and modalities. For example, CoRE50, Stream-51, and CLeAR are among the most well-known benchmarks for continuous object recognition and streaming classification from still images. 
vCLIMB is the first that provides a standard benchmark for video continual learning. The vCLIMB benchmark also proposes a novel approach that resolves the complexity of storing videos in memory by employing a temporal consistency loss, which reduces the number of frames stored. However, it focuses on class-incremental classification tasks and does not consider multimodal tasks. 
This motivates us to establish a new multimodal continual learning benchmark. The multimodal CL domain remains largely underexplored, and the crucial first step is to create a benchmark to effectively evaluate CL models.
(You could also refer to Backgrounds and Related Works section in our paper.)

### Research Milestones

**Timeline**

#### January, 2024
- **Literature Review and Problem Identification**  
  Conduct an extensive review of relevant literature and identify key gaps in existing CL research.
- **Proposal Submission**  
  Finalize and submit the research proposal for approval.

#### February-March, 2024
- **Data Collection**  
  Collect primary data from selected sources (Ego4d dataset) and organize it for analysis.
- **Initial Experimentation**  
  Begin initial experiments to test research hypotheses. We first tried to follow previous setting, such as class-incremental learning.

#### April-July, 2024
- **Data Analysis and Model Development**  
  Analyze collected data and refine models based on findings. We propose new query-incremental setting and evaluate SOTA CL methods. Moreover, we design a novel architecture for our benchmark within multi-modal data.
- **Paper Prepartion**  
  Write a paper outlining progress and preliminary results. Ready to sumbit to Neurips dataset and benchmark track.

#### June-August, 2024
- **Supplementary material and rebuttal submission**  
  Begin supplement the final research paper, including supplemenary material and rebuttal discussion.
- **Document and Code cleanup**  
  Prepare full documents and codes for handover.

## Technical Details

In this document, we mainly focus on the usage of code and detailed implementation. More details (such as data curation and metrics), please refer to our paper.

### Data and Datasets

We provide one download link from zonodo [Link](https://zenodo.org/records/11560095). You should download all files in your folder, then run ```unzip ViLCo_data.zip```.

The data structure is shown as below:
  ```bash
  ViLCo_data
  ├── feature_extraction
  │   └── Feature extraction script for custom dataset, download scripts for additional visual features
  ├── features
  │   └── pre-extratced visual features.
  ├── MQ_meta_data
  │   └── metadata for moments query.
  ├── NLQ_meta_data
  │   └── metadata for natural language query.
  ├── VQ_meta_data
  │   └── metadata for visual query.
  └── narration_data
      └── pre-extracted narration features for self-supervised training.
 ```

 We also leverage pre-extracted features for different backbones provided by other repos, you could download them and follow the instructions on Experimental Setup. The Links are shown as below:

 1. EgoVLP-v2 (https://github.com/facebookresearch/EgoVLPv2)
 -	Pre-extracted EgoMQ Features

	The EgoMQ metadata can be downloaded from the Ego4D official webpage (https://ego4d-data.org/). Follow the annotation conversion step here (https://github.com/EGO4D/episodic-memory/tree/main/MQ#annotation-conversion). Keep the metadata in jsons/ folder. For quickstart, the matadata can be easily downloaded as follows:
	```
	wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoMQ/jsons.tgz
	tar -xvzf jsons.tgz && rm jsons.tgz
	```

	The pre-extracted video features can be downloaded as:
	```
	wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/pre-extracted_features/EgoMQ/EgoVLPv2.tgz
	tar -xvzf EgoVLPv2.tgz && rm EgoVLPv2.tgz
	```
- Pre-extracted EgoNLQ Features

	The EgoNLQ metadata can be downloaded from the Ego4D official webpage. Follow the data preparation steps here (https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet#preparation). Keep the metadata in jsons/ folder. For quickstart, the matadata can be easily downloaded as follows:

	```
	wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/pre-extracted_features/EgoNLQ/EgoVLPv2.tgz
	tar -xvzf EgoVLPv2.tgz && rm EgoVLPv2.tgz
	```
2. EgoVLP (https://github.com/showlab/EgoVLP)
- Pre-extracted EgoMQ Features

	1. Make sure you have prepared the MQ metadata.
	2. Download the EgoVLP clip-level features for MQ. 
		- [MQ Train & Val]( https://drive.google.com/file/d/1-HEUCdyfNX7CBZhz40yiyTr7to_p7wUi/view ); 
		- [MQ Test]( https://drive.google.com/file/d/1-JmezY3eIkHKJ1JBA_AA8QWBoY3W2HpS/view)

- Pre-extracted EgoNLQ Features

	1. Make sure you have prepared the NLQ metadata. 
	2. For the video branch, download the EgoVLP clip-level features for NLQ.
		- [NLQ Train & Val](https://drive.google.com/file/d/1TXBlLDqDuL_XPCuXlgiikEfVXO8Ly6eM/view)
		- [NLQ Test](https://drive.google.com/file/d/1-CGZg9t-kpW5bmg9M62VHk5eYTllsPKV/view)

3. GroundNLQ (https://github.com/houzhijian/GroundNLQ/blob/main/feature_extraction/README.md)

	They provide the EgoVLP and InternVideo features.
	```
	git clone https://github.com/houzhijian/GroundNLQ.git
	cd GroundNLQ/feature_extraction
	python download_features.py --feature_types [Backbone Name]
	python convert_pt_to_lmdb.py
	(ps: changing path name)
	```

4. Ego4d (https://ego4d-data.org/)

	Follow the instructions here (https://ego4d-data.org/docs/start-here/) and (https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md).
	
	For full-scale data:
	```
	ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations
	```

	For visual features (changing datasets name to optional backbones, SlowFast as an example):
	```
	ego4d --output_directory="~/ego4d_data" --datasets slowfast8x8_r101_k400 annotations
	```

### Experimental Setup

### Data preparation
If you have prepared the above features/data. Please follow this instruction step by step:

**For MQ,**
1. You should put visual features into ``feat_folder``, textual features into ``text_feat_folder``. The dictionary name could be customized. (optional: if you want to use SSL, put narration features into ``narration_feat_folder``.)
2. ``cd ViLCo/MQ`` and change name of the above feature path. You could set up your own config file. We provide many template in configs folder. In ``configs/Your_Own_Config.yaml``, you should change ``feat_folder`` for visual features, ``text_feat_folder`` for textual_feat_folder, and ``narration_feat_folder`` for narration features.
3. You should put mata file for CL ``ego4d_mq_query_incremental_22_all.pkl`` into the folder ``./data/ego4d``. If the folder does not exist, build manually.

**For NLQ,**
1. You should put visual features into ``feat_folder``, textual features into ``text_feat_folder``. The dictionary name could be customized. (optional: if you want to use SSL, put narration features into ``narration_feat_folder``.)
2. ``cd ViLCo/NLQ`` and change name of the above feature path. You could set up your own config file. We provide many template in configs folder. In ``configs/Your_Own_Config.yaml``, you should change ``feat_folder`` for visual features, ``text_feat_folder`` for textual_feat_folder, and ``narration_feat_folder`` for narration features.
3. You should put mata file for CL ``ego4d_nlq_query_incremental_13.pkl`` into the folder ``./data/ego4d``. If the folder does not exist, build manually.

**For VQ,**
Please follow vq2d baseline (https://github.com/EGO4D/episodic-memory/tree/main/VQ2D#preparing-data-for-training-and-inference) step 1/2/4/5 to process the dataset into video clips.

### Model Architecture and Implementation

The full framework consists of several key components: a visual encoder (typically EgoVLP-v2), a textual encoder (usually CLIP), a multi-modal encoder, short-term/long-term memory modules, and task-specific heads.

Encoders: We start by leveraging pre-extracted visual and textual features using pre-trained encoders, which are mostly frozen during training. These encoders capture the essential information needed for the tasks without requiring extensive additional training.

Uni-Modal and Cross-Modal Encoders: We introduce additional trainable transformer layers as uni-modal encoders for visual and textual streams, followed by cross-modal encoders that integrate features across modalities. This hierarchical structure allows for better spatial-temporal representation and precise moment localization.

Memory Modules: The memory system includes both short-term and long-term memory:Short-Term Memory stores samples from recent tasks, allowing for efficient replay.Long-Term Memory retains critical episodic knowledge that is crucial for cross-task generalization. Both memories are updated using selective strategies during training.

Task-Specific Heads and Losses: Each task (e.g., MQ, NLQ, VQ) has a dedicated prediction head trained with the corresponding task-specific loss functions. We also introduce a self-supervised loss to fine-tune the trainable visual and textual encoders and a contrastive loss to refine the episodic memory, ensuring that the memory updates with relevant and diverse information.

### Training and Evaluation Protocols

Training,

For MQ,
```
cd MQ
bash train_cl.sh [your_config_name] [GPU_id] [Port]
```

For NLQ,
```
cd NLQ
torchrun --rdzv_endpoint=localhost:29910 --nproc_per_node=1  train_cl.py configs/your_config.yaml --output vilco
```

For VQ,
```
cd VQ
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=8 train_cl.py --cfg ./config/your_config.yaml
```

Run the following command to train and evaluate the model:
```
bash train_cl.sh mq_vilco DEVICE_NO PORT_NO
```
(taking MQ as an example), it will automatically validate performance on val-set (e.g., average mAP, Recall@1x).

### Completed Tasks

1. We complete the evaluation with query-incremental learning on moments query, natural language query and visual query tasks.
2. We could extend ViLCo with different datasets and backbones.

### Ongoing Work

We plan to extend our benchmark to more challenging scenarios and introduce LLMs for their strong zero-shot generalization. For example, OpenEQA is an open-vocabulary question-answer dataset with VQA tasks. We want to propose a new framework involving LLMs to keep the strong generalization ability and learn new knowledge continually.

### Future Work and Recommendations

1. Assignment-incremental Challenge: training models with the above three tasks (including MQ, NLQ and VQ). The first step is to pre-train backbone (e.g., EgoVLP-v2) on data. However, due to the time and GPU memory limitation, it is hard to complete before August, 2024.
2. Time-incremental Challenge: Pay more attention to long-term video understanding within continual learning. It is more challenging, we plan to move this aspect to our ongoing project (LLMs with continual learning for OpenEQA.)

## Cluster and Tool Usage

Internal sharing

### IONA

MQ visual features are stored in the dictionary: ``/mnt/data728/datasets/ego4d_data/features/EgoVLPv2``.

MQ textual features are stored in the dictionary: ``/mnt/data728/datasets/ego4d_data/features/CLIP_text_features_mq``.

MQ narration features are stored in the dictionary: ``/mnt/data728/datasets/ego4d_data/features/em_narration_clip_token_features``.

MQ/NLQ/VQ metadata is stored in the dictionary: ``/mnt/data728/datasets/ego4d_data/metadata``.

NLQ features are stored in: ``/mnt/data728/datasets/ego4d_data/features/features_lmdb``

## References and Appendices

Here list most helpful papers and codes. 

1. vCLIMB: A Novel Video Class Incremental Learning Benchmark [Link](https://arxiv.org/abs/2201.09381) [Code](https://github.com/ojedaf/vCLIMB_Benchmark)
2. CLiMB: The Continual Learning in Multimodality Benchmark [Link](https://arxiv.org/abs/2206.09059) [Code](https://github.com/GLAMOR-USC/CLiMB)
3. Ego4D: Around the World in 3,000 Hours of Egocentric Video [Link](https://arxiv.org/abs/2110.07058) [Code](https://github.com/facebookresearch/Ego4d)
4. Egocentric Video-Language Pretraining [Link](https://arxiv.org/abs/2206.01670) [Code](https://github.com/showlab/EgoVLP)
5. EgoVLPv2: Egocentric Video-Language Pre-training with Fusion in the Backbone [Link](https://arxiv.org/abs/2307.05463) [Code](https://github.com/facebookresearch/EgoVLPv2?tab=readme-ov-file)
6. Action Sensitivity Learning for the Ego4D Episodic Memory Challenge 2023 [Link](https://arxiv.org/abs/2306.09172) [Code](https://github.com/jonnys1226/ego4d_asl)
7. GroundNLQ @ Ego4D Natural Language Queries Challenge 2023 [Link](https://arxiv.org/abs/2306.15255) [Code](https://github.com/houzhijian/GroundNLQ)
8. Single-Stage Visual Query Localization in Egocentric Videos [Link](https://arxiv.org/abs/2306.09324) [Code](https://github.com/hwjiang1510/VQLoC)
9. Learning to Prompt (L2P) for Continual Learning [Link](https://arxiv.org/abs/2112.08654) [Code](https://github.com/google-research/l2p)