# Awesome-3D-Radiology-Analysis-FM

## Contents

1. [Foundation Models](#foundation-models)  
   1.1 [Self-Supervised Pre-training](#self-supervised-pre-training)  
   1.2 [Contrastive Learning (CLIP-based Methods)](#contrastive-learning-clip-based-methods)  
   1.3 [MLLM-based Methods](#mllm-based-methods)  
   1.3.1 [2D-only](#2d-only)  
   1.3.2 [Unified 2D + Serialized-3D](#unified-2d--serialized-3d)  
   1.3.3 [Native 3D-Volume](#native-3d-volume)
2. [Agentic Systems](#agentic-systems)
3. [Datasets & Benchmarks](#datasets--benchmarks)
4. [Related Surveys](#related-surveys)

---
## Foundation Models

### **Self-Supervised Pre-training**
| Method | Title | Venue | Date | Paper | Project |
| --- | --- | --- | --- | --- | --- |
|  | Models Genesis | Medical Image Analysis | 21.02 | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841520302048) | [Project](https://github.com/MrGiovanni/ModelsGenesis) |
| UniMiSS | UniMiSS: Universal Medical Self-Supervised Learning via Breaking Dimensionality Barrier | ECCV2022&TPAMI | 21.12 | [Paper](https://arxiv.org/abs/2112.09356) | [Project](https://github.com/YtongXie/UniMiSS-code) |
| Swin-UNETR | Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis | CVPR 2022 | CVPR 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf) | [Project](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) |
| PCRLv2 | A Unified Visual Information Preservation Framework for Self-supervised Pre-training in Medical Image Analysis | IEEE TPAMI | 2023.01 | [Paper](https://arxiv.org/abs/2301.00772) | [Project](https://github.com/RL4M/PCRLv2) |
| MIM-Med3D | Masked Image Modeling Advances 3D Medical Image Analysis | WACV 2023 | 2023.01 | [Paper](https://arxiv.org/abs/2204.11716) | [Project](https://github.com/chenz53/MIM-Med3D) |
| GVSL | Geometric Visual Similarity Learning in 3D Medical Image Self-supervised Pre-training | CVPR 2023 | 2023.03 | [Paper](https://arxiv.org/pdf/2303.00874) | [Project](https://github.com/YutingHe-list/GVSL) |
| M3AE | M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities | AAAI 2023 | 2023.03 | [Paper](https://arxiv.org/abs/2303.05302) | [Project](https://github.com/ccarliu/m3ae) |
| HybridMIM | HybridMIM: A Hybrid Masked Image Modeling Framework for 3D Medical Image Segmentation | IEEE JBHI | 2024.04 | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10416665) | [Project](https://github.com/ge-xing/HybridMIM) |
| VoCo | Large-Scale 3D Medical Image Pre-Training With Geometric Context Priors | CVPR 2024 / IEEE TPAMI | 2024.06 | [Paper](https://ieeexplore.ieee.org/abstract/document/11274411) | [Project](https://github.com/Luffy03/VoCo) |
| CDSSL-P3D | Cross-Dimensional Medical Self-Supervised Representation Learning Based on a Pseudo-3D Transformation | MICCAI 24 | 2024.10 | [Paper](https://arxiv.org/abs/2406.00947) | - |
| MDM | Masked Deformation Modeling for Volumetric Brain MRI Self-Supervised Pre-Training | IEEE TMI | 2024.12 | [Paper](https://ieeexplore.ieee.org/document/10777582) | [Project](https://github.com/CRazorback/MDM) |
| DAE | Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training | MIDL 2024 | 2024.06 | [Paper](https://arxiv.org/pdf/2307.16896) | [Project](https://github.com/Project-MONAI/research-contributions/tree/main/DAE) |
|  | Unified 3D MRI Representations via Sequence-Invariant Contrastive Learning | arXiv | 2025.01 | [Paper](https://arxiv.org/abs/2501.12057v2) | [Project](https://github.com/liamchalcroft/contrast-squared) |
| GzPT | Improving Self-Supervised Medical Image Pre-Training by Early Alignment With Human Eye Gaze Information | AAAI24/IEEE TMI | 2025.01 | [Paper](https://ieeexplore.ieee.org/document/10839445) | [Project](https://github.com/zhaozh10/McGIP) |
| FM-HCT | 3D Foundation AI Model for Generalizable Disease Detection in Head Computed Tomography | arXiv | 2025.02 | [Paper](https://arxiv.org/pdf/2502.02779) | [Project](https://github.com/NYUMedML/headCT_foundation) |
|  | MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders | MIDL 2025 | 2025.02 | [Paper](https://arxiv.org/abs/2502.14753) | [Project](https://github.com/StanfordMIMI/MedVAE) |
| MiM | MiM: Mask in Mask Self-Supervised Pre-Training for 3D Medical Image Analysis | IEEE TMI | 2025.04 | [Paper](https://arxiv.org/abs/2404.15580) | - |
| BrainMVP | BrainMVP: Multi-modal Vision Pre-training for Brain MRI Analysis | CVPR 2025 (Highlight) | 2025.06 | [Paper](https://arxiv.org/html/2410.10604v1) | [Project](https://github.com/shaohao011/BrainMVP) |
| SPECTRE | Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers | CVPR 2026 | 2025.11 | [Paper](https://arxiv.org/abs/2511.17209) | [Project](https://github.com/cclaess/SPECTRE) |
| SubFore | HU-based Foreground Masking for 3D Medical Masked Image Modeling | MICCAI25 | 2025.10 | [Paper](https://arxiv.org/pdf/2509.07534) | [Project](https://github.com/AISeedHub/SubFore/) |
| 3DINO | A generalizable 3D framework and model for self-supervised learning in medical imaging | npj Digital Medicine | 2025.11 | [Paper](https://www.nature.com/articles/s41746-025-02035-w) | [Project](https://github.com/AICONSlab/3DINO?tab=readme-ov-file) |
| TotalFM | TotalFM: An Organ-Separated Framework for 3D-CT Vision Foundation Models | arxiv | 2026.01 | [Paper](https://arxiv.org/pdf/2601.00260) | - |
| Curia-2 | Curia-2: Scaling Self-Supervised Learning for Radiology Foundation Models | arxiv | 2026.02 | [Paper](https://arxiv.org/pdf/2604.01987) | - |
| OCTCube-M | A 3D multimodal optical coherence tomography foundation model for retinal and systemic diseases with cross-cohort and cross-device validation | Nature Biomedical Engineering | 2026.04 | [Paper](https://www.nature.com/articles/s41551-026-01662-2) | [Project](https://github.com/ZucksLiu/OCTCubeM/tree/main) |
| NeuroSTORM | Towards a general-purpose foundation model for fMRI analysis | NBME | 2026.03 | [Paper](https://www.researchsquare.com/article/rs-6728658/v1) | [Project](https://github.com/CUHK-AIM-Group/NeuroSTORM?tab=readme-ov-file) |
| Triad | Vision foundation model for 3D magnetic resonance imaging segmentation, classification, and registration | MedIA | 2026.05 | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841526000617) | - |
| Foundation-VAE | Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation | ICML 2026 | 2026.05 | [Paper](https://arxiv.org/pdf/2605.30893) | [Project](https://github.com/qic999/Foundation-VAE) |
| CoralBay | CoralBay: A Self-Supervised CT Foundation Model | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.03888) | - |
|  | How Much MRI Preprocessing Is Enough? A Cost-Utility Study for Brain MRI Foundation Models | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.08164) | - |
| NeuroVFM | Health system learning enables generalist neuroimaging models | Nature Medicine | 2026.07 | [Paper](https://www.nature.com/articles/s41591-026-04497-1) | [Project](https://github.com/MLNeurosurg/neurovfm) |
| BrainFIBRE | BrainFIBRE: A Foundation Model via Information Decomposition for Brain Microstructure | ECCV 2026 / arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.00573) | - |
| BoneCoT / BoneFM | BoneCoT: multicentre validation of a whole-body skeleton foundation model for bone metastases guided by clinician-derived chain of thought | Nature Biomedical Engineering | 2026.07 | [Paper](https://doi.org/10.1038/s41551-026-01736-1) | [Project](https://github.com/FrankZhangRp/BoneCoT) |
| Cardiac CT FM | A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping: Human-in-the-Loop Data Annotation, Vision Foundation Model Development, Multicenter Evaluation and Clinical Validation | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.11287) | - |
| COJEPA | Contrastive Joint-Embedding Prediction for Representation Learning in Structural MRI | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.11962) | - |
| BrainNext | BrainNext: A General-Purpose Self-Supervised Foundation Model for Brain MRI Analysis | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.17782) | - |
| OrganLens | OrganLens: Organ-Specific Representation Learning for CT Foundation Models | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.25164) | - |
| Rad-JEPA 3D | Rad-JEPA 3D: Radiology Joint-Embedding Predictive Model for 3D Computed Tomography | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.26196) | - |
---

### **Contrastive Learning (CLIP-based Methods)**
| Method | Dimension | Title | Venue | Date | Paper | Project |
| --- | --- | --- | --- | --- | --- | --- |
| MedCLIP | 2D | MedCLIP: Contrastive Learning from Unpaired Medical Images and Text | EMNLP 2022 | 2022.12 | [Paper](https://arxiv.org/abs/2210.10163) | [Project](https://github.com/RyanWangZf/MedCLIP) |
| PubMedCLIP | 2D | PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain? | EACL 2023 (Findings) | 2023.05 | [Paper](https://aclanthology.org/2023.findings-eacl.88.pdf) | [Project](https://github.com/sarahESL/PubMedCLIP) |
| MedBLIP | 2D | MedBLIP: Bootstrapping Language-Image Pre-training from 3D Medical Images and Texts | arxiv | 2023.05 | [Paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Chen_MedBLIP_Bootstrapping_Language-Image_Pre-training_from_3D_Medical_Images_and_Texts_ACCV_2024_paper.pdf) | - |
| CLIP-Lung | 2D | CLIP-Lung: Textual Knowledge-Guided Lung Nodule Malignancy Prediction | MICCAI 2023 | 2023.10 | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_38) | [Project](https://github.com/ymLeiFDU/CLIP-Lung) |
| BioMedCLIP | 2D | BiomedCLIP: A Multimodal Biomedical Foundation Model Pretrained from Fifteen Million Scientific Image-Text Pairs | NEJM AI 2024 | 2024.01 | [Paper](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) | [Project](https://github.com/microsoft/BiomedCLIP_data_pipeline) |
| PMC-CLIP | 2D | PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents | MICCAI 2023 | 2023.10 | [Paper](https://arxiv.org/abs/2303.07240) | [Project](https://weixionglin.github.io/PMC-CLIP/) |
| UniMedCLIP | 2D | UniMed-CLIP: Towards a Unified Image-Text Pretraining Paradigm for Diverse Medical Imaging Modalities | Arxiv | 2024.12 | [Paper](https://arxiv.org/abs/2412.10372) | [Project](https://github.com/mbzuai-oryx/UniMed-CLIP) |
| ConceptCLIP | 2D | ConceptCLIP: Towards Trustworthy Medical AI via Concept-Enhanced Contrastive Language-Image Pre-training | arXiv | 2025.01 | [Paper](https://arxiv.org/abs/2501.15579) | [Project](https://github.com/JerrryNie/ConceptCLIP) |
| MMKD-CLIP | 2D | Unifying Biomedical Vision-Language Expertise: Towards a Generalist Foundation Model via Multi-CLIP Knowledge Distillation | Arxiv | 2025.06 | [Paper](https://arxiv.org/pdf/2506.22567) | - |
| RadiSimCLIP | 2D | RadiSimCLIP: A Radiology Vision-Language Model Pretrained on Simulated Radiologist Learning Dataset for Zero-Shot Medical Image Understanding | MICCAI 2025 Workshop | 2025.10 | [Paper](https://link.springer.com/chapter/10.1007/978-3-032-07845-2_8) | [Project](https://github.com/so-ux/RadiSimCLIP) |
| UniBrain | 3D | UniBrain: Universal Brain MRI Diagnosis with Hierarchical Knowledge-enhanced Pre-training | Computerized Medical Imaging and Graphics | 2023.09 | [Paper](https://arxiv.org/abs/2309.06828) | [Project](https://github.com/ljy19970415/UniBrain) |
| CT2Rep | 3D | CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging | MICCAI 2024 | 2024.03 | [Paper](https://arxiv.org/pdf/2403.06801) | [Project](https://github.com/ibrahimethemhamamci/CT2Rep) |
| CT-CLIP | 3D | Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography | Nat Biomed Eng | 2024.03 | [Paper](https://arxiv.org/abs/2403.17834) | [Project](https://github.com/ibrahimethemhamamci/CT-CLIP) |
| CT-GLIP | 3D | CT-GLIP: 3D Grounded Language-Image Pretraining with CT Scans and Radiology Reports for Full-Body Scenarios | arXiv | 2024.04 | [Paper](https://arxiv.org/abs/2404.15272) | - |
| RadCLIP | 3D | RadCLIP: Enhancing Radiologic Image Analysis Through Contrastive Language–Image Pretraining | TNNLS | 2024.03 | [Paper](https://arxiv.org/pdf/2403.09948) | - |
| Percival | 3D | A Pan-Organ Vision-Language Model for Generalizable 3D CT Representations | medRxiv | 2025.07 | [Paper](https://pubmed.ncbi.nlm.nih.gov/40630577/) | - |
| OpenVocabCT | 3D | Towards Universal Text-driven CT Image Segmentation | arXiv | 2025.03 | [Paper](https://arxiv.org/abs/2503.06030) | - |
| fVLM | 3D | Large-scale and Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding | ICLR 2025 | 2025.03 | [Paper](https://arxiv.org/abs/2501.14548) | [Project](https://github.com/alibaba-damo-academy/fvlm) |
| HLIP | 3D | Towards Scalable Language-Image Pre-training for 3D Medical Imaging | arxiv | 2025.05 | [Paper](https://arxiv.org/abs/2505.21862) | [Project](https://github.com/zch0414/hlip) |
| RadZero3D | 3D | RadZero3D: Bridging Self-Supervised Video Models and Medical Vision-Language Alignment for Zero-Shot Chest CT Interpretation | ICCV 2025 Workshop  | 2025.10 | [Paper](https://ieeexplore.ieee.org/abstract/document/11375501) | - |
| T3D | 3D | T3D: Advancing 3D Medical Vision-Language Pre-training by Learning Multi-View Visual Consistency | ICCV 2025 Workshop  | 2025.10 | [Paper](https://openaccess.thecvf.com/content/ICCV2025W/VLM3D/papers/Liu_T3D_Advancing_3D_Medical_Vision-Language_Pre-training_by_Learning_Multi-View_Visual_ICCVW_2025_paper.pdf) | - |
| ViSD-Boost | 3D | Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training | ICCV 2025 | 2025.08 | [Paper](https://arxiv.org/abs/2508.03742) | [Project](https://github.com/alibaba-damo-academy/ViSD-Boost) |
| VELVET-Med | 3D | VELVET-Med: Vision and Efficient Language Pre-training for Volumetric Imaging Tasks in Medicine | arXiv | 2025.08 | [Paper](https://arxiv.org/abs/2508.12108) | - |
| MedVista3D | 3D | MedVista3D: Vision-Language Modeling for Reducing Diagnostic Errors in 3D CT Disease Detection, Understanding and Reporting | arXiv | 2025.09 | [Paper](https://arxiv.org/abs/2509.03800) | - |
| COLIPRI | 3D | Comprehensive Language-Image Pre-training for 3D Medical Image Understanding | arXiv | 2025.10 | [Paper](https://arxiv.org/abs/2510.15042) | [Project](https://huggingface.co/microsoft/colipri) |
| MPS-CT | 3D | More performant and scalable: Rethinking contrastive vision-language pre-training of radiology in the LLM era | MICCAI 2025 | 2025.10 | [Paper](https://arxiv.org/pdf/2509.13175) | [Project](https://github.com/SadVoxel/More-performant-and-scalable) |
| PET-CLIP Captioner | 3D | Location-Guided Automated Lesion Captioning in Whole-body PET/CT Images | MICCAI 2025 | 2025.10 | [Paper](https://papers.miccai.org/miccai-2025/paper/0248_paper.pdf) | - |
| MR-CLIP | 3D | Metadata-Aligned 3D MRI Representations for Contrast Understanding and Quality Control | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.00681) | - |
| BrgSA | 3D | Bridged Semantic Alignment for Zero-shot 3D Medical Image Diagnosis | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2501.03565) | [Project](https://github.com/laihaoran/BrgSA) |
| SCALE-VLP | 3D | SCALE-VLP: Soft-Weighted Contrastive Volumetric Vision-Language Pre-training with Spatial-Knowledge Semantics | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.02996) | - |
| SPECTRE | 3D | Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers | CVPR 2026 | 2025.11 | [Paper](https://arxiv.org/abs/2511.17209) | [Project](https://github.com/cclaess/SPECTRE) |
| **Pillar-0** | 3D | Pillar-0: A New Frontier for Radiology Foundation Models | arxiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.17803) | [Project](https://yalalab.github.io/pillar-0/) |
| BTB3D | 3D | Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging | NeurIPS 2025 | 2025.12 | [Paper](https://arxiv.org/abs/2510.20639) | [Project](https://github.com/ibrahimethemhamamci/BTB3D) |
| NeuroVFM | 3D | NeuroVFM: A Contrastive Vision-Language Model for Medical Reasoning in Alzheimer's Disease Diagnosis | WACV 2026 Workshops | 2026.01 | [Paper](https://openaccess.thecvf.com/content/WACV2026W/P2P/html/Sajib_NeuroVLM_A_Contrastive_Vision-Language_Model_for_Medical_Reasoning_in_Alzheimers_WACVW_2026_paper.html) | - |
| TotalFM | 3D | TotalFM: An Organ-Separated Framework for 3D-CT Vision Foundation Models | arxiv | 2026.01 | [Paper](https://arxiv.org/pdf/2601.00260) | - |
| MG-3D | 3D | MG-3D: Multi-Grained Knowledge-Enhanced 3D Medical Vision-Language Pre-training | Medical Image Analysis (MedIA) | 2026.01 | [Paper](https://arxiv.org/abs/2503.06030) | [Project](https://github.com/Xuefeng-Ni/MG-3D) |
| MedMAP | 3D | 3D Modality-Aware Pre-training for Vision-Language Model in MRI Multi-organ Abnormality Detection | arxiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.23652) | [Project](https://github.com/RomantiDr/) |
| Prima | 3D | Learning neuroimaging models from health system-scale data | Nature Biomedical Engineering | 2026.02 | [Paper](https://www.nature.com/articles/s41551-025-01608-0) | [Project](https://github.com/MLNeurosurg/Prima) |
| SigVLP | 3D | SigVLP: Sigmoid Volume-Language Pre-Training for Self-Supervised CT-Volume Adaptive Representation Learning | arXiv | 2026.02 | [Paper](https://arxiv.org/pdf/2602.21735) | - |
| RadFinder | 3D | Learning to Read Where to Look: Disease-Aware Vision–Language Pretraining for 3D CT | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.02026) | [Project](https://radfinder.github.io/) |
| Decipher-MR | 3D | Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations | npj Digital Medicine | 2026.04 | [Paper](https://www.nature.com/articles/s41746-026-02596-4) | - |
|  | 3D | CLIP Architecture for Abdominal CT Image–Text Alignment and Zero-Shot Learning: Investigating Batch Composition and Data Scaling | arxiv | 2026.04 | [Paper](https://arxiv.org/pdf/2604.13561) | - |
| ASAP | 3D | ASAP: Advancing Medical Volumetric Representation Learning with Anatomy-aware Semantically-adaptive Pre-training | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2606.00602) | - |
| GLeVE | 3D | GLeVE: Graph-Guided Lesion Grounding with Proposal Verification in 3D CT | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.22619) | - |
| CA-GCL | 3D | CA-GCL: Cross-Anatomy Global-Local Contrastive Learning for Robust 3D Medical Image Understanding | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.13544) | - |
| SegReg-Rep | 3D | SegReg-Rep: Region-Aware Vision-Language Alignment for Fine-Grained Radiology Report Generation from 3D Medical Images | IEEE TPRMS | 2026.05 | [Paper](https://ieeexplore.ieee.org/abstract/document/11520942/authors) | [Project](https://github.com/BlueRainbow2000/SegReg-Rep) |
| GLINT | 3D | GLINT: Sparsely Gated Vision-Language Alignment for Fine-Grained Radiology Representations | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.03180) | - |
| RadGrounder | 2D | Scalable Training of Spatially Grounded 2D Vision-Language Models for Radiology | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.20477) | - |
| RenalCLIP | 3D | A Disease-Centric Vision-Language Foundation Model for Precision Oncology in Kidney Cancer | Nature Communications | 2026.06 | [Paper](https://www.nature.com/articles/s41467-026-74175-w) | - |
| Jolia / ConQuer | 3D | Jolia: Concept-Level Vision-Language Alignment for 3D CT Contrastive Learning | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.24570) | - |
|  | 3D | Disease-Centric Vision-Language Pretraining with Hybrid Visual Encoding for 3D Computed Tomography | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.25546) | - |
| MedReCo | 3D | A Vision-language Framework for Comparative Reasoning in Radiology | arXiv | 2026.06 | [Paper](https://arxiv.org/pdf/2606.06407) | - |
| SuG | 3D | Super-Generalist: Towards Comprehensive and Accurate Medical Image Understanding via Generalist-Specialist Synergy | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.09135) | - |
| OKA-CT | 3D | Learning Anatomy-Grounded CT Vision-Language Representations with Organ-Hierarchical Report Knowledge | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.10953) | - |
| OCP-CT | 3D | Fine-Grained Vision-Language Pretraining with Organ-Conditioned Pattern Tokens for CT Understanding | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.13892) | - |
| MseaCL | 3D | Multimodal Semantic-Aware Contrastive Learning For False Negative Mitigation in 3D Medical Imaging | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.14995) | - |
| CARVE | 3D | When Can Test-Time Adaptation Help Zero-Shot CT Vision-Language Models? | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.15556) | - |
| ACA | 3D | Anatomy Contextualized Adaption of CT Foundation Models | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.27154) | - |
| Spectrum | 3D | Learning How Much, Not Just What: Cross-Patient Burden Order for CT Vision-Language Pretraining | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2608.00231) | - |
| SCOPE | 3D | Semantically Calibrated Evidence Composition for CT Vision-Language Learning | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2608.00239) | - |
---

### MLLM-based Methods


#### **2D-only**
| Method | Title | Venue | Date | Paper | Project |
| --- | --- | --- | --- | --- | --- |
| BiomedGPT | BiomedGPT: A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks | Nat Med 2024 | 2023.05 | [Paper](https://arxiv.org/abs/2305.17100) | [Project](https://huggingface.co/PharMolix/BioMedGPT-LM-7B) |
| MedVInT | PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering | Arxiv/ Communications Medicine | 2023.05/2024.12 | [Paper](https://arxiv.org/abs/2305.10415) | - |
| LLaVA-Med | LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day | NeurIPS 2023 | 2023.06 | [Paper](https://arxiv.org/abs/2306.00890) | [Project](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) |
| Med-Flamingo | Med-Flamingo: a Multimodal Medical Few-shot Learner | ML4H 2023 | 2023.07 | [Paper](https://arxiv.org/abs/2307.15189) | [Project](https://huggingface.co/med-flamingo/med-flamingo) |
| Med-PaLM M | Towards Generalist Biomedical AI | NEJM AI 2024 / arXiv | 2023.07 | [Paper](https://arxiv.org/abs/2307.14334) | [Project](https://sites.research.google/gr/med-palm/) |
| Qilin-Med-VL | Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare | arXiv | 2023.10 | [Paper](https://arxiv.org/abs/2310.17956) | - |
| R2GenGPT | R2GenGPT: Radiology Report Generation with Frozen LLMs | Meta-Radiology | 2023.11 | [Paper](https://arxiv.org/abs/2309.09812) | - |
| BiRD | A Refer-and-Ground Multimodal Large Language Model for Biomedicine | MICCAI 2024 | 2024.06 | [Paper](https://arxiv.org/abs/2406.18146) | - |
| HuatuoGPT-Vision | Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale | EMNLP24 | 2024.06 | [Paper](https://arxiv.org/abs/2406.19280) | [Project](https://github.com/freedomintelligence/huatuogpt-vision) |
| Llama3-Med | Advancing High Resolution Vision-Language Models in Biomedicine | arXiv | 2024.06 | [Paper](https://arxiv.org/abs/2406.09454) | [Project](https://github.com/standardmodelbio/Llama3-Med) |
| MiniGPT-Med | MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis | arXiv | 2024.07 | [Paper](https://arxiv.org/abs/2407.04106) | [Project](https://github.com/Vision-CAIR/MiniGPT-Med) |
| TinyLLaVA-Med | Democratizing MLLMs in Healthcare: TinyLLaVA-Med for Efficient Healthcare Diagnostics in Resource-Constrained Settings | MICCAI24 | 2024.09 | [Paper](https://arxiv.org/abs/2409.12184) | - |
| Med-MoE | Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models | EMNLP24 | 2024.09 | [Paper](https://arxiv.org/pdf/2404.10237) | [Project](https://github.com/jiangsongtao/Med-MoE) |
| GMAI-VL | GMAI-VL & GMAI-VL-5.5M: A Large Vision-Language Model and A Comprehensive Multimodal Dataset Towards General Medical AI | arXiv/AAAI26 | 2024.11 | [Paper](https://arxiv.org/abs/2411.14522) | [Project](https://github.com/uni-medical/GMAI-VL) |
| BiMediX2 | BiMediX2: Bio-Medical EXpert LMM for Diverse Medical Modalities | Findings of EMNLP 2025 | 2025.01 | [Paper](https://aclanthology.org/2025.findings-emnlp.756/) | [Project](https://github.com/mbzuai-oryx/BiMediX2) |
| HealthGPT | HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation | ICML 2025 | 2025.02 | [Paper](https://arxiv.org/abs/2502.09838) | [Project](https://github.com/DCDmllm/HealthGPT) |
| MedVLM-R1 | MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning | MICCAI25 | 2025.02 | [Paper](https://arxiv.org/abs/2502.19634) | [Project](https://huggingface.co/JZPeterPan/MedVLM-R1) |
| Med-R1 | Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models | TMI | 2025.03 | [Paper](https://arxiv.org/abs/2503.13939) | [Project](https://github.com/Yuxiang-Lai117/Med-R1) |
| OmniV-Med | OmniV-Med: Scaling Medical Vision-Language Model for Universal Visual Understanding | arXiv | 2025.04 | [Paper](https://arxiv.org/abs/2504.14692) | - |
| UniBioMed | UniBiomed: A Universal Foundation Model for Grounded Biomedical Image Interpretation | arXiv | 2025.04 | [Paper](https://arxiv.org/abs/2504.21336) | [Project](https://github.com/Luffy03/UniBiomed) |
| MedRegA | MedRegA: Interpretable Bilingual Multimodal Large Language Model for Diverse Biomedical Tasks | ICLR25 | 2025.04 | [Paper](https://arxiv.org/abs/2410.18387) | [Project](https://medrega.github.io/) |
| UMed-LVLM | Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback | ACL 2025 | 2025.05 | [Paper](https://aclanthology.org/2025.acl-long.636/) | - |
| QoQ-Med | QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training | arXiv | 2025.06 | [Paper](https://arxiv.org/abs/2506.00711) | [Project](https://huggingface.co/ddvd233/QoQ-Med-VL-7B) |
| Lingshu | Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning | arXiv | 2025.06 | [Paper](https://arxiv.org/abs/2506.07044) | [Project](https://huggingface.co/collections/lingshu-medical-mllm/lingshu-mllms) |
| MedGemma | MedGemma Technical Report | arXiv | 2025.07 | [Paper](https://arxiv.org/abs/2507.05201) | [Project](https://huggingface.co/google/medgemma-4b-it) |
| Critus-V | Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning | arXiv | 2025.09 | [Paper](https://arxiv.org/abs/2509.19090) | [Project](https://github.com/jd-opensource/Citrus-V) |
| MedPLIB | Towards a Multimodal Large Language Model with Pixel-Level Insight for Biomedicine | AAAI25 | 2025.10 | [Paper](https://arxiv.org/pdf/2412.09278) | [Project](https://github.com/ShawnHuang497/MedPLIB) |
| OctoMed | OctoMed: Data Recipes for State-of-the-Art Multimodal Medical Reasoning | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.23269) | [Project](https://huggingface.co/OctoMed/OctoMed-7B) |
| MedMO | MedMO: Grounding and Understanding Multimodal Large Language Model for Medical Images | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.06965) | [Project](https://huggingface.co/MBZUAI/MedMO-4B) |
| MediX-R1 | MediX-R1: Open Ended Medical Reinforcement Learning | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.23363) | [Project](https://huggingface.co/MBZUAI/MediX-R1-2B) |
| MEDIC-AD | MEDIC-AD: Towards Medical Vision-Language Model’s Clinical Intelligence | CVPR26 Oral | 2026.03 | [Paper](https://arxiv.org/pdf/2603.27176) | [Project](https://github.com/AIDASLab/Medic-AD) |
| MedVR | MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning | ICLR 26 | 2026.04 | [Paper](https://arxiv.org/pdf/2604.08203) | [Project](https://github.com/alibaba-damo-academy/MedVR) |
---

#### **Unified 2D + Serialized-3D**

| Method | Title | Venue | Date | Paper | Project |
| --- | --- | --- | --- | --- | --- |
| RadFM | Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data | Nat Commun 2025 / arXiv | 2023.08 | [Paper](https://arxiv.org/abs/2308.02463) | [Project](https://github.com/chaoyi-wu/RadFM) |
| Med-Gemini | Advancing Multimodal Medical Capabilities of Gemini | Nat Med 2025 / arXiv | 2024.05 | [Paper](https://arxiv.org/abs/2405.03162) | [Project](https://research.google/blog/advancing-medical-ai-with-med-gemini/) |
| Med-2E3 | Med-2E3: A 2D-Enhanced 3D Medical Multimodal Large Language Model | BIBM25 | 2024.11 | [Paper](https://arxiv.org/abs/2411.12783) | - |
| Hulu-Med | Hulu-Med: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding | arXiv | 2025.10 | [Paper](https://arxiv.org/abs/2510.08668) | [Project](https://github.com/ZJUI-AI4H/Hulu-Med) |
| Fleming-VL | Fleming-VL: Towards Universal Medical Visual Reasoning with Multimodal LLMs | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.00916) | [Project](https://huggingface.co/UbiquantAI/Fleming-VL-8B) |
| MedM-VL | MedM-VL: What Makes a Good Medical LVLM? | International Workshop on Agentic AI for Medicine 2025 | 2025.09 | [Paper](https://arxiv.org/pdf/2504.04323?) | [Project](https://huggingface.co/OctoMed/OctoMed-7B) |
| CTInstruct | CTInstruct: Towards Unified 3D CT Understanding via Instruction Tuning | AAAI 26 | 2026.01 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37517) | [Project](https://github.com/ljy19970415/CTInstruct) |
|  | A data-efficient 3D medical vision-language model using only a 2D encoder | Scientific report | 2026.02 | [Paper](https://www.nature.com/articles/s41598-026-39526-z) | - |
| MedPruner | MedPruner: Training-Free Hierarchical Token Pruning for Efficient 3D Medical Image Understanding in Vision-Language Models | MICCAI2026 | 2026.03 | [Paper](https://arxiv.org/abs/2603.11625) | - |
| Photon | Photon: Speedup Volume Understanding with Efficient Multimodal Large Language Models | ICLR 26 | 2026.03 | [Paper](https://arxiv.org/pdf/2603.25155) | [Project](https://github.com/alibaba-damo-academy/Photon) |
| OmniCT | OmniCT: Towards a Unified Slice-Volume LVLM for Comprehensive CT Analysis | ICLR 26 | 2026.03 | [Paper](https://arxiv.org/pdf/2602.16110) | [Project](https://github.com/ZJU4HealthCare/OmniCT) |
| MedGemma1.5 | MedGemma 1.5 Technical Report | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.05081) | [Project](https://huggingface.co/google/medgemma-1.5-4b-it) |
| TGH-MoE | Adapting 2D Multi-Modal Large Language Model for 3D CT Image Analysis | arXiv | 2026.04 | [Paper](https://arxiv.org/pdf/2604.10233) | - |
| Brain-Adapter | Brain-Adapter: A Dual-Stream Vision-Language MIL Framework for Comprehensive 3D CT Diagnosis of Acute Intracranial Pathologies | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.23494) | - |
| UniReason-Med | UniReason-Med: A Shared Grounded Reasoning Interface for 2D-to-3D Transfer in Medical VQA | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.11740) | [Project](https://github.com/IQuestLab/unireason-med) |
| MedReCo-VLM | A Vision-language Framework for Comparative Reasoning in Radiology | arXiv | 2026.06 | [Paper](https://arxiv.org/pdf/2606.06407) | - |
| RadSight | RadSight: Towards Perceptually Reliable Multimodal Radiology Image Understanding | arXiv | 2026.07 | [Paper](https://arxiv.org/pdf/2607.22293) | [Project](https://github.com/alibaba-damo-academy/damo-RadSight/) |
| ClinFusion | ClinFusion: A Vision-Centric Multimodal LLM System for Holistic Medical Understanding | arXiv | 2026.07 | [Paper](https://arxiv.org/pdf/2607.24743) | [Project](https://github.com/alibaba-damo-academy/ClinFusion) |
| Hounsfield | Towards Enhancing 3D Spatial Reasoning in Medical Multimodal Large Language Models | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.13860) | [Project](https://github.com/2020420145009/hounsfield) |
| MedARC | MedARC: Training-Free Adaptive Redundancy Compression of Visual Tokens for 3D Medical Vision-Language Models | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.26554) | - |
| ORCA | ORCA: ORgan-Centroid Aggregation for Training-Free 3D CT Visual Token Compression | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2608.00345) | [Project](https://github.com/renjie-liang/ORCA-3DCT) |
---

#### **Native 3D-Volume**

| Method | Title | Venue | Date | Paper | Project |
| --- | --- | --- | --- | --- | --- |
| CT2Rep | CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging | MICCAI 2024 / arXiv | 2024.03 | [Paper](https://arxiv.org/abs/2403.06801) | [Project](https://github.com/ibrahimethemhamamci/CT2Rep) |
| Dia-LLaMA | Dia-LLaMA: Towards Large Language Model-driven CT Report Generation | MICCAI | 2024.03 | [Paper](https://arxiv.org/abs/2403.16386) | - |
| M3D-LaMed | M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models | ICLR 2025 / arXiv | 2024.04 | [Paper](https://arxiv.org/abs/2404.00578) | [Project](https://github.com/BAAI-DCAI/M3D) |
| Merlin | Merlin: A Computed Tomography Vision-Language Foundation Model and Dataset | Nature 2026 / arXiv | 2024.06 | [Paper](https://arxiv.org/abs/2406.06512) | [Project](https://github.com/StanfordMIMI/Merlin) |
| BrainGPT | Towards a Holistic Framework for Multimodal Large Language Models in Three-dimensional Brain CT Report Generation | Nat Commun 2025 / arXiv | 2024.07 | [Paper](https://arxiv.org/abs/2407.02235) | [Project](https://github.com/charlierabea/FORTE) |
| 3D-CT-GPT | 3D-CT-GPT: Generating 3D Radiology Reports through Integration of Large Vision-Language Models | arXiv | 2024.09 | [Paper](https://arxiv.org/abs/2409.19330) | - |
| E3D-GPT | E3D-GPT: Enhanced 3D Visual Foundation for Medical Vision-Language Model | arXiv | 2024.10 | [Paper](https://arxiv.org/abs/2410.14200) | - |
| Reg2RG | Large Language Model with Region-guided Referring and Grounding for CT Report Generation | arXiv | 2024.11 | [Paper](https://arxiv.org/abs/2411.15539) | - |
| MS-VLM | Read Like a Radiologist: Efficient Vision-Language Model for 3D Medical Imaging Interpretation | arXiv | 2024.12 | [Paper](https://arxiv.org/abs/2412.13558) | - |
| MEPNet | MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation | arXiv | 2025.03 | [Paper](https://arxiv.org/abs/2503.17784) | - |
| Med3DVLM | Med3DVLM: An Efficient Vision-Language Model for 3D Medical Image Analysis | IEEE JBHI 2025 / arXiv | 2025.03 | [Paper](https://arxiv.org/abs/2503.20047) | [Project](https://github.com/mirthAI/Med3DVLM) |
| HSENet | HSENet: Hybrid Spatial Encoding Network for 3D Medical Vision-Language Understanding | arXiv | 2025.06 | [Paper](https://arxiv.org/abs/2506.09634) | - |
| MedRegion-CT | MedRegion-CT: Region-Focused Multimodal LLM for Comprehensive 3D CT Report Generation | arXiv | 2025.06 | [Paper](https://arxiv.org/abs/2506.23102) | - |
| mpLLM | Multimodal LLM With Hierarchical Mixture-of-Experts for VQA on 3D Brain MRI | arXiv | 2025.09 | [Paper](https://arxiv.org/abs/2509.25889) | - |
| 3DReasonKnee | 3DReasonKnee: Advancing Grounded Reasoning in Medical Vision Language Models | arXiv | 2025.10 | [Paper](https://arxiv.org/abs/2510.20967) | - |
| PETAR | PETAR: Localized Findings Generation with Mask-Aware Vision-Language Modeling for PET Automated Reporting | arXiv | 2025.10 | [Paper](https://arxiv.org/abs/2510.27680) | - |
| BTB3D | Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging | NeurlPS 2025 | 2025.10 | [Paper](https://arxiv.org/pdf/2510.20639) | [Project](https://github.com/ibrahimethemhamamci/BTB3D) |
| PETRG-3D | Vision-Language Models for Automated 3D PET/CT Report Generation | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.20145) | - |
| CTest-Metric | CTest-Metric: A Unified Framework to Assess Clinical Validity of Metrics for CT Report Generation | ISBI2026 | 2026.01 | [Paper](https://arxiv.org/abs/2601.11488) | - |
| Brain3D | Brain3D: Brain Report Automation via Inflated Vision Transformers in 3D | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.22098) | [Project](https://github.com/PRAISELab-PicusLab/BrainGemma3D) |
| Med3D-R1 | Med3D-R1: Incentivizing Clinical Reasoning in 3D Medical Vision-Language Models for Abnormality Diagnosis | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.01200) | - |
| LoV3D | LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.12071) | - |
| Ker-VLJEPA-3B | Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.23308)  | [Project](https://huggingface.co/IBI-CAAI/Ker-VLJEPA-3B) |
| U-VLM | U-VLM: Hierarchical Vision Language Modeling for Report Generation | arXiv | 2026.02 | [Paper](https://arxiv.org/pdf/2603.00479) | [Project](https://github.com/yinghemedical/U-VLM) |
| CT-CHAT | Generalist foundation models from a multimodal dataset for 3D computed tomography | Nature Biomedical Engineering | 2026.02 | [Paper](https://www.nature.com/articles/s41551-025-01599-y) | [Project](https://github.com/ibrahimethemhamamci/CT-CHAT) |
| MedVL-SAM2 | MedVL-SAM2: A unified 3D medical vision–language model for multimodal reasoning and prompt-driven segmentation | arXiv | 2026.01 | [Paper](https://arxiv.org/pdf/2601.09879) | - |
| BoiD | Enhancing 3D medical multi-modal large language models with integrated human body priors for computed tomography | Pattern Recognition | 2026.04 | [Paper](https://www.sciencedirect.com/science/article/pii/S0031320326005066) | - |
| DCP-PD | Enhancing Fine-Grained Spatial Grounding in 3D CT Report Generation via Discriminative Guidance | arxiv | 2026.04 | [Paper](https://arxiv.org/pdf/2604.10437) | - |
| SegReg-Rep | SegReg-Rep: Region-Aware Vision-Language Alignment for Fine-Grained Radiology Report Generation from 3D Medical Images | IEEE TPRMS | 2026.05 | [Paper](https://ieeexplore.ieee.org/abstract/document/11520942/authors) | [Project](https://github.com/BlueRainbow2000/SegReg-Rep) |
| CLarGen | Generating Reports or Repeating Templates? Measuring and Mitigating Template Collapse in 3D CT Report Generation | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.30984) | - |
| TIF-GRPO | Regulating Anatomy-Aware Rewards via Trajectory-Integral Feedback for Volumetric Computed Tomography Analysis | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.20277) | - |
| RAD3D-Prefix | Revisiting LLM Adaptation for 3D CT Report Generation: A Study of Scaling and Diagnostic Priors | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.17213) | - |
| E-MRL | E-MRL: Cross-view Aligned Evidence-driven Multimodal Reinforcement Learning for Reliable 3D Tumor Analysis | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.23888) | - |
| MRI2Rep | MRI2Rep: Autoregressive Structured Report Generation for 3D Liver MRI | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.25279) | - |
| NeuroVFM | Health system learning enables generalist neuroimaging models | Nature Medicine | 2026.07 | [Paper](https://www.nature.com/articles/s41591-026-04497-1) | [Project](https://github.com/MLNeurosurg/neurovfm) |
| PIPA | PIPA: Prior-Driven Prompting with Diagnosis-Oriented Retrieval-Augmentation for 3D Radiology Report Generation | IEEE TMI | 2026.07 | [Paper](https://doi.org/10.1109/TMI.2026.3710717) | [Project](https://github.com/CUHK-AIM-Group/PIPA) |
| MonteRET | MonteRET: AI Agent Enhancing Multimodal LLMs with Multi-granularity Knowledge Retrieval for Chest CT Report Generation | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.14264) | - |
| Multi-LLM MRI | Multi-LLM Collaborative MRI Report Generation for Visual Instruction Tuning in Brain Oncology | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.14581) | - |
---
## **Agentic Systems**

| Method | Title | Venue | Date | Paper | Project |
| --- | --- | --- | --- | --- | --- |
| MDAgents | MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making | NeurIPS 2024 | 2024.12 (arXiv: 2024.04) | [Paper](https://arxiv.org/abs/2404.15155) | [Project](https://github.com/mitmedialab/MDAgents) |
| MMedAgent | MMedAgent: Learning to Use Medical Tools with Multi-modal Agent | Findings of EMNLP 2024 | 2024.11 (arXiv: 2024.07) | [Paper](https://arxiv.org/abs/2407.02483) | [Project](https://github.com/Wangyixinxin/MMedAgent) |
| MedAgent-Pro | MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow | ICLR 2026 | 2025.03 | [Paper](https://arxiv.org/abs/2503.18968) | [Project](https://github.com/jinlab-imvr/MedAgent-Pro) |
| DOLA | Autonomous Radiotherapy Treatment Planning Using DOLA: A Privacy-Preserving, LLM-Based Optimization Agent | arXiv | 2025.03 | [Paper](https://arxiv.org/abs/2503.17553) | - |
| GPT-Plan | A Feasibility Study of Automating Radiotherapy Planning with Large Language Model Agents | Physics in Medicine and Biology | 2025.03 | [Paper](https://doi.org/10.1088/1361-6560/adbff1) | - |
| VILA-M3 | VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge | CVPR25 | 2024.11 | [Paper](https://arxiv.org/abs/2411.12915) | [Project](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework) |
| CT-Agent | CT-Agent: A Multimodal-LLM Agent for 3D CT Radiology Question Answering | arXiv | 2025.05 | [Paper](https://arxiv.org/abs/2505.16229) | - |
| M^3Builder | M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging | AI for Clinical Applications 2025 | 2025.05 | [Paper](https://openreview.net/forum?id=CwZ7fYRZtv) | - |
| SAMIRA | Towards user-centered interactive medical image segmentation in VR with an assistive AI agent | arXiv | 2025.05 | [Paper](https://arxiv.org/abs/2505.07214) | - |
| MAM | MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration | Findings of ACL 2025 | 2025.07 (arXiv: 2025.06) | [Paper](https://arxiv.org/abs/2506.19835) | [Project](https://github.com/yczhou001/MAM) |
| AgentMRI | AgentMRI: A Vision Language Model-Powered AI System for Self-regulating MRI Reconstruction with Multiple Degradations | Journal of Imaging Informatics in Medicine | 2025.07 | [Paper](https://pubmed.ncbi.nlm.nih.gov/40696264/) | - |
| CTPA-Agent | Vision-language model for report generation and outcome prediction in CT pulmonary angiogram | npj Digital Medicine | 2025.07 | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12255762/) | [Project](https://github.com/zzs95/CTPA-Agent) |
| TissueLab | A co-evolving agentic AI system for medical imaging analysis | arXiv | 2025.09 | [Paper](https://arxiv.org/pdf/2509.20279) | [Project](https://github.com/zhihuanglab/TissueLab) |
| Scan-do Attitude | Scan-do Attitude: Towards Autonomous CT Protocol Management Using a Large Language Model Agent | Agentic AI for Medicine / Springer | 2025.09 | [Paper](https://arxiv.org/abs/2509.20270) | - |
| VoxelPrompt | VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis | arXiv | 2025.10 | [Paper](https://arxiv.org/abs/2410.08397) | - |
| MedAgentSim | MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions | MICCAI 2025 | 2025.10 (arXiv: 2025.03) | [Paper](https://arxiv.org/abs/2503.22678) | [Project](https://medagentsim.netlify.app/) |
| AURA | AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation | MICCAI Workshop 2025 | 2025.10 (arXiv: 2025.07) | [Paper](https://arxiv.org/abs/2507.16940) | [Project](https://nimafathi.github.io/AURA/) |
| MedEyes | MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.22018) | [Project](https://github.com/zhcz328/MedEyes) |
| MedSAM3 | MedSAM3: Delving into Segment Anything with Medical Concepts | arXiv | 2025.11 | [Paper](https://arxiv.org/abs/2511.19046) | [Project](https://github.com/Joey-S-Liu/MedSAM3) |
| Radiologist Copilot | Radiologist Copilot: An Agentic Framework Orchestrating Specialized Tools for Reliable Radiology Reporting | arXiv | 2025.12 | [Paper](https://arxiv.org/abs/2512.02814) | - |
| INFORM-CT | INFORM-CT: INtegrating LLMs and VLMs FOR Incidental Findings Management in Abdominal CT | MIDL 2026 | 2025.12 | [Paper](https://openreview.net/forum?id=ThdnZtQ6Dy) | [Project](https://anonymous.4open.science/r/InformCT_public-8A77/README.md) |
| IBISAgent | IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs for Universal Biomedical Object Referring and Segmentation | arXiv | 2026.01 | [Paper](https://arxiv.org/abs/2601.03054) | - |
| MedVistaGym | MEDVISTAGYM: A Scalable Training Environment for Thinking with Medical Images via Tool-Integrated Reinforcement Learning | arXiv | 2026.01 | [Paper](https://arxiv.org/abs/2601.07107) | - |
|  | An Explainable Agentic AI Framework for Uncertainty-Aware and Abstention-Enabled Acute Ischemic Stroke Imaging Decisions | arXiv | 2026.01 | [Paper](https://arxiv.org/abs/2601.01008) | - |
| LungNoduleAgent | LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules | AAAI 2026 | 2026.02 (arXiv: 2025.11) | [Paper](https://arxiv.org/abs/2511.21042) | [Project](https://github.com/ImYangC7/LungNoduleAgent) |
| 3DMedAgent | 3DMedAgent: Unified Perception-to-Understanding for 3D Medical Analysis | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.18064) | [Project](https://github.com/jinlab-imvr/3DMedAgent) |
| MedSAM-Agent | MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.03320) | [Project](https://github.com/CUHK-AIM-Group/MedSAM-Agent) |
| CARE | CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework | ICLR 2026 | 2026.05 (arXiv: 2026.03) | [Paper](https://arxiv.org/abs/2603.01607) | [Project](https://xypb.github.io/CARE-Project-Page/) |
| ToolSelect | Picking the Right Specialist: Attentive Neural Process-based Selection of Task-Specialized Models as Tools for Agentic Healthcare Systems | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.14901) | - |
| CoMMa | CoMMa: Contribution-Aware Medical Multi-Agents From A Game-Theoretic Perspective | arXiv | 2026.02 | [Paper](https://arxiv.org/abs/2602.09159) | - |
| MedSegAgent | MedSegAgent: A Universal and Scalable Multi-Agent System for Instructive Medical Image Segmentation | IEEE JBHI | 2026.03 | [Paper](https://ieeexplore.ieee.org/document/11455620) | [Project](https://github.com/uni-medical/MedSegAgent) |
| CT-Flow | CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.00123) | - |
| Agent-MIRA | Agent-MIRA: AI-orchestrated Medical Imaging Agent for PET Image Retrieval and Assistance | Computerized Medical Imaging and Graphics | 2026.03 | [Paper](https://doi.org/10.1016/j.compmedimag.2026.102725) | - |
| Meissa | Meissa: Multi-modal Medical Agentic Intelligence | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.09018) | [Project](https://github.com/Schuture/Meissa) |
| BT-RADS Agent | Agentic Automation of BT-RADS Scoring: End-to-End Multi-Agent System for Standardized Brain Tumor Follow-up Assessment | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.21494) | - |
| TheraAgent | TheraAgent: Multi-Agent Framework with Self-Evolving Memory and Evidence-Calibrated Reasoning for PET Theranostics | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.13676) | - |
| MedOpenClaw | MEDOPENCLAW: Auditable Medical Imaging Agents Reasoning over Uncurated Full Studies | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.24649) | - |
| MedMASLab | MedMASLab: A Unified Orchestration Framework for Benchmarking Multimodal Medical Multi-Agent Systems | arXiv | 2026.03 | [Paper](https://arxiv.org/pdf/2603.09909) | [Project](https://github.com/NUS-Project/MedMASLab/) |
| ClinicalAgents | ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.2618) | - |
| Doctorina MedBench | Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI | arXiv | 2026.03 | [Paper](https://arxiv.org/abs/2603.25821) | - |
| SEER | Skill-Evolving Grounded Reasoning for Free-Text Promptable 3D Medical Image Segmentation | arXiv | 2026.03 | [Paper](https://arxiv.org/pdf/2603.08215) |  |
| RadAgent | RadAgent: A Tool-Using AI Agent for Stepwise Interpretation of Chest Computed Tomography | arXiv | 2026.04 | [Paper](https://arxiv.org/pdf/2604.15231) | [Project](https://rad-agent.github.io/) |
| BAAI Cardiac Agent | BAAI Cardiac Agent: An intelligent multimodal agent for automated reasoning and diagnosis of cardiovascular diseases from cardiac magnetic resonance imaging | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.04078) | [Project](https://github.com/plantain-herb/Cardiac-Agent) |
| DosimeTron | DosimeTron: Automating Personalized Monte Carlo Radiation Dosimetry in PET/CT with Agentic AI | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.06280) | - |
| MARCH | MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation | ACL 2026 | 2026.04 | [Paper](https://arxiv.org/abs/2604.16175) | - |
| Neuro-Radiological Agent | Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.16729) | - |
| Agent4MR | Agentic MR sequence development: leveraging LLMs with MR skills for automatic physics-informed sequence development | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.13282) | - |
| Artifact-based Agent Framework | An Artifact-based Agent Framework for Adaptive and Reproducible Medical Image Processing | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.21936) | - |
| NeuroClaw | NeuroClaw: Closed-Loop Agentic AI for Executable and Reproducible Neuroimaging Research | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.24696) | - |
| Neuro-Oracle | Neuro-Oracle: A Trajectory-Aware Agentic RAG Framework for Interpretable Epilepsy Surgical Prognosis | arXiv | 2026.04 | [Paper](https://arxiv.org/abs/2604.14216) | - |
| MedScribe | MedScribe: Clinically Grounded CT Reporting through Agentic Workflows | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.01779) | - |
| GAZE | GAZE: Grounded Agentic Zero-shot Evaluation with Viewer-Level Tools and Literature Retrieval on Rare Brain MRI | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.00876) | - |
| NeuroAgent | NeuroAgent: LLM Agents for Multimodal Neuroimaging Analysis and Research | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.06584) | - |
| NEXUS | Towards a Virtual Neuroscientist: Autonomous Neuroimaging Analysis via Multi-Agent Collaboration | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2605.09366) | [Project](https://github.com/LearningKeqi/Virtual-Neuroscientist-NEXUS) |
| M2M-LLM-RT | A Machine-to-Machine Knowledge-Guided LLM Agent for Generalizable Radiotherapy Treatment Planning | arXiv | 2026.05 | [Paper](https://arxiv.org/abs/2606.00922) | - |
| SpineAgent | A Multi-Agent System for Spine MRI Report Generation from Multi-Sequence Imaging | arXiv | 2026.06 | [Paper](https://arxiv.org/abs/2606.08897) | - |
| MedToolica | MedToolica: Finetuning-Free Agentic Compositional Tool Learning for 3D CT Reasoning | Machine Learning and Knowledge Extraction | 2026.06 | [Paper](https://doi.org/10.3390/make8060162) | [Project](https://github.com/serag-ai/MedToolica) |
| MARTP | MARTP: A Multi-Agent Simulation Framework for Automated Radiation Therapy Planning Based on LLMs | Physics in Medicine and Biology | 2026.06 | [Paper](https://doi.org/10.1088/1361-6560/ae6af6) | - |
| SAGE | Automated Stereotactic Radiosurgery Planning Using a Human-in-the-Loop Reasoning Large Language Model Agent | Research Square | 2026.06 | [Paper](https://doi.org/10.21203/rs.3.rs-8612138/v1) | - |
| PET/CT Agent | End-to-End PET/CT Interpretation and Quantification with an LLM-Orchestrated AI Agent: A Real-World Pilot Study | Journal of Nuclear Medicine | 2026.06 | [Paper](https://doi.org/10.2967/jnumed.126.272362) | - |
| PD-CTAgent | Policy-Driven CT-Agent: Modeling Phase-Aware Diagnostic Control for Clinically Consistent CT Reasoning | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.10748) | - |
| MonteRET | MonteRET: AI Agent Enhancing Multimodal LLMs with Multi-granularity Knowledge Retrieval for Chest CT Report Generation | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.14264) | - |
| One-for-All | One-for-All Adaptive Radiotherapy Planning Agent: A Foundation Framework for Daily CBCT-guided Radiotherapy | arXiv | 2026.07 | [Paper](https://arxiv.org/abs/2607.14870) | - |

## **Datasets & Benchmarks**

| **Dataset** | **Title** | **Date** | **Venue** | **Paper Link** | **Project** |
| --- | --- | --- | --- | --- | --- |
| SLIVER07 | Segmentation in the liver 2007 (SLIVER07) challenge | 2007.09 | MICCAI Workshop | [Paper](https://ieeexplore.ieee.org/abstract/document/4781564) | [Project](https://sliver07.grand-challenge.org/) |
| SKI10 | Segmentation of Knee Images 2010 (SKI10) | 2010.09 | MICCAI Challenge | [Paper](https://www.researchgate.net/publication/266213062_Segmentation_of_Knee_Images_A_Grand_Challenge) | [Project](https://ski10.grand-challenge.org/Home/) |
| LIDC-IDRI | Data From LIDC-IDRI: The Lung Image Database Consortium and Image Database Resource Initiative | 2011.06 | Med Phys | [Paper](https://doi.org/10.1118/1.3528204) | [Project](https://www.cancerimagingarchive.net/collection/lidc-idri/) |
| LOLA11 | LOLA11: LObe and Lung Analysis 2011 Challenge | 2011.09 | MICCAI Workshop |  | [Project](https://lola11.grand-challenge.org/) |
| STACOM 2011 Motion Tracking | STACOM 2011: Cardiac Motion Tracking Challenge | 2011.09 | MICCAI Workshop | [Paper](https://link.springer.com/chapter/10.1007/978-3-642-28326-0_4) | [Project](https://www.cardiacatlas.org/motion-tracking-2011-challenge/) |
| Mindboggle-101 | Mindboggle-101: Evaluating Brain Image Labeling Methods | 2012.09 | NeuroImage | [Paper](https://www.sci-hub.ru/10.3389/fnins.2012.00171/full) | [Project](https://mindboggle.info/data.html) |
| PROMISE12 | PROMISE12: Prostate MR Image Segmentation 2012 Challenge | 2012.10 | MICCAI Challenge | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841513001734) | [Project](https://promise12.grand-challenge.org/) |
| NLST | The National Lung Screening Trial: overview and study design | 2013.01 | Radiology |  | [Project](https://www.cancerimagingarchive.net/collection/nlst/) |
| Farsiu Ophthalmology 2013 | Quantitative Classification of Eyes with and without Intermediate Age-related Macular Degeneration Using Optical Coherence Tomography | 2013.03 | Ophthalmology | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S016164201300612X) | [Project](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm) |
| Prostate-3T | Data From Prostate-3T | 2013.06 | TCIA Collection |  | [Project](https://www.cancerimagingarchive.net/collection/prostate-3t/) |
| MRBrainS13 | MRBrainS13: Grand Challenge on MR Brain Image Segmentation | 2013.09 | MICCAI Challenge |  | [Project](https://mrbrains13.isi.uu.nl/) |
| Chiu BOE 2014 | Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema | 2014.01 | Biomed Opt Express | [Paper](https://opg.optica.org/boe/fulltext.cfm?uri=boe-6-4-1172) | [Project](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm) |
| Srinivasan BOE 2014 | Fully automated detection of diabetic macular edema and dry age-related macular degeneration from optical coherence tomography images | 2014.03 | Biomed Opt Express | [Paper](https://opg.optica.org/boe/fulltext.cfm?uri=boe-5-10-3568) | [Project](https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm) |
| orCaScore | An evaluation of automatic coronary artery calcium scoring methods with cardiac CT using the orCaScore framework | 2014.09 | MICCAI Challenge | [Paper](https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.4945696) | [Project](https://orcascore.grand-challenge.org/) |
| CETUS2014 | CETUS: Cardiac Echocardiography Tracking and Segmentation Challenge | 2014.09 | MICCAI Challenge |  | [Project](https://www.creatis.insa-lyon.fr/Challenge/CETUS/index.html) |
| Prostate-Diagnosis | PROSTATE-DIAGNOSIS: Multiparametric MRI for Prostate Cancer | 2015.03 | TCIA Collection |  | [Project](https://www.cancerimagingarchive.net/collection/prostate-diagnosis/) |
| BTCV | MICCAI multi-atlas labeling beyond the cranial vault-workshop and challenge | 2015.04 | MICCAI Workshop |  | [Project](https://www.synapse.org/Synapse:syn3193805/challenge/) |
| ISMRM2015 HARDI | ISMRM 2015 Tractography Challenge | 2015.06 | ISMRM Challenge |  | [Project](https://tractometer.org/ismrm2015/dwi_data) |
| NEATBrainS15 | NEATBrainS15: Neonatal Brain Structure Segmentation | 2015.09 | MICCAI Challenge |  | [Project](https://www.isi.uu.nl/research/challenges/neatbrains/) |
| PDDCA | Public Domain Database for Computational Anatomy: Head and Neck | 2015.09 | MICCAI Challenge |  | [Project](https://www.imagenglab.com/newsite/pddca/) |
| HVSMR 2016 | HVSMR 2016: Whole Heart and Great Vessel Segmentation Challenge | 2016.07 | MICCAI Challenge | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_10#page-1) | [Project](http://segchd.csail.mit.edu/data.html) |
| MSSEG 2016 | Objective Evaluation of Multiple Sclerosis Lesion Segmentation using a Data Management and Processing Infrastructure | 2016.10 | MICCAI Challenge/Nature | [Paper](https://www.nature.com/articles/s41598-018-31911-7) | [Project](https://portal.fli-iam.irisa.fr/msseg-challenge/) |
| PROSTATEx | PROSTATEx: PROSTATE MR Image Dataset With Prostate Cancer Annotations | 2016.10 | SPIE-AAPM-NCI PROSTATEx Challenge |  | [Project](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=163875860) |
| WMH | WMH Segmentation Challenge: White Matter Hyperintensity Segmentation in Brain MR | 2017.03 | MICCAI Challenge | [Paper](https://ieeexplore.ieee.org/document/8669968) | [Project](https://wmh.isi.uu.nl/) |
| LGG-1p19qDeletion | LGG-1p19qDeletion: Low-Grade Glioma MRI with Genomic Annotations | 2017.03 | TCIA Collection |  | [Project](https://www.cancerimagingarchive.net/collection/lgg-1p19qdeletion/) |
| PROSTATEx-2 | PROSTATEx-2: Lesion Classification Challenge | 2017.06 | AAPM Grand Challenge |  | [Project](https://www.aapm.org/GrandChallenge/PROSTATEx-2/) |
| ACDC | Automatic Cardiac Diagnosis Challenge | 2017.09 | STACOM / MICCAI | [Paper](https://ieeexplore.ieee.org/document/8360453) | [Project](https://www.creatis.insa-lyon.fr/Challenge/acdc/) |
| RETOUCH | RETOUCH: Retinal OCT Fluid Segmentation Challenge | 2017.09 | MICCAI Challenge | [Paper](https://doi.org/10.1016/j.media.2019.101570) | [Project](https://retouch.grand-challenge.org/Home/) |
| ROCC | ROCC: Retinal OCT Classification Challenge | 2017.09 | MICCAI Challenge |  | [Project](https://rocc.grand-challenge.org/) |
| iSeg2017 | iSeg-2017: Infant Brain MRI Segmentation Challenge | 2017.09 | MICCAI Challenge | [Paper](https://iseg2017.web.unc.edu/wp-content/uploads/sites/14056/2019/02/Benchmark-on-Automatic-6-month-old-Infant-Brain-Segmentation-Algorithms.pdf) | [Project](https://iseg2017.web.unc.edu/) |
| DeepLesion | DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection in CT | 2017.10 | JMI / arXiv | [Paper](https://arxiv.org/abs/1710.01766) | [Project](https://nihcc.app.box.com/v/DeepLesion) |
| LUNA 16 | Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA 16 challenge | 2017.12 | Medical Image Analysis | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841517301020) | [Project](https://luna16.grand-challenge.org/) |
| Mandibular-CT-Dataset | Mandibular CT Dataset Collection for 3D Reconstruction and Segmentation | 2018.03 | figshare | [Paper](https://arxiv.org/pdf/1902.05255) | [Project](https://figshare.com/articles/dataset/Mandibular_CT_Dataset_Collection/6167726) |
| FUMPE | Computer-aided detection of pulmonary embolism in CT | 2018.03 | arXiv / Kaggle |  | [Project](https://www.kaggle.com/andrewmvd/pulmonary-embolism-in-ct-images) |
| ISLES 2018 | ISLES 2018 – Ischemic Stroke Lesion Segmentation | 2018.09 | MICCAI Challenge | [Paper](https://www.sci-hub.ru/10.1177/0271678X15610586) | [Project](http://www.isles-challenge.org/ISLES2018/) |
| MRBrainS18 | MRBrainS18: MR Brain Segmentation Challenge 2018 | 2018.09 | MICCAI Challenge |  | [Project](https://mrbrains18.isi.uu.nl/data/) |
| Atrial Segmentation Challenge | 2018 Atrial Segmentation Challenge | 2018.09 | MICCAI Challenge |  | [Project](https://www.cardiacatlas.org/atriaseg2018-challenge/) |
| IVDM3Seg | IVDM3Seg: Intervertebral Disc and Vertebrae Segmentation Challenge | 2018.09 | MICCAI Challenge |  | [Project](https://ivdm3seg.weebly.com/) |
| MRNet | MRNet: Knee MRI Dataset for Abnormality Detection | 2018.09 | NIPS Workshop |  | [Project](https://stanfordmlgroup.github.io/competitions/mrnet/) |
| OCT Glaucoma Detection | Glaucoma Detection in 3D Spectral-Domain OCT | 2018.10 | Sci Rep |  | [Project](https://doi.org/10.5281/zenodo.1481223) |
| BraTS | Brain Tumor Segmentation (BraTS) Challenge | 2018.11 | MICCAI Challenge (series) | [Paper](https://arxiv.org/abs/1811.02629) | [Project](https://www.med.upenn.edu/cbica/brats/) |
| fastMRI | fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee and Brain MR Images | 2018.11 | MRM | [Paper](https://arxiv.org/abs/1811.08839) | [Project](https://fastmri.med.nyu.edu/) |
| LiTS | Liver Tumor Segmentation (LiTS) Challenge | 2019.01 | MICCAI Challenge | [Paper](https://arxiv.org/abs/1901.04056) | [Project](https://competitions.codalab.org/competitions/17094) |
| OASIS-3 | OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease | 2019.01 | Sci Data | [Paper](https://www.medrxiv.org/content/10.1101/2019.12.13.19014902v1) | [Project](https://www.oasis-brains.org/) |
| MM-WHS | MM-WHS: Multi-Modality Whole Heart Segmentation | 2019.02 | MICCAI Challenge | [Paper](https://arxiv.org/abs/1902.07880) | [Project](https://zmiclab.github.io/zxh/0/mmwhs/) |
| CHAOS CT-MRI | CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation | 2019.02 | ISBI Challenge | [Paper](https://chaos.grand-challenge.org/Publications/) | [Project](https://chaos.grand-challenge.org/) |
| KiTS19 | KiTS19: Kidney Tumor Segmentation Challenge | 2019.04 | MICCAI Challenge | [Paper](https://arxiv.org/abs/1904.00445) | [Project](https://kits19.grand-challenge.org/) |
| AAPM-RT-MAC | AAPM RT-MAC: MR-only based Radiotherapy in Head and Neck | 2019.07 | AAPM Challenge | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7322982/) | [Project](https://www.cancerimagingarchive.net/collection/aapm-rt-mac/) |
| iSeg-2019 | iSeg-2019: Infant Brain MRI Segmentation Challenge | 2019.09 | MICCAI Challenge | [Paper](https://iseg2019.web.unc.edu/) | [Project](https://iseg2019.web.unc.edu/) |
| SegTHOR | SegTHOR: Segmentation of thoracic organs at risk in CT images | 2019.09 | Physica Medica | [Paper](https://arxiv.org/abs/1912.05950) | [Project](https://competitions.codalab.org/competitions/21145) |
| VerSe20 | VerSe 2020: Vertebral Segmentation Challenge at MICCAI | 2020.01 | MICCAI Challenge |  | [Project](https://verse2020.grand-challenge.org/) |
| VerSe19 | VerSe 2019: Vertebral Segmentation Challenge at MICCAI | 2020.01 | MICCAI Challenge | [Paper](https://arxiv.org/abs/2001.09193) | [Project](https://verse2019.grand-challenge.org/) |
| COVID-19-CT-Seg | COVID-19 CT lung and infection segmentation dataset | 2020.04 | zenodo |  | [Project](https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark) |
| M&Ms | M&Ms: Multi-Centre, Multi-Vendor & Multi-Disease Cardiac MR Segmentation Challenge | 2020.05 | MICCAI Challenge |  | [Project](https://www.ub.edu/mnms/) |
| CTPelvic1K | CTPelvic1K: A Large-Scale Pelvic CT Dataset for Multi-Task Parsing | 2020.06 | arXiv | [Paper](https://pubmed.ncbi.nlm.nih.gov/33864189/) | [Project](https://github.com/MIRACLE-Center/CTPelvic1K) |
| Prostate MR Segmentation Dataset (SAML) | Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space | 2020.09 | MICCAI |  | [Project](https://liuquande.github.io/SAML/) |
| EMIDEC | EMIDEC 2020: Myocardial Infarction Detection, Segmentation and Classification | 2020.09 | MICCAI Challenge | [Paper](https://emidec.com/) | [Project](https://emidec.com/) |
| KNOAP2020 | KNOAP2020: Knee Osteoarthritis Progression Prediction Challenge | 2020.09 | MICCAI Challenge |  | [Project](https://knoap2020.grand-challenge.org/Home/) |
| Learn2Reg Lung CT | Learn2Reg 2020: Lung CT Registration | 2020.09 | MICCAI Challenge |  | [Project](https://learn2reg.grand-challenge.org/) |
| Learn2Reg Abdomen CT-CT | Learn2Reg 2020: Abdominal CT-CT Registration | 2020.09 | MICCAI Challenge |  | [Project](https://learn2reg.grand-challenge.org/) |
| RibFrac2020 | RibFrac: Rib Fracture Detection and Classification Challenge | 2020.10 | MICCAI Challenge |  | [Project](https://ribfrac.grand-challenge.org/) |
| HECKTOR 2020 | HECKTOR 2020: Segmentation of Head and Neck Tumor in PET/CT | 2020.11 | MICCAI Challenge |  | [Project](https://www.aicrowd.com/challenges/miccai-2020-hecktor) |
| CT-ORG | CT-ORG, a new dataset for multiple organ segmentation in computed tomography | 2020.11 | Nature | [Paper](https://www.nature.com/articles/s41597-020-00715-8) | [Project](https://springernature.figshare.com/articles/dataset/Metadata_record_for_CT-ORG_a_dataset_for_multiple_organ_segmentation_in_computed_tomography/13055663) |
| RAD-ChestCT | Machine-Learning-Based Multiple Abnormality Prediction with Large-Scale Chest Computed Tomography Volumes | 2021.01 | Medical Image Analysis | [Paper](https://arxiv.org/abs/2002.04752) | [Project](https://zenodo.org/records/6406114) |
| Eye OCT Datasets (3D) | 3D Retinal OCT Classification and Segmentation Dataset | 2021.01 | Tianchi |  | [Project](https://tianchi.aliyun.com/dataset/dataDetail?dataId=90672) |
| HECKTOR 2021 | HECKTOR 2021: Head and Neck Tumor Segmentation and Outcome Prediction | 2021.05 | MICCAI Challenge |  | [Project](https://www.aicrowd.com/challenges/miccai-2021-hecktor) |
| CTSpine1K | CTSpine1K: A Large-Scale Dataset for Spine Parsing in CT | 2021.07 | arXiv |  | [Project](https://github.com/MIRACLE-Center/CTSpine1K) |
| MSSEG-2 | MSSEG-2 challenge: Multiple Sclerosis Lesion Segmentation at 7T and 3T MRI | 2021.07 | NeuroImage Clin |  | [Project](https://portal.fli-iam.irisa.fr/msseg-2/data/) |
| FLARE21 | FLARE 2021: A Challenge on Abdominal Multi-organ Segmentation | 2021.09 | MICCAI Challenge |  | [Project](https://flare.grand-challenge.org/FLARE21/) |
| QUBIQ2021 3D CT | QUBIQ 2021: Quantification of Uncertainty in Biomedical Image Quantification | 2021.09 | MICCAI Challenge |  | [Project](https://qubiq21.grand-challenge.org/QUBIQ2021/) |
| M&Ms-2 | M&Ms-2: Multi-Domain Cardiac MR Segmentation | 2021.09 | MICCAI Challenge |  | [Project](https://www.ub.edu/mnms-2/) |
| Learn2Reg Abdomen MR-CT | Learn2Reg 2021: Abdominal MR-CT Multi-Modal Registration | 2021.09 | MICCAI Challenge |  | [Project](https://learn2reg.grand-challenge.org/) |
| CrossMoDA2021 | CrossMoDA 2021: Unsupervised Domain Adaptation for Cross-Modality Vestibular Schwannoma Segmentation | 2021.09 | MICCAI Challenge |  | [Project](https://crossmoda.grand-challenge.org/CrossMoDA/) |
| WORD | WORD: A Whole-Organ CT Dataset for Robust Multi-Organ Segmentation | 2021.10 | arXiv | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841522002705) | [Project](https://github.com/HiLab-git/WORD) |
| MedMNIST v2 | MedMNIST v2 -- A large-scale lightweight benchmark for 2D and 3D biomedical image classification | 2021.10 | Nature | [Paper](https://arxiv.org/abs/2110.14795) | [Project](https://medmnist.com/) |
| CADA | CADA: Cerebral Aneurysm Detection and Analysis Challenge | 2022.04 | MICCAI Challenge |  | [Project](https://cada.grand-challenge.org/) |
| CADA-AS | CADA-AS: Aneurysm Segmentation Challenge | 2022.04 | MICCAI Challenge |  | [Project](https://cada-as.grand-challenge.org/) |
| CADA-RRE | CADA-RRE: Rupture Risk Estimation for Cerebral Aneurysms | 2022.04 | MICCAI Challenge |  | [Project](https://cada-rre.grand-challenge.org/) |
| TotalSegmentator | TotalSegmentator: robust segmentation of 104 anatomic structures in CT images | 2022.06 | arXiv | [Paper](https://arxiv.org/abs/2208.05868) | [Project](https://totalsegmentator.com/) |
| AMOS | AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation | 2022.06 | NeurIPS 2022 | [Paper](https://arxiv.org/abs/2206.08023) | [Project](https://amos22.grand-challenge.org/) |
| MSD | The medical segmentation decathlon | 2022.07 | Nature Communications | [Paper](https://doi.org/10.1038/s41467-022-30695-9) | [Project](http://medicaldecathlon.com/) |
| AutoPET | The AutoPET Challenge: Automated Lesion Segmentation in Whole-Body FDG-PET/CT | 2022.07 | MICCAI Challenge (autoPET2022) |  | [Project](https://autopet.grand-challenge.org/) |
| UPENN-GBM | UPENN-GBM: Multi-modal MRI Dataset for Glioblastoma Segmentation | 2022.07 | TCIA Collection |  | [Project](https://www.cancerimagingarchive.net/collection/upenn-gbm/) |
| KiPA22 | KiPA22: Kidney PArametric segmentation in contrast-enhanced CT | 2022.08 | MICCAI Challenge |  | [Project](https://kipa22.grand-challenge.org/) |
| OLIVES | OLIVES: A 3D OCT Dataset for Longitudinal Retinal Imaging | 2022.09 | arXiv |  | [Project](https://github.com/olivesgatech/OLIVES_Dataset) |
| PI-CAI | PI-CAI: Prostate Imaging–Cancer AI Challenge | 2022.09 | MICCAI Challenge |  | [Project](https://pi-cai.grand-challenge.org/PI-CAI/) |
| LAScarQS 2022 | LAScarQS 2022: Left Atrial Scar Quantification and Segmentation Challenge | 2022.09 | MICCAI Challenge |  | [Project](https://zmiclab.github.io/projects/lascarqs22/) |
| CrossMoDA2022 | CrossMoDA 2022: Domain Adaptation for Vestibular Schwannoma Segmentation and Koos Grading | 2022.09 | MICCAI Challenge |  | [Project](https://crossmoda2022.grand-challenge.org/crossmoda2022/) |
| FeTA 2022 | FeTA 2022: Fetal Brain Tissue Segmentation at MICCAI | 2022.09 | MICCAI Challenge |  | [Project](https://feta.grand-challenge.org/feta-2022/) |
| COSMOS 2022 | COSMOS 2022: Carotid Artery Vessel Wall Segmentation | 2022.09 | MICCAI Challenge |  | [Project](https://vessel-wall-segmentation-2022.grand-challenge.org/) |
| cSeg-2022 | cSeg 2022: Cerebellum Segmentation Challenge | 2022.09 | MICCAI Challenge |  | [Project](https://tarheels.live/cseg2022/) |
| ISLES 2022 | ISLES 2022 – Acute and Subacute Ischemic Stroke Lesion Segmentation | 2022.09 | MICCAI Challenge |  | [Project](https://isles22.grand-challenge.org/) |
| InSTANCE2022 | InSTANCE 2022: Intracranial Hemorrhage Segmentation Challenge | 2022.09 | MICCAI Challenge | [Paper](https://arxiv.org/abs/2301.03281) | [Project](https://instance.grand-challenge.org/Instance2022/) |
| Learn2Reg NLST | Learn2Reg 2022: Thoracic CT Registration with NLST | 2022.09 | MICCAI Challenge |  | [Project](https://learn2reg.grand-challenge.org/) |
| Shifts Challenge 2022 | Shifts 2022: Distribution Shifts in Multiple Sclerosis Lesion Segmentation | 2022.09 | MICCAI Challenge | [Paper](https://shifts.grand-challenge.org/) | [Project](https://shifts.grand-challenge.org/) |
| HECKTOR 22 | Overview of the HECKTOR challenge at MICCAI 2022: automatic head and neck tumor segmentation and outcome prediction in PET/CT | 2023 | MICCAI 2022 | [Paper](https://pubmed.ncbi.nlm.nih.gov/37195050/) | [Project](https://hecktor.grand-challenge.org/) |
| LNDb | LNDb challenge on automatic lung cancer patient management | 2023.03 | Medical Image Analysis | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-27420-6_1) | [Project](https://lndb.grand-challenge.org/) |
| Semi-TeethSeg | Semi-TeethSeg: Semi-Supervised 3D Tooth Segmentation in CBCT/CT | 2023.04 | arXiv |  | [Project](https://zenodo.org/record/7840021) |
| PARSE22 | Efficient automatic segmentation for multi-level pulmonary arteries: The parse challenge | 2023.04 | arXiv | [Paper](https://arxiv.org/abs/2304.03708) | [Project](https://parse2022.grand-challenge.org/Parse2022/) |
| STAGE | STAGE: Longitudinal OCT Dataset for Glaucoma Progression | 2023.04 | Dataset |  | [Project](https://doi.org/10.5281/zenodo.7835341) |
| KiTS21 | The kits21 challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase ct | 2023.07 | arXiv | [Paper](https://arxiv.org/abs/2307.01984) | [Project](https://kits21.kits-challenge.org/) |
| AutoPET II | AutoPET-II: Ensemble-based Uncertainty-Aware Lesion Segmentation in Multi-Center FDG-PET/CT | 2023.07 | MICCAI Challenge (AutoPET-II) |  | [Project](https://autopet-ii.grand-challenge.org/) |
| MedMD | Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data | 2023.08 | arXiv | [Paper](https://arxiv.org/abs/2308.02463) | [Project](https://radiopaedia.org/) |
| ULS23 | ULS23 Challenge: Universal Lesion Segmentation in CT for Oncological Imaging | 2023.08 | MICCAI Challenge |  | [Project](https://uls23.grand-challenge.org/uls23/) |
| SegRap2023 | SegRap 2023: Nasopharyngeal Carcinoma Radiotherapy Segmentation Challenge | 2023.08 | MICCAI Challenge |  | [Project](https://segrap2023.grand-challenge.org/) |
| LNQ2023 | LNQ2023: Lymph Node Quantification in Chest CT | 2023.08 | MICCAI Challenge |  | [Project](https://lnq2023.grand-challenge.org/) |
| FLARE23 | FLARE 2023: A Federated Learning Challenge for Abdominal Multi-Organ Segmentation | 2023.09 | MICCAI Challenge |  | [Project](https://codalab.lisn.upsaclay.fr/competitions/12239) |
| CrossMoDA2023 | CrossMoDA 2023: Multi-Center Domain Adaptation for VS Segmentation | 2023.09 | MICCAI Challenge |  | [Project](https://crossmoda-challenge.ml/) |
| ATLAS2023 | ATLAS 2023: Liver Tumor Segmentation Challenge | 2023.09 | MICCAI Challenge | [Paper](https://www.mdpi.com/2306-5729/8/5/79) | [Project](https://atlas-challenge.u-bourgogne.fr/) |
| SMILE-UHURA2023 | SMILE-UHURA 2023: Small Vessel Disease Lesion Segmentation | 2023.09 | MICCAI Challenge | [Paper](https://arxiv.org/abs/2411.09593) | [Project](https://www.synapse.org/#!Synapse:syn47164761/wiki/620033) |
| CAS2023 | CAS 2023: Brain Structure Segmentation Benchmark | 2023.09 | MICCAI Challenge |  | [Project](https://codalab.lisn.upsaclay.fr/competitions/9804) |
| CROWN2023 | CROWN 2023: White Matter Hyperintensity and Other Pathology Classification | 2023.09 | MICCAI Challenge |  | [Project](https://crown.isi.uu.nl/) |
| SLCN | SLCN: Structural Lesion and Connectivity in Neurodevelopmental Disorders | 2023.09 | MICCAI Challenge |  | [Project](https://slcn.grand-challenge.org/) |
| ToothFairy2023 | ToothFairy: 3D CBCT Dataset for Inferior Alveolar Nerve Segmentation | 2023.09 | MICCAI Challenge |  | [Project](https://toothfairy.grand-challenge.org/toothfairy/) |
| XPRESS2023 | XPRESS 2023: X-ray Phase-Contrast CT Neuroanatomy Segmentation | 2023.09 | MICCAI Challenge | [Paper](https://arxiv.org/abs/2302.03819) | [Project](https://xpress.grand-challenge.org/) |
| Learn2Reg ThoraxCBCT | Learn2Reg 2023: Thorax CBCT/FBCT Deformable Registration | 2023.09 | MICCAI Challenge |  | [Project](https://learn2reg.grand-challenge.org/) |
| TDSC-ABUS2023 | TDSC-ABUS2023: Automated Breast Ultrasound Segmentation Challenge | 2023.09 | MICCAI Challenge | [Paper](https://arxiv.org/abs/2501.15588) | [Project](https://tdsc-abus2023.grand-challenge.org/) |
| MVSeg-3DTEE2023 | MVSeg-3DTEE2023: Mitral Valve Segmentation from 3D TEE | 2023.09 | MICCAI Challenge |  | [Project](https://www.synapse.org/#!Synapse:syn51186045/wiki/621356) |
| RegPro2023 | RegPro 2023: Prostate MR-US Registration Challenge | 2023.09 | MICCAI Challenge |  | [Project](https://muregpro.github.io/) |
| KiTS23 | KiTS23: Kidney and Kidney Tumor Segmentation with Comprehensive Clinical Annotations | 2023.10 | arXiv |  | [Project](https://kits-challenge.org/kits23/) |
| WBMR-NF | WBMR-NF: Whole-Body MRI for Neurofibromatosis | 2023.11 | Dataset |  | [Project](https://mgh-3dqi.github.io/) |
| GAMMA | GAMMA Challenge: Glaucoma Assessment with Multi-Modality Data | 2023.12 | MICCAI Challenge | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841523001986) | [Project](https://gamma.grand-challenge.org/) |
| ATM'22 | ATM'22: Airway Tree Modeling in Thoracic CT | 2023.12 | MICCAI Challenge | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841523002177) | [Project](https://atm22.grand-challenge.org/) |
| INSPECT | INSPECT: A Multimodal Dataset for Pulmonary Embolism Diagnosis and Prognosis | 2023.12 | NeurIPS 2023 | [Paper](https://arxiv.org/abs/2311.10798) | [Project](http://www.isles-challenge.org/) |
| HaN-Seg | HaN-Seg 2023: Head and Neck Organ at Risk Segmentation Challenge | 2024 | MICCAI Challenge | [Paper](https://www.sciencedirect.com/science/article/pii/S0167814024006807) | [Project](https://han-seg2023.grand-challenge.org/) |
| VALDO | Where is VALDO? Vascular Lesions Detection and Segmentation Challenge | 2024.01 | MICCAI Challenge | [Paper](https://www.sciencedirect.com/science/article/pii/S136184152300289X) | [Project](https://valdo.grand-challenge.org/) |
| BIMCV-R | BIMCV-R: Large-Scale Thoracic CT Reconstruction Benchmark | 2024.01 | MICCAI24 | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72120-5_12) | [Project](https://huggingface.co/datasets/cyd0806/BIMCV-R) |
| IXI | Information eXtraction from Images (IXI) Dataset | 2024.01 | Dataset |  | [Project](http://brain-development.org/ixi-dataset/) |
| RAOS | RAOS: A Large-Scale Radiotherapy Abdominal Organ Segmentation Dataset | 2024.01 | MICCAI24 | [Paper](https://arxiv.org/abs/2406.13674) | [Project](https://github.com/Luoxd1996/RAOS) |
| ISLES 2024 | Ischemic Stroke Lesion Segmentation Challenge 2024 (ISLES 2024) | 2024.02 | MICCAI Challenge |  | [Project](https://www.isles-challenge.org/) |
| AbdomenAtlas | AbdomenAtlas-20K: A Large-Scale Benchmark for Abdominal Multi-Organ Segmentation in CT | 2024.02 | arXiv | [Paper](https://arxiv.org/abs/2407.16697) | [Project](https://www.zongweiz.com/dataset) |
| OpenMind | OpenMind: Large-Scale Head-and-Neck MR Dataset for Foundation Models | 2024.02 | arXiv |  | [Project](https://huggingface.co/datasets/AnonRes/OpenMind) |
| TriALS2024 | TriALS 2024: Liver Tumor Segmentation and Outcome Prediction – Task 1 | 2024.03 | MICCAI24 |  | [Project](https://github.com/xmed-lab/TriALS) |
| LAScarQS++ 2024 | LAScarQS++ 2024: Multi-Center Atrial Scar Segmentation | 2024.03 | CARE Workshop (MICCAI) |  | [Project](https://www.zmic.org.cn/care_2024/track2/) |
| MyoPS | MyoPS: A Benchmark of Myocardial Pathology Segmentation Combining Three-Sequence Cardiac Magnetic Resonance Images# MyoPS: A Benchmark of Myocardial Pathology Segmentation Combining Three-Sequence Cardiac Magnetic Resonance Images | 2024.03 | CARE Workshop | [Paper](https://arxiv.org/abs/2201.03186) | [Project](https://www.zmic.org.cn/care_2024/track4/) |
| WHS++ 2024 | WHS++ 2024: Multi-Center Whole Heart Segmentation | 2024.03 | CARE Workshop |  | [Project](https://www.zmic.org.cn/care_2024/track5/) |
| AMOS-MM | AMOS-MM: Multi-Phase Abdominal CT Benchmark for Translation and Synthesis | 2024.03 | arXiv | [Paper](https://doi.org/10.5281/zenodo.10992155) | [Project](https://zenodo.org/records/10992155) |
| CT2Rep | CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging | 2024.03 | MICCAI 2024 | [Paper](https://arxiv.org/abs/2403.06801) | [Project](https://github.com/ibrahimethemhamamci/CT2Rep) |
| TotalSegmentator MRI | TotalSegmentator MRI: Whole-body MRI Segmentation of 150 Structures | 2024.04 | arXiv | [Paper](https://arxiv.org/abs/2405.19492) | [Project](https://zenodo.org/doi/10.5281/zenodo.11367004) |
| RadGenome-ChestCT | RadGenome-Chest CT: a grounded vision-language dataset for chest CT analysis | 2024.04 | arXiv | [Paper](https://doi.org/10.48550/arXiv.2404.16754) | [Project](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |
| M3D | M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models | 2024.04 | ICLR 2025 | [Paper](https://arxiv.org/abs/2404.00578) | [Project](https://github.com/BAAI-DCAI/M3D) |
| AIIB23 | AIIB23: Airway Inflammation Imaging Biomarkers Challenge | 2024.06 | MICCAI Challenge | [Paper](https://doi.org/10.1016/j.media.2024.103253) | [Project](https://codalab.lisn.upsaclay.fr/competitions/13238) |
| CT-3DRRG | Argus: Benchmarking and Enhancing Vision-Language Models for 3D Radiology Report Generation | 2024.06 | arXiv | [Paper](https://arxiv.org/abs/2406.07146) | — |
| RadGenome-Brain MRI | AutoRG-Brain: Grounded Report Generation for Brain MRI | 2024.10 | MICCAI 2024 | [Paper](https://arxiv.org/abs/2407.16684) | [Project](https://github.com/ljy19970415/AutoRG-Brain) |
| MedShapeNet | MedShapeNet -- A Large-Scale Dataset of 3D Medical Shapes for Computer Vision | 2024.12 | Biomedizinische Technik | [Paper](https://arxiv.org/abs/2308.16139) | [Project](https://medshapenet.ikim.nrw/) |
| RadA-BenchPlat | How well can modern LLMs act as agent cores in radiology environments? | 2024.12 | arXiv | [Paper](https://arxiv.org/abs/2412.09529) |  |
| MedVL-CT69K | Large-scale and Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding | 2025.01 | ICLR 2025 | [Paper](https://arxiv.org/abs/2501.14548) | [Project](https://github.com/alibaba-damo-academy/fvlm) |
| Triad | Triad: Vision Foundation Model for 3D Magnetic Resonance Imaging | 2025.02 | arXiv | [Paper](https://arxiv.org/abs/2502.14064) |  |
| 3D-BrainCT | Towards a holistic framework for multimodal LLM in 3D brain CT radiology report generation | 2025.03 | Nat. Commun. | [Paper](https://www.nature.com/articles/s41467-025-57426-0) | — |
| PENGWIN2024-Task1 | PENGWIN 2024: Pelvic Fracture Segmentation in Trauma CT | 2025.04 | MICCAI Challenge | [Paper](https://arxiv.org/abs/2504.02382) | [Project](https://pengwin.grand-challenge.org/) |
| RibFrac | Deep rib fracture instance segmentation and classification from ct on the ribfrac challenge | 2025.04 | IEEE | [Paper](https://arxiv.org/abs/2402.09372) | [Project](https://ribfrac.grand-challenge.org/) |
| DeepTumorVQA | Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering | 2025.05 | NeurIPS 2025 | [Paper](https://arxiv.org/abs/2505.18915) | [Project](https://github.com/Schuture/DeepTumorVQA) |
| NOVA | NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI | 2025.05 | NeurIPS 2025 | [Paper](https://arxiv.org/abs/2505.14064) |  |
| Lingshu | Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning | 2025.06 | arXiv | [Paper](https://arxiv.org/abs/2506.07044) | [Project](https://alibaba-damo-academy.github.io/lingshu/) |
| ReXGroundingCT | ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports | 2025.07 | arXiv | [Paper](https://arxiv.org/abs/2507.22030) | [Project](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT) |
| ViPET-ReportGen | Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation | 2025.12 | NeurIPS 2025 | [Paper](https://arxiv.org/abs/2509.24739) | [Project](https://github.com/AIoT-Lab-AI4LIFE/ViPET-ReportGen) |
| 3D-RAD | 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks | 2025.12 | NeurIPS 2025 | [Paper](https://arxiv.org/abs/2506.11147) | [Project](https://github.com/Tang-xiaoxiao/3D-RAD) |
| MR-RATE | MR-RATE: A Vision-Language Foundation Model and Dataset for Magnetic Resonance Imaging | 2026 | — | — | [Project](https://github.com/forithmus/MR-RATE) |
| CT-RATE | Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography | 2026.02 | Nature | [Paper](https://arxiv.org/abs/2403.17834) | [Project](https://github.com/ibrahimethemhamamci/CT-CLIP) |
| CT-FlowBench | CT-FlowBench: Benchmark for CT interpretation workflow and tool-use | 2026.03 | arXiv | [Paper](https://arxiv.org/abs/2603.00123) |  |
| Merlin | Merlin: A Vision Language Foundation Model for 3D Computed Tomography | 2026.03 | Nature | [Paper](https://www.nature.com/articles/s41586-026-10181-8) | [Project](https://github.com/StanfordMIMI/Merlin) |
| Gastric-X | Gastric-X: A Multimodal Multi-Phase Benchmark Dataset for Advancing Vision-Language Models in Gastric Cancer Analysis | 2026.03 | CVIPPR 2026 | [Paper](https://arxiv.org/abs/2603.19516) | — |
| SpatialMed | Beyond Medical Diagnostics: How Medical Multimodal Large Language Models Think in Space | 2026.03 | arXiv | [Paper](https://arxiv.org/abs/2603.13800) |  |
| BONBID-HIE2023 | BONBID-HIE 2023: Neonatal Hypoxic-Ischemic Encephalopathy Lesion Segmentation | 2026.04 | MICCAI Challenge | [Paper](https://pubmed.ncbi.nlm.nih.gov/41379890/) | [Project](https://bonbid-hie2023.grand-challenge.org/bonbid-hie2023/) |
| SGMRI-VQA | Beyond a Single Frame: Multi-Frame Spatially Grounded Reasoning Across Volumetric MRI | 2026.04 | arXiv | [Paper](https://arxiv.org/abs/2604.15808) |  |
| Curia-2 | Curia-2: Scaling Self-Supervised Learning for Radiology Foundation Models | 2026.04 | arXiv | [Paper](https://arxiv.org/abs/2604.01987) |  |
| CT-SpatialVQA | Lost in Volume: The CT-SpatialVQA Benchmark for Evaluating Semantic-Spatial Understanding of 3D Medical Vision-Language Models | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.08787) |  |
| Med-StepBench | Med-StepBench: A Hierarchical Reasoning Framework for Evaluating Hallucinations in Medical Vision-Language Models | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.10002) |  |
| DeepTumorVQA-H | DeepTumorVQA: A Hierarchical 3D CT Benchmark for Stage-Wise Evaluation of Medical VLMs and Tool-Augmented Agents | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.09679) |  |
| ABRA | ABRA: Agent Benchmark for Radiology Applications | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.11224) |  |
| RadSaFE-200 | Safety and Accuracy Follow Different Scaling Laws in Clinical Large Language Models | 2026.05 | arXiv | [Paper](https://arxiv.org/abs/2605.04039) |  |
| Oncology VQA Benchmark | Automated Report-Derived Oncology VQA Benchmark for Evaluating Vision-Language Models on 3D Medical Imaging | 2026.06 | arXiv | [Paper](https://arxiv.org/abs/2606.02809) |  |
| Abdomen-NCCT Benchmark | A Multi-Center Benchmark for Abdominal Disease Diagnosis and Report Generation from Non-Contrast CT | 2026.06 | arXiv | [Paper](https://arxiv.org/abs/2606.16991) |  |
| RadOT-Eval | RadOT-Eval: Auditable Structured-Evidence Transport for Radiology Report Evaluation | 2026.06 | arXiv | [Paper](https://arxiv.org/abs/2606.08769) |  |
| ReportQA | ReportQA: QA-Based Radiology Report Evaluation | 2026.06 | arXiv | [Paper](https://arxiv.org/abs/2606.15037) |  |
| CORTEX | CORTEX: A Structured Reasoning Benchmark for Trustworthy 3D Chest CT MLLMs | 2026.06 | arXiv | [Paper](https://arxiv.org/abs/2606.27264) |  |
| MedCTA | MedCTA: A Benchmark for Clinical Tool Agents | 2026.06 | arXiv | [Paper](https://arxiv.org/abs/2606.11702) |  |
| Lung CT FM Benchmark | Foundation Models vs. Radiomics for Lung Computed Tomography: A Benchmark of Feature Extractors, Classification Heads, and Segmentation Choices | 2026.07 | arXiv | [Paper](https://arxiv.org/abs/2607.01001) | [Project](https://github.com/AI4HealthUOL/lung-ct-benchmarking) |
| Brain Oncology 3D MRI-Text | Multi-LLM Collaborative MRI Report Generation for Visual Instruction Tuning in Brain Oncology | 2026.07 | arXiv | [Paper](https://arxiv.org/abs/2607.14581) | - |
| COBRA2026 | COBRA2026: a large-scale multicenter pelvic cone-beam computed tomography projection dataset | 2026.07 | arXiv | [Paper](https://arxiv.org/abs/2607.20037) | [Project](https://doi.org/10.5281/zenodo.21322350) |
| GLI-AL | GLI-AL: A Multi-Modal Glioma MRI Label Resource with Unified Anatomy-Lesion Labels | 2026.07 | arXiv | [Paper](https://arxiv.org/abs/2607.22135) | [Project](https://www.synapse.org/Synapse:syn75210889/wiki/) |
---

<!-- ## **Related Surveys**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **Large-Scale Foundation Models for Radiological Image Analysis: Clinical Applications, Technical Challenges, and Future Directions** | J Imaging Inform Med | 2026 | [Paper](https://pubmed.ncbi.nlm.nih.gov/41530420/) | Not Available |
| **Multi-modal large language models in radiology: principles, applications, and potential** | Abdom Radiol | 2025 | [Paper](https://pubmed.ncbi.nlm.nih.gov/39621074/) | Not Available | -->
