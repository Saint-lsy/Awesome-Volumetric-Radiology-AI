# Awesome-LLM-3D-Radiology-Analysis

---

## **Contrastive Learning (CLIP-based Methods)**

### **2D Pretraining**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **UniChest: Conquer-and-Divide Pre-Training for Multi-Source Chest X-Ray Classification** | IEEE TMI | 2024 | [Paper](https://arxiv.org/abs/2312.11038) | [Project](https://github.com/Elfenreigen/UniChest) |
| **ECAMP: Entity-centered Context-aware Medical Vision Language Pre-training** | Med Image Anal | 2025 | [Paper](https://arxiv.org/abs/2312.13316) | [Project](https://github.com/ToniChopp/ECAMP) |
| **A multimodal vision-language model for generalizable annotation-free pathology localization** | Nat Biomed Eng | 2026 | [Paper](https://www.nature.com/articles/s41551-025-01574-7) | Not Available |
| **Meta-EyeFM: Meta-learning Foundation Model for Ophthalmic Multi-modal Understanding** | Cell Rep Med | 2025 | [Paper](https://arxiv.org/abs/2505.08414) | Not Available |
| **ConTEXTual Net: A Multimodal Vision-Language Model for Segmentation of Pneumothorax** | J Imaging Inform Med | 2024 | [Paper](https://arxiv.org/abs/2303.01615) | Not Available |

### **3D Pretraining**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography** | arXiv / Nat Biomed Eng | 2024 / 2026 | [Paper](https://arxiv.org/abs/2403.17834) | [Project](https://github.com/ibrahimethemhamamci/CT-CLIP) |
| **A Pan-Organ Vision-Language Model for Generalizable 3D CT Representations** | medRxiv | 2025 | [Paper](https://pubmed.ncbi.nlm.nih.gov/40630577/) | Not Available |
| **Large-Scale 3D Medical Image Pre-Training With Geometric Context Priors** | IEEE TPAMI | 2025 | [Paper](https://ieeexplore.ieee.org/abstract/document/11274411) | Not Available |
| **NeuroVFM: A Contrastive Vision-Language Model for Medical Reasoning in Alzheimer's Disease Diagnosis** | WACV | 2026 | [Paper](https://openaccess.thecvf.com/content/WACV2026W/P2P/html/Sajib_NeuroVLM_A_Contrastive_Vision-Language_Model_for_Medical_Reasoning_in_Alzheimers_WACVW_2026_paper.html) | Not Available |
| **OpenVocabCT: Towards Universal Text-driven CT Image Segmentation** | IEEE TMI | 2025 | [Paper](https://arxiv.org/abs/2503.06030) | Not Available |

---

## **Multimodal Large Language Models (MLLM-based Methods)**

> **Note**
> We categorize MLLM-based methods by the actual visual input form rather than only by the source modality name:
> - **2D ViT + LLM**: native 2D medical image input.
> - **3D ViT + LLM**: direct volumetric CT/MRI input.
> - **Hybrid / Serialized Volume + LLM**: supports both 2D inputs and full 3D studies, including slice-serialization or flattened-volume designs.
> - **Agentic Systems**: tool-augmented or reasoning-agent style systems.
> - **Workflow / Application Systems**: MLLM-based applications/workflows that are not core foundation-model papers.

### **2D ViT + LLM**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **CXR-LLaVA: a multimodal large language model for interpreting chest X-ray images** | Eur Radiol | 2025 | [Paper](https://arxiv.org/abs/2310.18341) | [Project](https://github.com/ECOFRI/CXR_LLAVA) |
| **Constructing a Unified Vision-Language Model for Chest Radiograph-based Diagnostics, Medical Education, and Data Augmentation** | Radiol Cardiothorac Imaging | 2025 | [Paper](https://pubmed.ncbi.nlm.nih.gov/41410571/) | Not Available |
| **LUMEN: Longitudinal Multi-Modal Radiology Model for Prognosis and Diagnosis** | arXiv | 2026 | [Paper](https://arxiv.org/abs/2602.21142) | Not Available |
| **LLaVA-Med: Large Language and Vision Assistant for Bio-Medical** | arXiv | 2023 | [Paper](https://arxiv.org/abs/2306.00890) | [Project](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) |
| **HuatuoGPT-Vision, Towards Injecting Medical Visual Knowledge into Multimodal LLMs at Scale** | arXiv | 2024 | [Paper](https://arxiv.org/abs/2406.19280) | [Project](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B) |
| **BiMediX2: Bio-Medical EXpert LMM for Diverse Medical Modalities** | Findings of EMNLP | 2025 | [Paper](https://aclanthology.org/2025.findings-emnlp.756.pdf) | [Project](https://github.com/mbzuai-oryx/BiMediX2) |
| **MediX-R1: Medical Vision Language Model with Reinforcement Learning** | arXiv | 2026 | [Paper](https://arxiv.org/abs/2602.23363) | [Project](https://huggingface.co/MBZUAI/MediX-R1-2B) |
| **BiomedGPT: A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks** | Nat Med | 2024 | [Paper](https://arxiv.org/abs/2305.17100) | [Project](https://huggingface.co/PharMolix/BioMedGPT-LM-7B) |
| **Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2503.13939) | [Project](https://github.com/Yuxiang-Lai117/Med-R1) |
| **MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2502.19634) | [Project](https://huggingface.co/JZPeterPan/MedVLM-R1) |
| **Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs** | arXiv | 2025 | [Paper](https://ui.adsabs.harvard.edu/abs/2025arXiv250616962S/abstract) | [Project](https://huggingface.co/manglu3935/Chiron-o1-2B) |
| **QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2506.00711) | [Project](https://huggingface.co/ddvd233/QoQ-Med-VL-7B) |
| **MedMO: Medical Vision Language Model with Mixture of Experts** | arXiv | 2026 | [Paper](https://arxiv.org/abs/2602.06965) | [Project](https://huggingface.co/MBZUAI/MedMO-4B) |
| **Fleming-VL: Towards Universal Medical Visual Reasoning with Multimodal LLMs** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2511.00916) | [Project](https://huggingface.co/UbiquantAI/Fleming-VL-8B) |
| **Med-Flamingo: a Multimodal Medical Few-shot Learner** | ML4H / arXiv | 2023 | [Paper](https://arxiv.org/abs/2307.15189) | [Project](https://huggingface.co/med-flamingo/med-flamingo) |
| **MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis** | OpenReview / arXiv | 2024 | [Paper](https://arxiv.org/abs/2407.04106) | [Project](https://github.com/Vision-CAIR/MiniGPT-Med) |
| **XRaySwinGen: Automatic medical reporting for X-ray exams with multimodal model** | Heliyon | 2024 | [Paper](https://www.sciencedirect.com/science/article/pii/S2405844024035473) | Not Available |
| **CheXagent: A multimodal foundation model for chest X-ray interpretation** | arXiv | 2024 | [Paper](https://arxiv.org/abs/2401.12208) | [Project](https://huggingface.co/StanfordAIM/CheXagent) |
| **MAIRA-1: A specialised large multimodal model for radiology report generation** | arXiv | 2023 | [Paper](https://arxiv.org/abs/2311.13668) | [Project](https://aka.ms/maira) |
| **MAIRA-2: Grounded Radiology Report Generation** | arXiv | 2024 | [Paper](https://arxiv.org/abs/2406.04449) | [Project](https://huggingface.co/microsoft/maira-2) |
| **RadVLM: A Multitask Conversational Vision-Language Model for Radiology** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2502.03333) | [Project](https://github.com/uzh-dqbm-cmi/RadVLM) |
| **Towards a clinically accessible radiology foundation model: open-access and lightweight, with automated evaluation** | arXiv | 2024 | [Paper](https://arxiv.org/abs/2403.08002) | [Project](https://github.com/microsoft/LLaVA-Rad) |

### **3D ViT + LLM**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **Merlin: A Vision Language Foundation Model for 3D Computed Tomography** | arXiv / Nature | 2024 / 2026 | [Paper](https://arxiv.org/abs/2406.06512) | [Project](https://github.com/StanfordMIMI/Merlin) |
| **CT-CHAT: Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography** | arXiv / Nat Biomed Eng | 2024 / 2026 | [Paper](https://arxiv.org/abs/2403.17834) | [Project](https://github.com/ibrahimethemhamamci/CT-CLIP) |
| **Med3DVLM: An Efficient Vision-Language Model for 3D Medical Image Analysis** | IEEE JBHI | 2025 | [Paper](https://arxiv.org/abs/2503.20047) | [Project](https://github.com/mirthAI/Med3DVLM) |
| **Brain3D: Brain Report Automation via Inflated Vision Transformers in 3D** | arXiv | 2026 | [Paper](https://arxiv.org/abs/2602.22098) | [Project](https://huggingface.co/praiselab-picuslab/BrainGemma3D) |

### **Hybrid / Serialized Volume + LLM**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **MedGemma Technical Report** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2507.05201) | [Project](https://huggingface.co/google/medgemma-4b-it) |
| **Hulu-Med: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2510.08668) | [Project](https://github.com/ZJUI-AI4H/Hulu-Med) |
| **OctoMed: Data Recipes for State-of-the-Art Multimodal Medical Reasoning** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2511.23269) | [Project](https://huggingface.co/OctoMed/OctoMed-7B) |
| **RadFM: Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data** | arXiv / Nat Commun | 2023 / 2025 | [Paper](https://arxiv.org/abs/2308.02463) | [Project](https://github.com/chaoyi-wu/RadFM) |
| **VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge** | CVPR | 2025 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper.html) | [Project](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework) |
| **Advancing Multimodal Medical Capabilities of Gemini** | arXiv / Nat Med | 2024 / 2025 | [Paper](https://arxiv.org/abs/2405.03162) | [Project](https://research.google/blog/advancing-medical-ai-with-med-gemini/) |

### **Agentic Systems**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **MedRAX: Medical Reasoning Agent for Chest X-ray** | arXiv | 2025 | [Paper](https://arxiv.org/abs/2502.02673) | [Project](https://github.com/bowang-lab/MedRAX) |
| **CXR-Agent: Vision-language models for chest X-ray interpretation with uncertainty aware radiology reporting** | arXiv | 2024 | [Paper](https://arxiv.org/abs/2407.08811) | Not Available |
| **3DMedAgent: Unified Perception-to-Understanding for 3D Medical Analysis** | arXiv | 2026 | [Paper](https://arxiv.org/abs/2602.18064) | Not Available |

### **Workflow / Application Systems**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **MiniGPT-Pancreas: Multimodal Large Language Model for Pancreas Cancer Observation and Localization in CT Images** | J Healthc Inform Res | 2025 | [Paper](https://pubmed.ncbi.nlm.nih.gov/41658402/) | Not Available |
| **A vision-language model-based approach for lung cancer diagnosis using lossless 3D CT images: evaluation of GPT-4.1 and GPT-4o for patient-level malignancy assessment** | Med Biol Eng Comput | 2025 | [Paper](https://pubmed.ncbi.nlm.nih.gov/41439200/) | Not Available |
| **Prospective evaluation of artificial intelligence (AI) in lumbar spine magnetic resonance imaging (MRI) workflow: from deep learning (DL)-enhanced accelerated acquisition to simultaneous vision-language model (VLM)-based automated report generation** | Eur J Radiol | 2026 | [Paper](https://pubmed.ncbi.nlm.nih.gov/41579672/) | Not Available |
| **BioVLM-T: A temporal framework for radiology report generation using pre-trained vision language foundational models** | SPIE Medical Imaging | 2025 | [Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13410/134100W/BioVLM-T--A-temporal-framework-for-radiology-report-generation/10.1117/12.3047498.short) | Not Available |

### **Other Related LLM / MLLM Papers**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **Med-PaLM: Large Language Models for Medical Question Answering** | Nature | 2023 | [Paper](https://www.nature.com/articles/s41586-023-06291-2) | Not Available |

---

## **Datasets, Benchmarks & Evaluation Metrics**

### **Datasets & Benchmarks**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **3DReasonKnee: Advancing Grounded Reasoning in Medical Vision Language Models** | arXiv / Pac Symp Biocomput | 2025 / 2026 | [Paper](https://arxiv.org/abs/2510.20967) | [Project](https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee) |
| **RadGenome-Chest CT: A Grounded Vision-Language Dataset for Chest CT Analysis** | arXiv / Sci Data | 2024 / 2025 | [Paper](https://arxiv.org/abs/2404.16754) | [Project](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT) |

### **Evaluation Metrics**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |

---

## **Surveys**
| Title | Venue | Date | Paper Link | Project Page |
| :--- | :--- | :--- | :--- | :--- |
| **Large-Scale Foundation Models for Radiological Image Analysis: Clinical Applications, Technical Challenges, and Future Directions** | J Imaging Inform Med | 2026 | [Paper](https://pubmed.ncbi.nlm.nih.gov/41530420/) | Not Available |
| **Multi-modal large language models in radiology: principles, applications, and potential** | Abdom Radiol | 2025 | [Paper](https://pubmed.ncbi.nlm.nih.gov/39621074/) | Not Available |
