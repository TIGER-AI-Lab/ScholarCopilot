# ScholarCopilot

🔥 [Try our live demo here](https://huggingface.co/spaces/TIGER-Lab/ScholarCopilot)

Scholar Copilot is an intelligent academic writing assistant that enhances the research writing process through AI-powered text completion and citation suggestions. Built by [TIGER-Lab](https://huggingface.co/TIGER-Lab), it aims to streamline academic writing while maintaining high scholarly standards.

## 🌟 Key Features

### 📝 Smart Text Generation
- **Next-3-Sentence Suggestions**: Get contextually relevant suggestions for your next three sentences
- **Full Section Auto-Completion**: Generate complete sections with appropriate academic structure and flow
- **Context-Aware Writing**: All generations consider your existing text to maintain coherence

### 📚 Intelligent Citation Management
- **Real-time Citation Suggestions**: Receive relevant paper citations based on your writing context
- **One-Click Citation Insertion**: Easily select and insert citations in proper academic format
- **Citation Bibtex Generation**: Automatically generate and export bibtex entries for your citations

## Inference Pipeline Overview

Scholar Copilot employs a unified model architecture that seamlessly integrates retrieval and generation through a dynamic switching mechanism. During the generation process, the model autonomously determines appropriate citation points using learned citation patterns. When a citation is deemed necessary, the model temporarily halts generation, utilizes the hidden states of the citation token to retrieve relevant papers from the corpus, inserts the selected references, and then resumes coherent text generation.

<img width="1022" alt="image" src="https://github.com/user-attachments/assets/487890f7-c450-49d6-ac3c-da2d9fb48eba">


## 🚀 Getting Started

To set up the ScholarCopilot demo on your own server, follow these simple steps:

1. Clone the repository:
```bash
git clone git@github.com:TIGER-AI-Lab/ScholarCopilot.git
cd ScholarCopilot/run_demo
```

2. Set up the environment:
```bash
pip install -r requirements.txt
```

3. Download the required model and data:
```bash
bash download.sh
```

4. Launch the demo:
```
bash run_demo.sh
```

## 📖 Demo Video

[![Scholar Copilot Demo Video](https://img.youtube.com/vi/QlY7S52sWDA/maxresdefault.jpg)](https://www.youtube.com/watch?v=QlY7S52sWDA)
