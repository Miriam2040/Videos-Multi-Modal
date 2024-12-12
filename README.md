# Video Understanding Suite: Search & QA 🎥

A comprehensive system for video search and question answering using state-of-the-art multimodal AI.

![Video Understanding System](clip-architecture-simple.svg)

## 🎯 Key Features

- **Video Search**: Semantic search using CLIP embeddings
- **Video Q&A**: Natural language interaction using LLaVA
- **Auto Question Generation**: Smart question generation based on video content
- **Visual Context**: Display of videos alongside analysis
- **Multimodal Understanding**: Integration of visual and textual information

## 💻 Notebooks

### 1. Video Search ([video_search.ipynb](video_search.ipynb))

Uses CLIP embeddings for semantic video search with frame-by-frame analysis.

```python
# Initialize system
video_dataset = VideoDataset(df)

# Search example
results = video_dataset.search("person cooking pasta", top_k=5)
```

**Sample Output:**
```
Query: "person cooking pasta"
Top Results:
1. video123: "chef preparing Italian pasta" (similarity: 0.876)
2. video456: "cooking demonstration in kitchen" (similarity: 0.843)
[...]
```

**Evaluation Results:**
```
Metrics:
- Top-1 Accuracy: 0.345
- Top-5 Accuracy: 0.567
- Mean Similarity: 0.456
```

### 2. Video Question Answering ([video_qa.ipynb](video_qa.ipynb))

Interactive Q&A system using LLaVA for video understanding.

```python
# Initialize system
video_qa = VideoQASystem(df)

# Ask question
answer = video_qa.answer_question(
    video_id="video8869",
    question="What activities are shown?"
)
```

**Sample Interaction:**
```
Video: video8869
Description: "a girl shows a pack of toy building blocks"

Q: What activities are shown?
A: The video shows a young girl demonstrating a set of toy building 
   blocks, displaying different pieces and explaining their features.
```

## 🛠 System Architecture

### Video Search Pipeline
1. **Frame Extraction**: 8 frames per video
2. **CLIP Embeddings**: 512-dimensional vectors
3. **Pooling**: Max pooling across frames
4. **Similarity**: Cosine similarity matching

### Video Q&A Pipeline
1. **Frame Analysis**: LLaVA processes key frames
2. **Context Integration**: Combines frame insights
3. **Answer Generation**: Coherent response synthesis
4. **Visual Display**: Video with Q&A results

## 📊 Performance Analysis

**Search System:**
```
Query Length Impact:
- Short queries (1-5 words): 67% accuracy
- Long queries (>5 words): 54% accuracy

Video Length Impact:
- Short videos (<30s): 72% accuracy
- Long videos (>30s): 61% accuracy
```

## 🚀 Future Improvements

**Enhanced Understanding:**
- MiniGPT4-video integration for end-to-end processing
- DINO-v2 object detection for enriched context
- Scene segmentation for temporal analysis

**Interactive Features:**
- Multi-turn conversations
- Timestamp-specific questions
- Visual object highlighting

## 🔧 Installation

```bash
pip install torch clip opencv-python pillow requests numpy pandas
```

## 🤝 Dependencies

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- CLIP
- LLaVA (via Ollama)

## 📚 References

- [LLaVA Project](https://llava-vl.github.io/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [MiniGPT4-video](https://vision-cair.github.io/MiniGPT4-video/)

## 📝 Citation

```bibtex
@misc{video-understanding-suite,
    title={Video Understanding Suite},
    author={Your Name},
    year={2024},
    publisher={GitHub},
    url={https://github.com/yourusername/video-understanding}
}
```

## 📋 License

MIT License

