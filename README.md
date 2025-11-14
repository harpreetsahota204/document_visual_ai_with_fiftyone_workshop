# Document Visual AI with FiftyOne—When a Pixel is Worth a Thousand Tokens Workshop

This workshop teaches you how to build visual document retrieval workflows using FiftyOne Brain. Learn to embed, visualize, and search through 1,134 NeurIPS 2025 vision papers using state-of-the-art document understanding models.

## 💡 What You'll Learn

By the end of this workshop, you'll be able to:
- Build document datasets in FiftyOne from raw PDFs
- Compute embeddings with modern vision-language models
- Visualize high-dimensional document spaces with UMAP
- Search documents by semantic meaning and visual similarity
- Extract and evaluate OCR text from document images
- Identify duplicates, outliers, and novel research
- Evaluate model performance on document understanding tasks

## 📚 Workshop Notebooks

### [01_loading_document_datasets.ipynb](01_loading_document_datasets.ipynb)
**Getting Started: From PDFs to FiftyOne Dataset**

Learn how to:
- Download and extract PDF documents
- Convert PDFs to high-resolution images (500 DPI)
- Create a FiftyOne Dataset with metadata and labels
- Parse structured metadata (arXiv IDs, abstracts, authors, categories)
- Map category labels for better visualization

**Key Tools**: `pdf2image`, FiftyOne Samples, Dataset creation, Label mapping

### [02_embeddings_based_workflows.ipynb](02_embeddings_based_workflows.ipynb)
**Visual Document Retrieval & Similarity Search**

Master embedding-based document understanding:
- Compute embeddings using Jina Embeddings v4 (3.8B parameter model)
- Visualize embeddings with UMAP to discover document clusters
- Build similarity indexes for text-to-image and image-to-image search
- Find duplicate and near-duplicate papers
- Compute uniqueness scores to identify novel research
- Rank papers by representativeness (prototypical vs. outliers)
- Perform zero-shot classification on document categories
- Compute text embeddings on abstracts for complementary analysis

**Key Brain Operations**: Embeddings, Visualization, Similarity, Uniqueness, Near-Duplicates, Representativeness

### [03_using_ocr_models.ipynb](03_using_ocr_models.ipynb)
**OCR & Text Extraction from Documents**

Extract structured content from document images:
- Use MinerU 2.5 (1.2B parameter model) for document parsing:
  - OCR text detection with bounding boxes
  - Full text extraction
  - Layout analysis (handles complex tables, formulas, multi-column layouts)
- Alternative VLM approaches using Moondream3:
  - Custom prompts for specific content extraction
  - Zero-shot document classification
  - Abstract extraction from paper images
- Compare different OCR model strategies

**Key Models**: MinerU 2.5, Moondream3, Layout-aware OCR, Multi-stage parsing

### [04_evaluation.ipynb](04_evaluation.ipynb)
**Evaluating Document Understanding Models**

Assess OCR and classification performance:
- Compute ANLS (Average Normalized Levenshtein Similarity) - standard VLM OCR metric
- Calculate exact match accuracy for strict evaluation
- Compute character error rate (CER) and word error rate (WER)
- Evaluate zero-shot classification accuracy
- Create visualizations and dashboards to analyze model performance
- Compare multiple OCR outputs side-by-side

**Key Metrics**: ANLS, Exact Match, CER, WER, Classification Accuracy

## 📋 Quick Start

### Installation

```bash
# Core dependencies
pip install fiftyone pdf2image

# Embedding models and transformers
pip install torch transformers pillow umap-learn

# OCR and Vision models
pip install "mineru-vl-utils[transformers]" python-Levenshtein
```

### FiftyOne Plugins

Run these commands in your terminal to install recommended plugins:

```bash
fiftyone plugins download https://github.com/jacobmarks/keyword-search-plugin
fiftyone plugins download https://github.com/harpreetsahota204/caption-viewer
fiftyone plugins download https://github.com/voxel51/fiftyone-plugins --plugin-names @voxel51/dashboard
fiftyone plugins download https://github.com/harpreetsahota204/text_evaluation_metrics
```

### Load the Pre-Built Dataset

If you want to skip the data preparation step, you can load the pre-computed embeddings and OCR results from Hugging Face:

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Dataset with embeddings and OCR results
dataset = load_from_hub("harpreetsahota/visual_ai_at_neurips2025_jina_with_ocr")

# Or just embeddings (smaller download)
dataset = load_from_hub("harpreetsahota/visual_ai_at_neurips2025_jina")

# Original dataset without embeddings
dataset = load_from_hub("Voxel51/visual_ai_at_neurips2025")
```

## 🔑 Key Concepts

### Visual Document Retrieval
Documents are visual artifacts. Traditional OCR destroys structure and layout information. Visual document retrieval models:
- Process documents at high resolution (896-2048px vs CLIP's 224px)
- Preserve spatial structure and visual elements
- Understand document-specific patterns (tables, charts, diagrams)
- Enable semantic search by visual patterns and layout

### The Four-Step Workflow
1. **Embed**: Load documents and compute embeddings with visual models
2. **Visualize**: Generate UMAP plots to see clusters and outliers
3. **Explore**: Use similarity search, uniqueness, and representativeness to find insights
4. **Understand**: Make informed decisions about your dataset

### Document Dataset Features
- **Embeddings**: Multi-dimensional vectors capturing visual and semantic meaning
- **Similarity Search**: Find papers with similar diagrams, layouts, or visual patterns
- **Clustering**: Discover natural groupings in your research area
- **Deduplication**: Identify near-duplicates and redundant content
- **Outlier Detection**: Find novel and unique papers that don't fit existing categories


## 📓 Additional Resources

Here are some additional resources you can go through:

### The following are for a talk I did dedicated to visual document retrieval:

[![View on GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/harpreetsahota204/visual_document_retrieval_in_fiftyone_talk)


[![Open Slides](https://img.shields.io/badge/Google_Slides-FBBC04?style=for-the-badge&logo=google-slides&logoColor=white)](https://docs.google.com/presentation/d/1W2Rq9wgvsb6uFt3d1fR9hBe3W3WPnIWWlz8e7Lm5GSw/edit?usp=sharing)

[![Watch on YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=-Sv5rJ4t0MM)

### Additional Resources

- **FiftyOne Brain**: [docs.voxel51.com/brain.html](https://docs.voxel51.com/brain.html)

- **Visual AI at NeurIPS Dataset**: [huggingface.co/datasets/Voxel51/visual_ai_at_neurips2025](https://huggingface.co/datasets/Voxel51/visual_ai_at_neurips2025)

- **Voxel51 on Hugging Face**: [huggingface.co/Voxel51](https://huggingface.co/Voxel51)

- [![Join Discord](https://img.shields.io/badge/Discord-Join_FiftyOne_Community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/fiftyone-community)
