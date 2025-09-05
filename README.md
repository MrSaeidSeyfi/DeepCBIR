# DeepCBIR - Content-Based Image Retrieval System

A deep learning-based Content-Based Image Retrieval (CBIR) system that uses ResNet50 embeddings and cosine similarity to find similar images.

## Project Structure

```
DeepCBIR/
├── src/
│   ├── __init__.py
│   ├── models.py          # Neural network model handling
│   ├── preprocessing.py   # Image preprocessing and embedding extraction
│   ├── similarity.py      # Similarity search functionality
│   └── cbir_system.py     # Main CBIR system orchestrator
├── main.py                # Command-line interface
├── app.py                 # Gradio web interface
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
```

## Features

- **Deep Learning Embeddings**: Uses pre-trained ResNet50 for feature extraction
- **Cosine Similarity**: Efficient similarity search using cosine distance
- **Batch Processing**: Processes entire folders of images
- **GPU Support**: Automatic GPU acceleration when available
- **Modular Design**: Clean, maintainable code structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MrSaeidSeyfi/DeepCBIR
cd DeepCBIR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 🌐 Web Interface

Launch the Gradio web interface for an easy-to-use graphical interface:


```bash
python gradio_app.py
```

The web interface will be available at `http://localhost:7860` and provides graphical tools for searching similar images.


### 💻 Command Line Interface

```bash
python main.py --folder /images_directory --query /query_image_path --topk 5
```

### Parameters

- `--folder`: Path to the folder containing images to search through
- `--query`: Path to the query image
- `--topk`: Number of similar images to return (default: 5)

### Example

```bash
python main.py --folder data/images --query query.jpg --topk 10
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

## Technical Details

- **Model**: ResNet50 pre-trained on ImageNet
- **Feature Extraction**: 2048-dimensional embeddings from the penultimate layer
- **Similarity Metric**: Cosine similarity
- **Image Preprocessing**: Resize to 224x224, normalize with ImageNet statistics

## Performance

- GPU acceleration automatically enabled when CUDA is available
- Efficient batch processing for large image collections
- Memory-optimized embedding extraction

