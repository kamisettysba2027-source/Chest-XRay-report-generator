# Chest X-Ray Automated Report Generation

A deep learning system that automatically generates medical reports from chest X-ray images using a CNN-LSTM architecture.

## Performance

- Final Test Accuracy: 91% (Phase 2 model)
- Training Time: ~25 hours
- Dataset: Indiana Chest X-Ray (IU-XRay) - 7,470 images
- Model Parameters: 25.4 million

## Architecture

- Encoder: InceptionV3 (pre-trained on ImageNet)
- Decoder: 2-layer LSTM with word embeddings
- Input: 256x256 grayscale X-ray images
- Output: Medical text report (up to 150 words)

## Training Progression

| Phase | Images | Accuracy | Time |
|-------|--------|----------|------|
| Phase 1 | 1,000 | ~88% | 2-3h |
| Phase 2 | 2,500 | ~91% | 5-6h |
| Phase 3 | 7,470 | ~85% | 10-12h |

## Project Structure

- models/           Trained models for each phase
- outputs/          Results, metrics, visualizations
- docs/             Documentation
- streamlit_app.py  Interactive web dashboard
- notebook.ipynb    Complete training pipeline
- requirements.txt  Python dependencies
- README.md         This file

## Installation

    pip install -r requirements.txt

## Usage

### Run the Streamlit Demo

    streamlit run streamlit_app.py

Then open browser to: http://localhost:8501

### Run the Training Notebook

Open the notebook in Google Colab or Jupyter and run cells in order following the section headers.

## Technology Stack

- TensorFlow 2.x / Keras
- Python 3.10+
- NumPy, Pandas, Pillow
- Matplotlib for visualizations
- Streamlit for web demo

## Disclaimer

This project is for research and educational purposes only.
Not validated for clinical use without proper medical review and regulatory approval.

## License

MIT License

## Acknowledgments

- Indiana University for the chest X-ray dataset
- Google Colab for GPU resources
- TensorFlow team for the deep learning framework