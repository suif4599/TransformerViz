# TransformerViz

A visualization tool for exploring attention mechanisms in transformer models.

## Overview

TransformerViz is a PyQt5-based GUI application that allows users to visualize and analyze attention patterns in transformer models. Currently supports BERT models for both English and Chinese languages.

## Features

- Interactive visualization of transformer attention weights
- Support for multiple visualization modes:
  - Position modes: which attention to use (encoder, decoder or encoder-decoder)
  - Layer mixing modes: which layer to use (first, final, average)
  - Head mixing modes: which head to show (all, first, average)
- Adjustable temperature parameter for attention weight visualization
- Real-time updates for attention pattern visualization
- Multi-head attention visualization
- High scalability: any model that implements the methods in `/code/abstract_module.py` can be added easy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TransformerViz.git
cd TransformerViz
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Bert
- Download bert model
  - English model: Download the following files from [website](https://huggingface.co/google-bert/bert-base-uncased/tree/main) and create a folder `/code/english` to contain these files:
    - config.json
    - model.safetensors
    - tokenizer_config.json
    - tokenizer.json
  - Chinese model: Download the following files from [website](https://huggingface.co/google-bert/bert-base-chinese/tree/main) and create a folder `/code/chinese` to contain these files:
    - config.json
    - model.safetensors
    - tokenizer_config.json
    - tokenizer.json

## Usage

Run the application:
```bash
python main.py
```

### Basic Operations:

1. Select a model from the left panel (English BERT or Chinese BERT)
2. Enter text in the input box (use "_" for mask tokens)
3. Click "Visualize" to see attention patterns
4. Adjust visualization parameters:
   - AttentionPos: Select attention position mode
   - LayerMixMode: Choose how to combine layer information
   - HeadMixMode: Select head combination method
   - Temperature: Adjust attention weight distribution
   - FontSize: Change text display size

## Project Structure

- `core/`: Core model implementations
  - `abstract_module.py`: Base class for transformer modules
  - `bert.py`: BERT model implementation
- `gui/`: GUI implementation
  - `ui.py`: Main GUI logic
  - `viz_frame.py`: Visualization components
  - `validators.py`: Input validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.