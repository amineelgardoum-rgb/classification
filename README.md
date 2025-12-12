# Classification Project

A machine learning classification project with comprehensive data processing, model training, and evaluation capabilities.

## Project Structure

```
classification/
│
├── data/
│   ├── raw.csv              # Original dataset
│   └── processed.csv        # Cleaned and preprocessed data
│
├── models/
│   ├── encoders/            # Label encoders and feature transformers
│   ├── scaler/              # Feature scaling objects
│   ├── model.1.pkl          # Trained model version 1
│   ├── model.2.pkl          # Trained model version 2
│   └── model.3.pkl          # Trained model version 3
│
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis
│   ├── 02_process.ipynb     # Data preprocessing pipeline
│   ├── 03_visualize.ipynb   # Data visualization
│   ├── train.ipynb          # Model training experiments
│   ├── train1.ipynb         # Model training iteration 1
│   └── train2.ipynb         # Model training iteration 2
│
├── venv/                    # Virtual environment
├── .gitignore              # Git ignore rules
├── main.py                 # Main application entry point
├── requirements.txt        # Project dependencies
└── README.md              # This file

```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Notebooks

The project includes several Jupyter notebooks for different stages of the ML pipeline:

- **01_EDA.ipynb**: Explore the dataset, identify patterns, and understand feature distributions
- **02_process.ipynb**: Clean and preprocess the raw data
- **03_visualize.ipynb**: Create visualizations for insights and model performance
- **train.ipynb, train1.ipynb, train2.ipynb**: Train and evaluate different model configurations

## Models

Three model versions are saved in the `models/` directory:
- `model.1.pkl`: First iteration
- `model.2.pkl`: Second iteration
- `model.3.pkl`: Third iteration

Supporting objects like encoders and scalers are stored in their respective subdirectories.

## Data

- **raw.csv**: Original unprocessed dataset
- **processed.csv**: Cleaned dataset ready for model training

## Contributing

Feel free to open issues or submit pull requests for improvements.
