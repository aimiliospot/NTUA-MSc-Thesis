
# Classification of Cancer Cells in Lymph Nodes using Convolutional Neural Networks

The purpose of this thesis is to investigate how capable convolutional neural networks are at classifying the degree of cancer cell metastasis from the breast to the axillary lymph nodes using MRIs from patients.

A comprehensive understanding of the subject requires some basic knowledge of breast cancer, magnetic resonance imaging, and machine learning. For this reason, the thesis begins with a discussion on the pathogenesis, symptomatology, and subtypes of breast cancer, the methods of its diagnosis, its spread to nearby lymph nodes, and the available therapeutic approaches. Following this, an in-depth analysis of magnetic resonance imaging is provided, with emphasis on the role of the hydrogen nucleus, magnetization, resonance, energy delivery through RF pulses, relaxation times, and the method of signal detection. Additionally, an introduction to machine learning is included, along with an extensive analysis of convolutional neural networks.

All of the gained knowledge is combined in the final chapter, where the available dataset and the implementation of the convolutional neural networks presented in this thesis are thoroughly analyzed. Finally, the results of the study are presented and evaluated, and conclusions and possible future steps are documented.


## Documentation

[Thesis pdf file](https://github.com/aimiliospot/NTUA-MSc-Thesis/raw/main/thesis/Aimilios%20Potoupnis%20Master%20Thesis.pdf)

## Installation

> I assume that Python is already installed :grinning:

Clone the repository

```bash
  git clone https://github.com/aimiliospot/Personal-Blog.git
```

Create and activate virtual environment (on Windows)

```bash
  python -m venv virtualenv

  virtualenv\Scripts\activate (Windows)
```

Create and activate virtual environment (on Linux)

```bash
  python -m venv virtualenv

  source virtualenv/bin/activate (Linux)
```

Install the dependecies:

```bash
  pip install -r requirements.txt
```

## Usage

The implemented models are **AlexNet**, **DenseNet 121**, **EffientNet B0**, **GoogleNet**, **ResNet 50**, **ShuffleNet**,  each using either the **Adam** or **AdamW** optimizer. For each model there is a unique script to execute. For example, to train ShuffleNet with Adam optimizer run the following command:

```bash
python shufflenet_adam.py
```

After training is complete, the specified script will save the trained model in the 'models' directory, the training diagrams in the 'figures' directory, and the confusion matrix in the 'conf_matrices' directory.

Once all models have been trained, you can calculate evaluation metrics for all the trained models and export them into a CSV file in the evaluation_metrics directory by running the following command:

```bash
python evaluation_metrics_calculator.py
```

## Tech Stack

[![Tech Stack](https://skillicons.dev/icons?i=python,pytorch)](https://skillicons.dev)

## License

[MIT](https://choosealicense.com/licenses/mit/)
