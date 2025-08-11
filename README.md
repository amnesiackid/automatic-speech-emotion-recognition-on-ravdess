
# Speech Emotion Recognition (SER) Project

This is a student project for course "Computational Linguistics Team Laboratory: Phonetics"

This project aims to tackle SER task by training machine learning models primarily on [Ravdess dataset](https://zenodo.org/records/1188976), which includes 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised). 

Our task is to train models to automatically detect the aforementioned emotions when receiving an audio input.
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Installation

This project includes three different models,

To deploy the baseline model and the Wav2Vec2 integrated model, please run this in terminal:

```bash
  conda env create -f ser_model.yml
```
This will install all required packages and dependencies with the correct versions.

Once the installation is complete, activate the environment:
```bash
conda activate ser_model
```

Fine_tune_distill_hubert_on_ravdess_data.ipynb is tested in [Google Colab](https://colab.google/), therefore we encourage you to run the notebook there. The first cell of the notebook will install all the needed libraries for you.


## Usage/Examples

Baseline model training script and a python script to test the model can be found in "Baseline model" directory.

A "disabled" Wav2Vec2 integrated model is trained with wav2vec2_model.ipynb. To use it, you have to apply pretrained Wav2Vec2 model on your audio data, then save the last hidden state as feature, which will be the model input.

A fintuned hubert model is trained with Fine_tune_distill_hubert_on_ravdess_data.ipynb. This model is published on [Hugging face](https://huggingface.co/amnesiackid/distilhubert-finetuned-ravdess). You can use it with a pipeline.

```python
from transformers import pipeline
classifier = pipeline("audio-classification", "amnesiackid/distilhubert-finetuned-ravdess")

audio = "/content/1001_DFA_SAD_XX.wav" # replace with your test audio path
result = classifier(audio)
result
```


## Web application
A web application which applies the Wav2Vec2 model can be found in "front end" directory.
This is a screenshot from the homepage.
![](images/screenshot.png)
## Project story
More detailed information about this project is in .pdf.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
