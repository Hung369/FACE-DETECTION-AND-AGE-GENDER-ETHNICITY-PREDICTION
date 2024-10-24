# FACE-DETECTION-AND-AGE-GENDER-ETHNICITY-PREDICTION
Given a image or video, model can detect face areas and predict age, gender and ethnicity from those detected faces

## Dependencies

* Python 3.10.11 is needed before installing program.
* Any Python IDE can be use for this project, recommend: VSCode and Pycharm
* OS: Ubuntu Linux or WSL2

## Installation

Install packages in virtual environment (venv) via this command:

```bash
  pip3 install -r requirements.txt
```

## Dataset

Dataset Link
```
https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv
```
This dataset includes a CSV of facial images that are labeled on the basis of age, gender, and ethnicity. The dataset includes 27305 rows and 5 columns.

**age** is an integer from 0 to 116, indicating the age.

**gender** is either 0 (male) or 1 (female).

**race** is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

Download dataset and move the csv file into dataset folder. For further dataset info. please check the ``` note.txt ``` file in dataset folder.

## Multi-head model
Before running inference script, you have to running training script or download a trained model via provided link in ```model_link.txt``` file in **model** folder and store it in checkpoint folder with the path ```model/checkpoint```. For detailed instruction, read the *model_link.txt* file.

**model_plot.png** shows the multi-head model's structure.

## Deployment

After setting up all the steps use bellow command to run the program
* Showing data distribution and characteristics, run this script:
```bash
python3 survey_dataset.py
```

* Want to run the training model process, run this script: 
```bash
python3 training_model.py
```
* Running model inference on image via this script:
```bash
python3 inference_image.py
```

* Running model inference on video via this script:
```bash
python3 inference_video.py
```


