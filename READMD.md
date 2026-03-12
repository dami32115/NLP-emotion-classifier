This project implements an emotion classification model using TF-IDF features and machine-learning models to classify tweets into six emotion categories. I used content from the labs and lectures and built a clean pipeline that can be evaluated easily.

## Dataset location:
Place the dataset CSV here:
data/Option 1-Training Dataset.csv

It must include at least these columns:

text → the tweet text  
label → an integer 0–5 (see label mapping below)  

Code location:
Main script: src/main.py   
Output image (created automatically): reports/figures/confusion_matrix.png
data: dataset folder

## How to run
1. Create and activate a virtual environment
Mac/Linux:
python3 -m venv venv
source venv/bin/activate
Windows:
python -m venv venv
venv\Scripts\activate
2. Install the required packages
pip install -r requirements.txt
Your requirements.txt should be generated using:
pip freeze > requirements.txt
3. Run the project
python src/main.py
The script preprocess the text, extract TF-IDF features, train Logistic Regression and Linear SVM, evaluate on a test splits and saves the confusion matrix to: reports/figures/confusion_matrix.png 


## Methods

### Preprocessing
- lowercasing
- Removing URLS
- Removing @mentions
- Cleaning hashtags
- Removing extra whitespace 

### Feautures 
- TD-IDF unigrams
- TD-IDF unigrams + bigrams tested
- min_df=2
- max_df=0.9

### Models
- logistic regression
- linear SVM
- Hyperparameter sweep over C = 0.25, 0.5, 1.0, 2.0

### Metrics
- Accuracy
- Macro-F1
- Classification report
- Confusion matrix

## Results 

Best model: **Linear SVM(Unigrams)
- Test Accuracy: **0.8854**
- Test Macro-F1: **0.8645**

## label Mapping
 
 0 = anger 
 1 = fear
 2 = joy
 3 = love
 4 = sadness
 5 = suprise
 
 ## Reproducibility 

 - Random seed = 42
 - Stratisfied splits 
 - All package versions pinned in requirements.txt

## Requirements
click==8.3.1
contourpy==1.3.3
cycler==0.12.1
fonttools==4.60.1
joblib==1.5.2
kiwisolver==1.4.9
matplotlib==3.10.7
nltk==3.9.2
numpy==2.3.5
packaging==25.0
pandas==2.3.3
pillow==12.0.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
regex==2025.11.3
scikit-learn==1.7.2
scipy==1.16.3
six==1.17.0
threadpoolctl==3.6.0
tqdm==4.67.1
tzdata==2025.2
















