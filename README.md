# SMS Spam Detector 

A machine learning project to classify SMS messages as **spam** or **ham** (not spam). This repository contains the code, data preprocessing steps, and model training for an SMS spam detection system using Python.

##  Features
 Load and preprocess SMS spam dataset  
 Vectorize text data using TF-IDF  
 Train a classification model (e.g., Naive Bayes)  
 Save trained model and vectorizer  
 Make predictions on new messages  
 Evaluate model performance with accuracy, precision, recall  

##  Quick Start
1. **Clone this repo:**
   git clone https://github.com/Codemaster1803/SMS-Spam-detector.git
   cd SMS-Spam-detector

2. **Install dependencies:**
   pip install -r requirements.txt
   *(Make sure your `requirements.txt` includes libraries like pandas, scikit-learn, numpy, etc.)*

3. **Run the Jupyter Notebook:**
   jupyter notebook sms_spam_detector.ipynb

4. **Try the trained model:**
   You can load the saved model (model3.pkl) and vectorizer (vectorizer3.pkl) to classify new SMS texts.

##  Dataset
- Dataset used: SMS Spam Collection Dataset (UCI Machine Learning Repository or Kaggle).
- The dataset contains SMS messages labeled as `spam` or `ham`.

##  How It Works
- **Text Cleaning:** Remove unnecessary characters, convert text to lowercase, etc.
- **Vectorization:** Convert cleaned text into numerical features using TF-IDF.
- **Model Training:** Train a classifier (e.g., Naive Bayes) on the training data.
- **Evaluation:** Check the performance on test data using metrics like accuracy, precision, recall, and F1-score.
- **Deployment Ready:** Save trained model and vectorizer to classify new incoming messages.


## 🙋‍♂ Author
- **Codemaster1803**
- GitHub: [@Codemaster1803](https://github.com/Codemaster1803)
