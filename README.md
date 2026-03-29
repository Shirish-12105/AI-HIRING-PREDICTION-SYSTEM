# 🤖 AI-Based Hiring Prediction System

An end-to-end Machine Learning project that predicts whether a candidate should be **Hired or Rejected** based on their skills, experience, education, certifications, and salary expectations — using classic ML models and NLP feature engineering.

---

## 📌 Project Overview

Manual resume screening is slow and biased. This project automates that process using machine learning — combining **structured data** (experience, salary, projects) with **unstructured text data** (skills, certifications, job role) to make data-driven hiring decisions.

---

## 🗂️ Dataset

| Column | Type | Description |
|---|---|---|
| Skills | Text | Candidate's listed skills |
| Certifications | Text | Professional certifications |
| Job Role | Text | Desired job role |
| Experience (Years) | Numeric | Years of work experience |
| Salary Expectation ($) | Numeric | Expected salary in USD |
| Projects Count | Numeric | Number of completed projects |
| Education | Categorical | Highest education level |
| Recruiter Decision | Target | Hire / Reject |

> `Resume_ID`, `Name`, and `AI Score (0-100)` are dropped before training.

---

## 🔧 Tech Stack

- **Python 3.x**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **scikit-learn** — ML models, preprocessing, evaluation
- **TF-IDF Vectorizer** — text to numerical conversion

---

## 🚀 Project Pipeline

```
Raw CSV
   ↓
Data Cleaning (drop irrelevant cols, fill nulls, encode target)
   ↓
Text Feature Engineering (combine Skills + Certifications + Job Role)
   ↓
TF-IDF Vectorization (max 500 features)
   ↓
Label Encoding (Education)
   ↓
Feature Assembly [Numeric | Education | TF-IDF]
   ↓
Train-Test Split (80/20)
   ↓
Standard Scaling (numeric columns only)
   ↓
Model Training (LR, RF, SVM, KNN)
   ↓
Evaluation & Comparison
   ↓
Prediction Function
```

---

## 🧠 Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Linear decision boundary, fast baseline |
| Random Forest ⭐ | Best performer — handles mixed data types |
| SVM | Finds optimal separating hyperplane |
| KNN | Classifies based on nearest neighbors |

> **Best Model: Random Forest** — selected for the final prediction function.

---

## 📊 Evaluation Metrics

Each model is evaluated on:
- **Accuracy** — overall correct predictions
- **Precision** — of predicted hires, how many were correct
- **Recall** — of actual hires, how many were caught
- **F1-Score** — harmonic mean of precision and recall

---

## 🧪 Sample Prediction

```python
decision, confidence = predict_hiring(
    skills   = "Python SQL TensorFlow",
    exp      = 5,
    edu      = "B.Tech",
    cert     = "AWS",
    role     = "Data Scientist",
    projects = 4,
    salary   = 80000
)

print(decision)     # Hire
print(confidence)   # 0.91
```

---

## 📁 Project Structure

```
├── AI-Based Hiring Prediction System.csv   # Dataset
├── hiring_prediction.py                    # Full pipeline code
└── README.md                               # Project documentation
```

---

## ⚙️ How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/ai-hiring-prediction.git
cd ai-hiring-prediction
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn
```

3. Add your dataset CSV to the project folder and update the path in the script

4. Run the script
```bash
python hiring_prediction.py
```

> Or open directly in **Google Colab** and run all cells top to bottom.

---

## ✅ Key Design Decisions

- **Education not scaled** — label-encoded education is ordinal/categorical, so it's excluded from StandardScaler to avoid distorting values
- **Text cleaned consistently** — same regex cleaning applied both at training time and inside the prediction function
- **Column order locked** — features assembled as `[Numeric | Education | TF-IDF]` in both training and prediction to prevent silent mismatches
- **Unseen education handling** — prediction function defaults gracefully if an unseen education label is passed

---

## 📝 Conclusion

This project successfully developed an AI-based hiring prediction system using machine learning. The dataset contained both structured and unstructured data, requiring preprocessing steps such as handling missing values, encoding categorical variables, and converting text into numerical features using TF-IDF.

Multiple models were trained and compared. Random Forest performed best due to its ability to handle mixed data types and capture complex patterns. This project demonstrates how AI can be applied in real-world HR systems to automate resume screening and improve hiring efficiency.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn:www.linkedin.com/in/shirish-b-820a02389

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
