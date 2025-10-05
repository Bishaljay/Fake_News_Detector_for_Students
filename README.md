# 📢 Fake News Detector for Students

A **web-based application** that helps students **detect fake news articles** using Machine Learning and Natural Language Processing techniques. The project aims to educate users about **credible vs fake news** and provide **insights into article reliability**.

---

## 🛠 Features

- **Analyze Articles:** Check if an article is real or fake.  
- **Score Breakdown:** Understand credibility, reliability, and trustworthiness.  
- **Learning Module:** Learn how to spot fake news and identify red flags.  
- **History Tracking:** Keep track of previously analyzed articles.  
- **Interactive & Educational:** Designed for students to learn while analyzing.

---

## 💻 Technologies Used

- **Frontend / App Framework:** Streamlit  
- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn (Naive Bayes, TF-IDF, etc.)  
- **Visualization:** Plotly, Matplotlib  
- **Web Scraping (Optional):** BeautifulSoup  

---

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Bishaljay/fake_news_detector_for_students.git
cd fake_news_detector_for_students
````

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the app:**

```bash
streamlit run app.py
```

---

## 📂 Folder Structure

```
fake_news_detector_for_students/
│
├─ app.py                  # Main Streamlit app
├─ requirements.txt        # Dependencies
├─ data/                   # Dataset files (CSV/JSON)
└─ pages/                  # Optional multiple pages for Streamlit
```

---

## 🎓 How It Works

1. User inputs or pastes an article into the app.
2. The system uses a **Machine Learning model** to predict if the news is fake or real.
3. Displays a **score breakdown** and **credibility indicators**.
4. Provides a **learning section** to educate students about spotting fake news.

---

## 🌟 Footer

The app includes a professional footer:

```
© 2025 Bishal Jaysawal | Coding + ❤️ + Passion = Project
```

---

## 📢 Acknowledgements

* Inspired by educational projects to combat misinformation.
* Uses open-source Python libraries: Streamlit, Pandas, NumPy, Scikit-learn, Plotly.

---

## 📌 License

This project is **for educational purposes only**.
