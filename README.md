## 🧠 About the Project

As a 2nd-year Data Science enthusiast, I built this project to dive deeper into **real-world sports analytics**. The goal is to leverage historical IPL ball-by-ball data and use machine learning to predict the **probability of a wicket** between any batsman and bowler matchup, based on past stats and phase of play.

This project not only showcases predictive analytics but also demonstrates my learning around **model building, feature engineering, and interactive dashboards using Streamlit**.

---

## 🌐 Real-World Applications

- 📊 **Team strategy analysis**: Knowing which bowler is likely to dismiss a batsman helps in match planning  
- 🎯 **In-game tactics**: Choose bowling matchups based on data-driven insights  
- 🧠 **Cricket analytics platforms**: Enhance fan engagement with real-time probabilities  
- 📺 **Commentary augmentation**: Feed prediction data into broadcasters' pre-match analysis

---

## 🔍 Features

- 🎯 Predicts chance of a **wicket** between batsman and bowler
- 📈 Bar charts for **dismissals and strike rate matchups**
- 📊 Visual analytics using Altair
- 🔮 Uses **Random Forest Classifier** trained on IPL data
- 🧪 Interactive Streamlit interface
- ⚡ Deployed and accessible [Live Here](https://player-matchup-app.streamlit.app)

---

## 🛠️ Tech Stack

- Python, Pandas, Scikit-learn
- Joblib + Gzip for model compression
- Altair for visuals
- Streamlit for deployment
- Git + GitHub for version control

---

## 🚀 How We Built It

1. Collected and cleaned IPL ball-by-ball data  
2. Engineered features like:
   - Batsman vs Bowler strike rate  
   - Dismissal count between pairs  
   - Match phase (Powerplay, Middle, Death)  
3. Trained a Random Forest Classifier to classify **wicket vs no wicket**
4. Built a clean UI using Streamlit + added Altair charts
5. Compressed files using `joblib` and `gzip` for deployment
6. Uploaded to GitHub → Deployed on Streamlit Cloud

---

## 🧪 How to Use

1. Open the live app: [https://player-matchup-app.streamlit.app](https://player-matchup-app.streamlit.app)  
2. Select a **batsman**, **bowler**, and **match phase**
3. Click “🔍 Predict Outcome”
4. See:
   - Prediction result
   - Wicket chance %
   - Visual analysis of strike rate & past dismissals

---

## 📁 File Structure

├── app.py
├── model.pkl.gz
├── data_with_features.csv.gz
├── le_batsman.pkl / le_bowler.pkl / le_phase.pkl
├── requirements.txt
├── screenshots/
│ ├── home.png
│ ├── result.png
│ └── app-demo.gif
