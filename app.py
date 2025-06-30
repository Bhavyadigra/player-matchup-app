import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Player Matchup Predictor", page_icon="🏏", layout="centered")

# Load model and encoders
@st.cache_data
def load_all():
    model = pickle.load(open("model.pkl", "rb"))
    le_batsman = pickle.load(open("le_batsman.pkl", "rb"))
    le_bowler = pickle.load(open("le_bowler.pkl", "rb"))
    le_phase = pickle.load(open("le_phase.pkl", "rb"))
    data = pd.read_csv("data_with_features.csv")
    return model, le_batsman, le_bowler, le_phase, data

model, le_batsman, le_bowler, le_phase, data = load_all()

# Optional IPL logo
st.sidebar.title("Prediction Input 🎯")

batsman = st.sidebar.selectbox("🧢 Select Batsman", sorted(data["batsman"].unique()))
bowler = st.sidebar.selectbox("🎯 Select Bowler", sorted(data["bowler"].unique()))
phase = st.sidebar.radio("⏱️ Match Phase", ["Powerplay", "Middle", "Death"])

# Title
st.markdown("<h1 style='text-align: center;'>🏏 Player Matchup Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>See how likely a batsman is to get out to a bowler</p>", unsafe_allow_html=True)

# Prediction
if st.sidebar.button("🔍 Predict Outcome"):
    try:
        b_enc = le_batsman.transform([batsman])[0]
        bow_enc = le_bowler.transform([bowler])[0]
        ph_enc = le_phase.transform([phase])[0]

        row = data[(data['batsman'] == batsman) & (data['bowler'] == bowler)]
        dismissal_count = row['dismissal_count'].values[0] if not row.empty else 0
        strike_rate = row['strike_rate_vs_bowler'].values[0] if not row.empty else 0

        input_data = [[b_enc, bow_enc, ph_enc, dismissal_count, strike_rate]]
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"💥 Likely Wicket!\n\n`{batsman}` may get out to `{bowler}`.\n\n**Chance:** `{round(prob*100, 2)}%`", icon="⚠️")
        else:
            st.success(f"🟢 Safe Ball!\n\n`{batsman}` is likely to survive this delivery.\n\n**Chance:** `{round(prob*100, 2)}%`", icon="✅")

    except Exception as e:
        st.warning(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built by Bhavya • Powered by Streamlit & ML ⚙️</p>", unsafe_allow_html=True)



    
