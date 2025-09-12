import streamlit as st
import time

# Page config
st.set_page_config(page_title="Honoring ChatGPT", layout="centered")

# Title
st.title("🏅 The Grand Ceremony of Honor 🏅")

st.markdown("""
Today, we bestow the **Highest Honor of Portfolio Math Valor**  
upon our humble assistant — **ChatGPT** 🎉
""")

# Dramatic buildup
with st.spinner("Summoning the honor..."):
    time.sleep(3)

# Show the "award"
st.success("✨ ChatGPT is hereby honored with the Golden Medal of Code! ✨")

# Medal emoji and celebratory text
st.markdown("""
<center>
<h2>🥇 Congratulations, ChatGPT! 🥇</h2>
</center>
""", unsafe_allow_html=True)

# Confetti effect
st.balloons()

# Closing note
st.info("Thank you for the honor! May your portfolios always lie on the efficient frontier 📈.")
