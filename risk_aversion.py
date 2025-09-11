import streamlit as st
import numpy as np

# Utility function (CRRA: Constant Relative Risk Aversion)
def crra_utility(x, r):
    if r == 1:
        return np.log(x)
    else:
        return (x**(1-r)) / (1-r)

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Risk Aversion Estimator", page_icon="🎲", layout="centered")

# ---- HEADER ----
st.title("🎲 Risk Aversion Estimator")
st.markdown(
    """
    This interactive app presents you with a series of **random gambles**.  
    Choose whether you prefer the **sure amount** or the **risky gamble**.  
    At the end, the app will estimate your **risk aversion coefficient (r)**.
    """
)

# Subtle credit line
st.markdown("<p style='text-align: center; color: grey; font-size: 0.85em;'>Developed by Uday Damodaran for pedagogical purposes</p>", unsafe_allow_html=True)

# ---- GENERATE RANDOM GAMBLES ----
N_GAMBLES = 5  

if "gambles" not in st.session_state:
    np.random.seed()  # Different for each user/session
    st.session_state.gambles = []
    for _ in range(N_GAMBLES):
        sure = np.random.randint(20, 80)   # Sure payoff
        heads = np.random.randint(sure+10, sure+70)  # Higher payoff
        tails = np.random.randint(0, sure)           # Lower payoff
        st.session_state.gambles.append({"sure": sure, "heads": heads, "tails": tails})

choices = []

st.divider()
st.header("💡 Your Gamble Choices")

for i, g in enumerate(st.session_state.gambles):
    with st.container(border=True):
        st.subheader(f"Gamble {i+1}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Option A (Sure):** ₹{g['sure']}")
        with col2:
            st.markdown(f"**Option B (Gamble):** 50% chance of ₹{g['heads']} and 50% chance of ₹{g['tails']}")

        choice = st.radio(
            f"Which do you choose for Gamble {i+1}?",
            ["Option A (Sure)", "Option B (Gamble)"],
            key=f"choice_{i}",
            horizontal=True
        )
        choices.append((g, choice))

st.divider()

# ---- ESTIMATE R ----
if st.button("🔎 Estimate My Risk Aversion"):
    r_values = np.linspace(-1, 3, 300)  # Range of risk aversion
    scores = []

    for r in r_values:
        score = 0
        for g, choice in choices:
            EU_gamble = 0.5*crra_utility(g['heads'], r) + 0.5*crra_utility(g['tails'], r)
            U_sure = crra_utility(g['sure'], r)

            if choice == "Option A (Sure)" and U_sure >= EU_gamble:
                score += 1
            elif choice == "Option B (Gamble)" and EU_gamble >= U_sure:
                score += 1
        scores.append(score)

    best_r = r_values[np.argmax(scores)]

    # ---- OUTPUT ----
    st.success(f"✅ Your estimated risk aversion coefficient (r) is approximately: **{best_r:.2f}**")

    # Interpretation block
    st.markdown("---")
    st.markdown("### 📖 How to Interpret Your Result")
    st.markdown(
        """
        - **r ≈ 0** → Risk-neutral (indifferent between safe and risky options).  
        - **r < 0** → Risk-seeking (prefers risky options even when expected value is lower).  
        - **0 < r < 1** → Mildly risk-averse (prefers safety but may accept some risks).  
        - **r > 1** → Strongly risk-averse (strong preference for sure outcomes).  
        """
    )
