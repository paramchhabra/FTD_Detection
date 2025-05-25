from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import streamlit as st
import json
from dotenv import load_dotenv
import os
from symptom import symptomtest
from ftdmri import predictftd, predictmonth

# Load environment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# State setup
if "mrichecked" not in st.session_state:
    st.session_state.mrichecked = False
if "ftdprob" not in st.session_state:
    st.session_state.ftdprob = {"Patient": 0, "Non_Patient": 0}
if "monthprob" not in st.session_state:
    st.session_state.monthprob = {"M0": 0, "M12": 0, "M18": 0, "M6": 0}
if "symptom" not in st.session_state:
    st.session_state.symptom = []
if "svalues" not in st.session_state:
    st.session_state.svalues = []
if "thresh" not in st.session_state:
    st.session_state.thresh = 0.6

# Load symptoms
if "symptom_dict" not in st.session_state:
    with open("symptoms.json", "r") as f:
        st.session_state.symptom_dict = json.load(f)
if "symptom_list_str" not in st.session_state:
    all_symptoms = []
    for category, symptoms in st.session_state.symptom_dict.items():
        for sym, score in symptoms.items():
            all_symptoms.append(f"{sym}:{score}")
    st.session_state.symptom_list_str = ", ".join(all_symptoms)

# Load LLM
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

# Define Prompt
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", """
You are a medical expert specializing in Frontotemporal Dementia (FTD).
- MRI_FTD_Probability: Indicates FTD likelihood from MRI.
- Month_Probability: Estimates disease progression (month-wise).
- Symptoms: Symptom-to-FTD relevance score dictionary.
- FTD_Score: Score from combining symptoms and MRI probability.

NEVER reveal internal values like FTD score, MRI or month scores. DO NOT give definitive diagnosis.

Threshold: {thresh}
Symptoms: {symptoms}
FTD_Score: {FTD_score}
MRI Probabilities: {ftdprob}
Month Probabilities: {monthprob}
Symptom List: {symptom_list_str}
    """)
])

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Chain
if "chain" not in st.session_state:
    st.session_state.chain = LLMChain(
        llm=st.session_state.llm,
        prompt=prompt,
        memory=st.session_state.memory,
        verbose=False
    )

# UI
st.set_page_config(page_title="FTD Chatbot", page_icon="🤖")
st.title("🤖 FrontoTemporal Dementia Helper Chatbot")
st.subheader("Choose how you want to start")

user_choice = st.radio("Select an option:", ["Upload an MRI", "Just continue with the symptoms"], index=1)
if user_choice == "Upload an MRI":
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file and not st.session_state.mrichecked:
        prediction = predictftd(uploaded_file)
        mprediction = predictmonth(uploaded_file)
        st.session_state.ftdprob = {"Patient": float(prediction[1]), "Non_Patient": float(prediction[0])}
        st.session_state.monthprob = {
            "M0": float(mprediction[0]),
            "M12": float(mprediction[1]),
            "M18": float(mprediction[2]),
            "M6": float(mprediction[3])
        }
        st.session_state.mrichecked = True
        st.success("MRI analysis completed.")
    elif not uploaded_file:
        st.stop()

# Chat input
user_input = st.chat_input("Describe your symptom or ask a question...")

# Score calculator
def get_score(mri_prob, symptom_scores):
    alpha, beta = 0.4, 0.6
    avg_symptom_score = sum(symptom_scores) / len(symptom_scores) if symptom_scores else 0
    return alpha * mri_prob + beta * avg_symptom_score

# Chatbot interaction
if user_input:
    sympret = symptomtest(user_input)
    if sympret == "NONE":
        st.session_state.symptom.append(f"{user_input}, Symptom not related to FTD")
    elif sympret != "UN":
        symL = sympret.split(",")
        for i in symL:
            st.session_state.symptom.append(i)
            st.session_state.svalues.append(float(i.split(":")[1]))

    FTD_score = get_score(st.session_state.ftdprob["Patient"], st.session_state.svalues)

    with st.chat_message("user"):
        st.markdown(user_input)

    response = st.session_state.chain.invoke({
        "input": user_input,
        "thresh": st.session_state.thresh,
        "symptoms": st.session_state.symptom,
        "FTD_score": FTD_score,
        "ftdprob": st.session_state.ftdprob,
        "monthprob": st.session_state.monthprob,
        "symptom_list_str": st.session_state.symptom_list_str
    })

    with st.chat_message("assistant"):
        st.markdown(response["text"])
