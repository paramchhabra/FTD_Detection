from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os
from symptom import symptomtest
import json
from ftdmri import predictftd, predictmonth
import tensorflow as tf

# Load environment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Session state
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
    symptom_path = os.path.join(os.path.dirname(__file__), "..", "symptoms.json")
    with open(symptom_path, "r") as s:
        st.session_state.symptom_dict = json.loads(s.read())

if "symptom_list_str" not in st.session_state:
    symptomlist = []
    for category, symptoms in st.session_state.symptom_dict.items():
        for symptom, value in symptoms.items():
            symptomlist.append(f"{symptom}:{value}")
    st.session_state.symptom_list_str = ", ".join(symptomlist)

# Load LLM
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

# Prompt
if "prompt" not in st.session_state:
    st.session_state.prompt = ChatPromptTemplate.from_template("""
    You are a medical expert specializing in Frontotemporal Dementia (FTD).
    Your role is to assist in the clinical interpretation of FTD based on MRI findings and symptom profiles.

    MRI_FTD_Probability: Indicates FTD likelihood from MRI.
    Month_Probability: Estimates disease progression (month-wise).
    Symptoms: Symptom-to-FTD relevance score dictionary.
    FTD_Score: Score from combining symptoms and MRI probability.
    Threshold: {thresh}
    Symptoms: {symptoms}
    FTD_Score: {FTD_score}
    MRI Probabilities: {ftdprob}
    Month Probabilities: {monthprob}
    
    Symptom List (only use to help collect more symptoms if user gives very little info):
    {symptom_list_str}

    Important Guidelines:
    - DO NOT reveal internal values like FTD score, MRI or month scores.
    - Only give suggestions (e.g., "consider seeing a neurologist") when symptoms + score are serious.
    - Never give a definitive diagnosis.
    - Speak with empathy and sensitivity.
    
    Human: {input}
    """)

# Memory setup
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Create ConversationChain
if "chain" not in st.session_state:
    st.session_state.chain = ConversationChain(
        llm=st.session_state.llm,
        prompt=st.session_state.prompt,
        memory=st.session_state.memory,
        verbose=False
    )

# UI setup
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
        st.session_state.monthprob = {"M0": float(mprediction[0]), "M12": float(mprediction[1]),
                                      "M18": float(mprediction[2]), "M6": float(mprediction[3])}
        st.session_state.mrichecked = True
        st.success("MRI analysis completed.")
    elif not uploaded_file:
        st.stop()

# Chat input
user_input = st.chat_input("Describe your symptom or ask a question...")

# Scoring function
def get_score(mri_prob, symptom_scores):
    avgsymp = sum(symptom_scores) / len(symptom_scores) if symptom_scores else 0
    alpha = 0.4
    beta = 0.6
    return alpha * mri_prob + beta * avgsymp

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

    response = st.session_state.chain.predict(
        input=user_input,
        thresh=st.session_state.thresh,
        symptoms=st.session_state.symptom,
        FTD_score=FTD_score,
        ftdprob=st.session_state.ftdprob,
        monthprob=st.session_state.monthprob,
        symptom_list_str=st.session_state.symptom_list_str
    )

    with st.chat_message("assistant"):
        st.markdown(response)
