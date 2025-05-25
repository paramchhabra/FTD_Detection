from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import streamlit as st
import os
from symptom import symptomtest
import json
from ftdmri import predictftd, predictmonth
import tensorflow as tf

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initial session state setup
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

# Load and store symptoms.json in session state
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

# Setup memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a medical expert specializing in Frontotemporal Dementia (FTD).
Your role is to assist in the clinical interpretation of FTD based on MRI findings and symptom profiles.
Inputs You May Receive:
MRI_FTD_Probability: A float value between 0 and 1 indicating the likelihood of FTD based on MRI scan analysis.
Month_Probability: A mapping of months (e.g., Month 1 to Month 12) to the probability that the MRI scan corresponds to that specific disease month.
Symptoms: A dictionary where each symptom is labeled with its relevance, or specificity to FTD.
The Current Symptoms uploaded by the user are: {symptoms}
FTD_Score: The Current Score is {FTD_score}.
A threshold value {thresh} is used to determine whether the case should be flagged for further evaluation.

Guidelines:
Do not reveal backend scores directly. Use them only for internal logic.
If most symptoms are unrelated, reassure the user.
If the score is high and symptoms are relevant, suggest professional follow-up.
You may use monthprob if confidence is high and FTD is likely.

Only use this list to help gather more symptoms if the user is confused:
{symptom_list_str}"""),
    ("human", "{input}")
])

# Create conversation chain
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = ConversationChain(
        llm=st.session_state.llm,
        prompt=prompt,
        memory=st.session_state.memory,
        input_variables=["input", "symptoms", "FTD_score", "thresh", "symptom_list_str"],
        verbose=False
    )

# Streamlit UI
st.set_page_config(page_title="FTD Chatbot", page_icon="🤖")
st.title("🤖 FrontoTemporal Dementia Helper Chatbot")
st.subheader("Choose how you want to start")

user_choice = st.radio("Select an option:", ["Upload an MRI", "Just continue with the symptoms"], index=1)
if user_choice == "Upload an MRI":
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file and not st.session_state.mrichecked:
        prediction = predictftd(uploaded_file)
        mprediction = predictmonth(uploaded_file)

        st.session_state.ftdprob = {
            "Patient": float(prediction[1]),
            "Non_Patient": float(prediction[0])
        }
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

# Display memory messages
for msg in st.session_state.memory.chat_memory.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# User input
user_input = st.chat_input("Describe your symptom or ask a question...")

# Score function
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

    # Get response
    response = st.session_state.conversation_chain.predict(
        input=user_input,
        symptoms=st.session_state.symptom,
        thresh=st.session_state.thresh,
        FTD_score=FTD_score,
        symptom_list_str=st.session_state.symptom_list_str
    )

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
