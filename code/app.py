from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thresh" not in st.session_state:
    st.session_state.thresh = 0.6

# Load and store symptoms.json in session state
if "symptom_dict" not in st.session_state:
    with open("..\\symptoms.json", "r") as s:
        st.session_state.symptom_dict = json.loads(s.read())

if "symptom_list_str" not in st.session_state:
    symptomlist = []
    for category, symptoms in st.session_state.symptom_dict.items():
        for symptom, value in symptoms.items():
            symptomlist.append(f"{symptom}:{value}")
    st.session_state.symptom_list_str = ", ".join(symptomlist)

# Load LLM and prompt only once
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)

if "prompt" not in st.session_state:
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """You are a medical expert specializing in Frontotemporal Dementia (FTD).
        Your role is to assist in the clinical interpretation of FTD based on MRI findings and symptom profiles.

        Inputs You May Receive:
        MRI_FTD_Probability: A float value between 0 and 1 indicating the likelihood of FTD based on MRI scan analysis.

        Month_Probability: A mapping of months (e.g., Month 1 to Month 12) to the probability that the MRI scan corresponds to that specific disease month, assuming FTD is present.
        If the confidence in these estimates is low or ambiguous, do not communicate this to the user, as it may cause unnecessary fear or confusion.

        Symptoms: A dictionary where each symptom is labeled with its relevance, or specificity to FTD.
        Example:"Impaired_Repetition": 0.2,"Impaired_Naming": 0.4 (with higher value showing closeness to FTD),
        If most symptoms are marked as "Not related", inform the user that their condition likely does not correspond to FTD.
        The Current Symptoms uploaded by the user are : {symptoms}

        FTD_Score: A numerical score derived from combining symptom relevance and MRI probability. 
        A threshold value {thresh} is used to determine whether the case should be flagged for further evaluation.
        The Current Score is {FTD_score} and changes with every symptom given by the user.
        MAKE SURE YOU GET ALL THE SYMPTOMS FROM THE USER BEFORE MAKING A VERDICT OR SUGGESTION AS EVERY SYMPTOM CHANGES THE SCORE

        Guidelines:
        Do not return the Backend values, like the ftd scores, month scores or the symptom scores. They are only for you to understand the user better. Do not reveal them to the user.
        Use the inputs to estimate the likelihood of FTD, but never provide a definitive diagnosis.
        If The symptoms feel serious and recurring, FTD_Score are high and symptoms are relevant to FTD, suggest visiting a neurologist or specialist for further evaluation.
        If FTD Score is low and symptoms are mostly unrelated, suggest exploring alternative medical explanations.
        If the month prediction is present and confident (as in High FTD_Score), you may mention it cautiously — only if you believe the user would benefit medically from this information.

        Always be medically precise, empathetic, and cautious. Recommend clinical evaluation when in doubt.
        Important: Avoid alarming the user. Speak with sensitivity and always prioritize suggesting professional follow-up over giving a conclusion.

        Here is the MRI and montly probabilities. If they are all 0 that means the user wants to just classify based on the symptoms. 
        Only use these values as a reference when the user askes you to.
        {ftdprob} [FTD Prediction by mri]
        {monthprob} [Shows how many months can the user be into the disease, only to be used if the above probability shows some certainity]

        Finally, I am also providing you with the symptom list, but ONLY use it to ask questions if the user is 
        not providing enough symptoms or is confused, to collect more symptoms:
        {symptom_list_str}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

if "chain" not in st.session_state:
    st.session_state.chain = LLMChain(llm=st.session_state.llm, prompt=st.session_state.prompt)

# Streamlit UI
st.set_page_config(page_title="GenZ Chatbot", page_icon="🤖")
st.title("🤖 GenZ Chatbot")
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

# Display previous messages
for msg in st.session_state.chat_history:
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
    print(FTD_score)
    print(st.session_state.ftdprob)
    print(st.session_state.monthprob)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    response = st.session_state.chain.invoke({
        "symptoms": st.session_state.symptom,
        "thresh": st.session_state.thresh,
        "FTD_score": FTD_score,
        "ftdprob": st.session_state.ftdprob,
        "monthprob": st.session_state.monthprob,
        "symptom_list_str": st.session_state.symptom_list_str,
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    output = response['text']

    st.session_state.chat_history.append(AIMessage(content=output))
    with st.chat_message("assistant"):
        st.markdown(output)
