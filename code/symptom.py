from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json

# Load environment variables (for GROQ_API_KEY)

# Define the symptom dictionary
def symptomtest(user_input):
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    with open("../symptoms.json", "r") as s:
        symptom_dict = json.loads(s.read())

    # Flatten dictionary to symptom:value string list
    symptomlist = []
    for category, symptoms in symptom_dict.items():
        for symptom, value in symptoms.items():
            symptomlist.append(f"{symptom}:{value}")

    symptom_list_str = ", ".join(symptomlist)

    # Initialize the LLM
    llm = ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a highly accurate medical language parser.

            The user will describe symptoms in layman terms. Your job is to interpret the meaning of their input and map it to the closest matching symptom from the list below.

            These listed symptoms include definitions and example phrases to help you match based on meaning, even if the exact wording is different.

            **Your rules:**
            - Match the user's input to one or more symptoms from the list based on meaning, even if the user's phrasing is different.
            - You must return each matched symptom exactly as shown below, followed by its corresponding value.
            - If multiple symptoms match, return them as a string in this exact format: "Symptom Name:value".
            - If no symptoms match meaningfully, return "NONE".
            - If the user message is not medically relevant, return "UN".

            Symptom List:
            {symptom_list_str}

            Return only the string with the matched symptom and value, as per the format above — nothing else.
            Give "," separated multiple values ONLY when the user gives multiple symptoms.
            """

        ),
        ("human", "{input}")
    ])

    # Format and send prompt to LLM
    messages = prompt.format_messages(input=user_input)
    response = llm.invoke(messages)

    return response.content

# print(symptomtest("difficulty in trying to learn"))