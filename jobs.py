import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from textwrap import dedent
import os
from functools import lru_cache

@st.cache_data
def chatgpt4_response(prompt):
    """
    Connects to ChatGPT-4 and returns the response for the given message.

    Args:
    message (str): The message to send to ChatGPT-4.

    Returns:
    str: The response from ChatGPT-4.

    Raises:
    Exception: If there is an error connecting to the OpenAI API.
    """
    client = OpenAI(api_key=st.session_state.llm_pwd)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an assistant who is eager to answer all kinds of questions, and your task is to provide users with professional, accurate, and insightful advice. Additionally, you possess expertise in job specialization."},
                {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error connecting to OpenAI API: {str(e)}")
        return ""

@st.cache_data
def gemini(prompt, web_search=False, top_p=0.7, temperature=0.1, max_tokens=8192, model_name="gemini-1.5-flash"):
    """
    Generates a response using the Gemini model.

    Args:
    prompt (str): The input prompt.
    web_search (bool): Whether to use web search (default: False).
    top_p (float): Top-p sampling parameter (default: 0.7).
    temperature (float): Temperature for sampling (default: 0.1).
    max_tokens (int): Maximum number of tokens to generate (default: 8192).
    model_name (str): Name of the Gemini model to use (default: "gemini-1.5-flash").

    Returns:
    str: The generated response.

    Raises:
    Exception: If there is an error connecting to the Gemini API.
    """
    try:
        genai.configure(api_key=st.session_state.llm_pwd)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 64,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }

        llm = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction="You are an assistant who is eager to answer all kinds of questions, and your task is to provide users with professional, accurate, and insightful advice. Additionally, you possess expertise in job specialization.",
        )
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error connecting to Gemini API: {str(e)}")
        return ""

@st.cache_data
def perplexity(prompt, web_search=False, top_p=0.7, temperature=0.1, max_tokens=1024):
    """
    Generates a response using the Perplexity model.

    Args:
    prompt (str): The input prompt.
    web_search (bool): Whether to use web search (default: False).
    top_p (float): Top-p sampling parameter (default: 0.7).
    temperature (float): Temperature for sampling (default: 0.1).
    max_tokens (int): Maximum number of tokens to generate (default: 1024).

    Returns:
    str: The generated response.

    Raises:
    Exception: If there is an error connecting to the Perplexity API.
    """
    try:
        llm = OpenAI(api_key=st.session_state.llm_pwd, base_url="https://api.perplexity.ai")
        messages = [
            {
                "role": "system",
                "content": "You are an assistant who is eager to answer all kinds of questions, and your task is to provide users with professional, accurate, and insightful advice. Additionally, you possess expertise in job specialization."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = llm.chat.completions.create(
            model="llama-3-sonar-large-32k-online",
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error connecting to Perplexity API: {str(e)}")
        return ""

def get_response(prompt):
    """
    Gets a response from the selected LLM.

    Args:
    prompt (str): The input prompt.

    Returns:
    str: The generated response.
    """

    if not st.session_state.llm_pwd:
        st.error("Please enter an API key.")
        return ""

    llm_functions = {
        "Gemini": gemini,
        "OpenAI": chatgpt4_response,
        "Perplexity": perplexity
    }

    llm_function = llm_functions.get(st.session_state.llm)
    if llm_function:
        return llm_function(prompt=prompt)
    else:
        st.error("Invalid LLM selected.")
        return ""

# Main Streamlit app
st.set_page_config(page_title="Cover letter", page_icon="🐉", initial_sidebar_state="expanded", layout="wide")
st.title("🐉 Job Application - Cover letter 🐉")

# Initialize session state variables
for key in ["llm", "llm_pwd", "txt_analyst"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "llm" else "Gemini"

# Sidebar
with st.sidebar:
    st.session_state.llm = st.selectbox("Select a LLM", ("Gemini", "OpenAI", "Perplexity"))
    st.session_state.llm_pwd = st.text_input(label="API Key:", type="password")
    st.write("---")
    st.write(f"""Get Gemini API key here:{os.linesep}https://aistudio.google.com/app/prompts/new_chat{os.linesep}
Get your OpenAI API key here:{os.linesep}https://platform.openai.com{os.linesep}
Get you Perplexity API Key here:{os.linesep}
Login to your account(https://perplexity.ai), navigate to the settings/account->API tab->Generate
        """)

# Input fields
txt_resume = st.text_area(label="Resume")
txt_company = st.text_input(label="Company Name")
txt_company_address = st.text_area(label="Company Address")
txt_job_title = st.text_input(label="Job Title")
txt_job_description = st.text_area(label="Job Description")

prompt = dedent(f"""
    Resume:
    <resume>{txt_resume}</resume>
    <Company>Company: {txt_company}</Company>
    <Company Address>Company Address: {txt_company_address}</Company Address>
    <Job Title>Job Title: {txt_job_title}</Job Title>
    <Job Description>Job Description:
    {txt_job_description}
    </Job Description>
""")

with st.expander(label="Input details", expanded=False):
    st.code(prompt, language=None)

p1 = f"Write a breakdown of resume and its alignment with the {txt_job_title} position at {txt_company}. Assign an initial score on a scale of 1 to 10 for your assessment."

if st.button(label="Analyze"):
    st.session_state.txt_analyst = get_response(prompt=f"{p1}{os.linesep}{prompt}")
    with st.expander(label="Analysis", expanded=True):
        st.write(st.session_state.txt_analyst)

if st.button(label="Generate cover letter"):
    if st.session_state.txt_analyst:
        p2 = f"Create a cover letter for job application based on the information: {prompt} and your analysis: {st.session_state.txt_analyst}. Use the information provided and nothing else"
        with st.expander(label="Analysis"):
            st.write(st.session_state.txt_analyst)
        st.write("Cover Letter:" + os.linesep + get_response(prompt=f"{p2}{os.linesep}{prompt}"))
    else:
        st.warning("Please run the analysis first")
