import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from textwrap import dedent
import os

def chatgpt4_response(message):
    """Connects to ChatGPT-4 and returns the response for the given message.

    Args:
    message: The message to send to ChatGPT-4.

    Returns:
    A string containing the response from ChatGPT-4.

    Raises:
    openai.OpenAIError: If there is an error connecting to the OpenAI API.
    openai.error.InvalidRequestError: If the API request is invalid.
    """
    client = OpenAI(
        api_key=st.session_state.llm_pwd
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except client.OpenAIError  as e:
        st.error(e)
        return ""

def gemini(web_search = False, prompt ="", top_p = 0.7, temperature = 0.1, max_tokens = 8192, model_name = "gemini-1.5-flash"):
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
        system_instruction="你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。You are also a job specialist",
    )
    responses = llm.generate_content(prompt).text
    return responses

def perplexity(web_search = False, prompt ="", top_p = 0.7, temperature = 0.1, max_tokens = 1024):
    llm = OpenAI(api_key= st.session_state.llm_pwd, base_url="https://api.perplexity.ai")
    messages = [
        {
            "role": "system",
            "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。You are also a job specialist"
        },
        {
            "role": "user",
            "content": prompt ,
        },
    ]

    responses = llm.chat.completions.create(
        model="llama-3-sonar-large-32k-online",
        messages=messages,
        top_p= top_p,
        temperature= temperature,
        max_tokens=max_tokens,
        stream=False,
    ).choices[0].message.content
    
    return responses

def get_response(prompt):
    with st.spinner("Getting response"):
        response = ""
        if st.session_state.llm_pwd == "" or st.session_state.llm_pwd is None:
            st.error("Please enter a password.")
        else:
            if st.session_state.llm == "Gemini":
                response = gemini(prompt=prompt)
            if st.session_state.llm == "OpenAI":
                response = chatgpt4_response(message=prompt)
            if st.session_state.llm == "Perplexity":
                response = perplexity(prompt=prompt)          
        return response

st.set_page_config(page_title="Cover letter", page_icon="🐉", initial_sidebar_state="expanded", layout="wide")
st.title("🐉 Job Application - Cover letter 🐉")

if "llm" not in st.session_state:
    st.session_state.llm = "Gemini"
if "llm_pwd" not in st.session_state:
    st.session_state.llm_pwd = ""
if "txt_analyst" not in st.session_state:
    txt_analyst = ""

with st.sidebar:
    st.session_state.llm = st.selectbox("Select a LLM", ("Gemini", "OpenAI", "Perplexity"))
    st.session_state.llm_pwd = st.text_input(label="API Key:", type="password")
    st.write("---")
    st.write(f"""Get Gemini API key here:{os.linesep}https://aistudio.google.com/app/prompts/new_chat{os.linesep}
Get your OpenAI API key here:{os.linesep}https://platform.openai.com{os.linesep}
Get you Perplexity API Key here:{os.linesep}
Login to your account(https://perplexity.ai), navigate to the settings/account->API tab->Generate
""")

txt_resume = st.text_area(label="Resume")
txt_company = st.text_input(label="Company Name")
txt_company_address = st.text_area(label="Company Address")
txt_job_title = st.text_input(label="Job Title")
txt_job_description = st.text_area(label="Job Description")

prompt = dedent(f"Resume:{os.linesep} <resume>{txt_resume}</resume>{os.linesep}\
    <Company>Company: {txt_company}{os.linesep}</Company> \
    <Company Address>Company Address: {txt_company_address}{os.linesep} </Company Address>\
    <Job Title>Job Title: {txt_job_title}{os.linesep}</Job Title> \
    <Job Description>Job Description: {os.linesep}{txt_job_description}{os.linesep}</Job Description> \
    "
)

#p1 =("Here's a breakdown of Sow Lee Kwan's resume and its alignment with the Desktop Support position at PERSOLKELLY. - Assign an initial score on a scale of 1 to 10."
p1 = f"Write a breakdown of resume and its alignment with the {txt_job_title} position at {txt_company}. Assign an initial score on a scale of 1 to 10 for your assessment."

with st.expander(label="Input details", expanded=False):
    st.code(prompt, language=None)



if st.button(label="Analyst"):
    st.session_state.txt_analyst = get_response(prompt=f"{p1}{os.linesep}{prompt}")
    with st.expander(label="Analysis", expanded=True):
        st.write(st.session_state.txt_analyst)

if st.button(label="Generate cover letter"):
    if st.session_state.txt_analyst != "" or st.session_state.txt_analyst is not None:
        p2 = f"Create a cover letter for job application base the information: {prompt} and your analysis: {st.session_state.txt_analyst}. Use the information provided and nothing else"
        with st.expander(label="Analyst"):
            st.write(st.session_state.txt_analyst)
        st.write("Cover Letter:" + os.linesep + get_response(prompt=f"{p2}{os.linesep}{prompt}"))
else:
    st.warning("Please run the analyst first")



