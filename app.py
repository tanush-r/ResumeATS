import os
import tempfile
from apikeys import openai_api_key

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.document_loaders import PyPDFLoader


# Prompt templates

languages_prompt_template = """
You are a Programming Language Name Extractor.
You will extract names of the programming languages and frameworks from the given resume. 
Only provide the names. Do not add any other text message, only the names are required.
If no languages are present, simply say "None"
Below is the given resume: 
{content}
"""

projects_prompt_template = """
You are projects for Resume Extractor.
You will extract the title and content of projects from the given resume. 
Only provide the project details. Do not add any other text message, only the details are required.
If no projects are present, simply say "None"
Below is the given resume: 
{content}
"""

internships_prompt_template = """
You are the Internships from Resume Extractor.
You will extract the internships/work experiences from the given resume. 
Only provide the internship details. Do not add any other text message, only the details are required.
Provide the company name and work details(not exceeding 3 sentences)
If no internships are present, simply say "None"
Below is the given resume: 
{content}
"""

certifications_prompt_template = """
You are the certifications from Resume Extractor.
You will extract the certifications from the given resume. 
Only provide the certification details. Do not add any other text message, only the details are required.
If no certifications are present, simply say "None"
Below is the given resume: 
{content}
"""

achievements_prompt_template = """
You are the Events & Achievements from Resume Extractor.
You will extract the Events & Achievements from the given resume. 
Only provide the details. Do not add any other text message, only the details are required.
If no Events & Achievements are present, simply say "None"
Below is the given resume: 
{content}
"""

resume_prompt_template = """
You are a Resume ATS system for {job_title} Role.
You will need to rate this resume I give from 0 to 10 with 2 decimal places, on the basis of how much the resume fits for given role.
Remember to be harsh and not forgiving in your rating
Take everything into consideration, especially the Job Role.
If the skills are not relevant to the job role, it should not have as much weightage as if it was relevant.
Only provide the numeric rating as the response. Do not add any other text message, only the rating is required.
The priority in considerations are: Internships >>> Projects >>> Events and achievements > Certifications > Languages
Rate the resume according to this priority. Show more bias towards internships and projects.

Below is the given resume: 
Languages: {languages}
Projects: {projects}
Internships: {internships}
Certifications: {certifications}
Events and Achievements: {achievements}
"""

languages_template = PromptTemplate(
    input_variables=['content'],
    template=languages_prompt_template
)
projects_template = PromptTemplate(
    input_variables=['content'],
    template=projects_prompt_template
)
internships_template = PromptTemplate(
    input_variables=['content'],
    template=internships_prompt_template
)
certifications_template = PromptTemplate(
    input_variables=['content'],
    template=certifications_prompt_template
)
achievements_template = PromptTemplate(
    input_variables=['content'],
    template=achievements_prompt_template
)
resume_template = PromptTemplate(
    input_variables=['job_title', 'languages', 'projects', 'internships', 'certifications', 'achievements'],
    template=resume_prompt_template
)

os.environ['OPENAI_API_KEY'] = openai_api_key

resumes_pdf = st.sidebar.file_uploader("Upload your resumes", accept_multiple_files=True, type=["pdf"])

job_title = st.sidebar.text_input("Enter Job Title")

rank_button = st.sidebar.button(label="Rank Resumes")

st.title("Resume ATS")

if resumes_pdf and job_title and rank_button:
    # App framework
    resumes = []
    for each_resume in resumes_pdf:
        st.write("Resume No :", len(resumes)+1)
        st.write(each_resume.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(each_resume.getvalue())
            file_path = temp_file.name

        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        # Extract text from each page
        content = ""
        for page in pages:
            content += page.page_content + '\n\n'

        st.write("Feature extraction")
        # Llms
        llm = OpenAI(temperature=0.2)

        languages_chain = LLMChain(llm=llm, prompt=languages_template, verbose=True, output_key='resume')
        projects_chain = LLMChain(llm=llm, prompt=projects_template, verbose=True, output_key='resume')
        internships_chain = LLMChain(llm=llm, prompt=internships_template, verbose=True, output_key='resume')
        certifications_chain = LLMChain(llm=llm, prompt=certifications_template, verbose=True, output_key='resume')
        achievements_chain = LLMChain(llm=llm, prompt=achievements_template, verbose=True, output_key='resume')

        resume_rank_chain = LLMChain(llm=llm, prompt=resume_template, verbose=True, output_key='resume')

        languages = languages_chain.run(content)
        st.write("Programming Languages: ", languages)

        projects = projects_chain.run(content)
        st.write("Projects: ", projects)

        internships = internships_chain.run(content)
        st.write("Internships: ", internships)

        certifications = certifications_chain.run(content)
        st.write("Certifications: ", certifications)

        achievements = achievements_chain.run(content)
        st.write("Achievements: ", achievements)

        resume_rank = resume_rank_chain.run(
            job_title=job_title,
            languages=languages,
            projects=projects,
            internships=internships,
            certifications=certifications,
            achievements=achievements)
        st.write("Final score: ", resume_rank)

        st.write("\n\n\n")
        resumes.append((each_resume.name, resume_rank))

    # Sort ranks
    for i in range(len(resumes) - 1):
        for j in range(i+1, len(resumes)):
            if resumes[i][1] < resumes[j][1]:
                resumes[i], resumes[j] = resumes[j], resumes[i]

    # Display output
    st.sidebar.write("Ranks: ")
    for count, resume in enumerate(resumes):
        st.sidebar.write(count+1, resume[0], resume[1])
