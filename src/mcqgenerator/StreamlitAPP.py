import os
import json
import traceback
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from mcqgenerator.utils import read_file, get_table_data
from mcqgenerator.logger import logging
from mcqgenerator.MCQgenerator import chain


# load environment variables
load_dotenv()

st.title("MCQs Creator Application with Groq + Llama3 🚀")


with st.form("user_inputs"):

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT file"
    )

    mcq_count = st.number_input(
        "Number of MCQs",
        min_value=3,
        max_value=50,
        value=5
    )

    subject = st.text_input(
        "Subject",
        max_chars=50
    )

    tone = st.text_input(
        "Difficulty Level",
        placeholder="Simple"
    )

    button = st.form_submit_button("Generate MCQs")


if button and uploaded_file is not None:

    with st.spinner("Generating questions..."):

        try:

            text = read_file(uploaded_file)

            response = chain.invoke({
                "text": text,
                "number": mcq_count,
                "subject": subject,
                "tone": tone
            })


        except Exception as e:

            traceback.print_exception(type(e), e, e.__traceback__)

            st.error("Error generating MCQs")


        else:

            if isinstance(response, dict):

                quiz = response.get("quiz")

                if quiz:

                    table_data = get_table_data(quiz)

                    df = pd.DataFrame(table_data)

                    df.index = df.index + 1

                    st.subheader("Generated MCQs")

                    st.table(df)


                st.subheader("Review")

                st.write(
                    response.get(
                        "review",
                        "No review generated"
                    )
                )

            else:

                # fallback if JSON parsing fails

                st.subheader("Generated Output")

                st.write(response)