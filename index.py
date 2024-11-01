from flask import Flask, request, render_template
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

app = Flask(__name__)

# Set up Hugging Face API
hf_key = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key

# LangChain setup
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, model_kwargs={"max_length": 2048})

# Define the biography prompt template
template="""
You are an expert in biographies. Your job is to only write biographies for famous celebrities.
    If the input text does not match a well-known celebrity, respond with "Not a valid celebrity input."

    Input: {input_text}

    Instructions:
    - If the input text matches a celebrity's name, write a detailed biography including:
      1. Early life and family background
      2. Career beginnings and rise to fame
      3. Key achievements and awards
      4. Notable works or contributions
      5. Personal life and any philanthropic efforts
      write all this in single paragraph continuous.
    - If the input text does not match a celebrity's name, respond with "Not a valid celebrity input."
    - dont leave content incomplete. if you getting out of word limits, shrink content to fit.
    Biography:
"""

prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template=template
)

biography_chain = LLMChain(prompt=prompt_template, llm=llm)




@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["input_text"]
        biography = biography_chain.invoke(name)
        print(biography['text'])
        if "Not a valid celebrity input" in biography['text'] or "I'll assume the celebrity" in biography['text']:
            biography['text'] = "Ask me about celebrities only."    
        return render_template("index.html", name=biography['input_text'], biography=biography['text'],placeholder_text=name)
    return render_template("index.html",placeholder_text="Enter a celebrity name...")

if __name__ == "__main__":
    app.run(debug=True)
