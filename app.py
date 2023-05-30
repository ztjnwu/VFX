from flask import Flask, render_template, request
import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from docx import Document
import re
import requests
from dotenv import load_dotenv


#
import os
import json
import time
import base64
import openai
from PIL import Image, ImageFont, ImageDraw 
import threading
#



#
app = Flask(__name__)

#
load_dotenv()
SECRET_KEY = os.getenv('OPENAI_KEY')

data = ""
highlighted_content = ""
flag = 0



#
#Initialization stability AI
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
url = f"{api_host}/v1/user/account"
api_key = "sk-JpvcoRP61W8M9wJij8GG9EcvpqD0GOHxmUIMP2tRKT7xeWV9"
engine_id = "stable-diffusion-v1-5"

#initliazaile openAI AI
openai.api_key = "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"
#

#New Feature2
def get_explanation(text):
    #build a prompt
    prompt = f"'{text}'"
    
    #call to openAI API
    response = openai.Completion.create(engine = 'text-davinci-003',
                                        prompt = prompt,
                                        max_tokens = 200, 
                                        temperature = 0.5,
                                        top_p = 1,
                                        frequency_penalty = 0,
                                        presence_penalty = 0
                                        )

    #
    explanation = response.choices[0].text.strip()
    
    return(explanation)


#New Feature2
def get_image(text, k):
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers = 
        {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
        },
        
        json = {
                "text_prompts": [
                    {
                        "text": text
                    }
                ],
                "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 30,
        },
    
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    
    #Return data
    data = response.json()

    #Parse data
    path = ""
    for i, image in enumerate(data["artifacts"]):
        with open(f"/workspace/VFX/static/v1_txt2img_{k}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))
            path = f"/workspace/VFX/static/v1_txt2img_{k}.png"
    
    #Return
    return(path)


#New Feature3
def multiple_thread(text):
    # creating thread
    t1 = threading.Thread(target = get_image, args=(text, 1))
    t2 = threading.Thread(target = get_image, args=(text, 2))
    t3 = threading.Thread(target = get_image, args=(text, 3))
    t4 = threading.Thread(target = get_image, args=(text, 4))
 
    # starting thread 1
    t1.start()
    t2.start()
    t3.start()
    t4.start()
 
    # wait until thread 1 is completely executed
    t1.join()
    t2.join()
    t3.join()
    t4.join()
 
    # both threads completely executed
    print("Done!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Get textbox from the form
    text = request.form.get('textbox')

    #Call OpenAI
    text = get_explanation(text)
    print(text)

    #Communicate with stability AI
    multiple_thread(text)
    
    message = "RESULT"
    return render_template('index.html', message = message)


@app.route("/check")
def check():
    global data
    global highlighted_content

    sample_files = [doc for doc in os.listdir(app.static_folder) if doc.endswith('.txt')]

    # Read all files into a list
    files_content = []
    for sample_file in sample_files:
        with open(os.path.join(app.static_folder, sample_file)) as f:
            files_content.append(f.read())

    # Define the vectorizer using the entire corpus of texts
    vectorizer = TfidfVectorizer()
    vectorizer.fit(files_content)

    # Get the content of file1.txt
    with open(os.path.join(app.static_folder, 'file1.txt')) as file1:
        file1_content = file1.read()

    plagiarism_result = set()

    # Compare file1.txt with all other files
    for sample_b in sample_files:
        # Skip file1.txt itself
        if sample_b == 'file1.txt':
            continue

        with open(os.path.join(app.static_folder, sample_b)) as file2:
            file2_content = file2.read()

        # Transform each individual text using the same vectorizer
        file1_vec = vectorizer.transform([file1_content])
        file2_vec = vectorizer.transform([file2_content])

        # Compute similarity score between file1.txt and the current file
        similarity_score = round(cosine_similarity(file1_vec, file2_vec)[0][0], 4)

        # Store the score and file names in the plagiarism_result set
        sample_pair = sorted(('file1.txt', sample_b))
        score = sample_pair[0], sample_pair[1], similarity_score
        plagiarism_result.add(score)

    # Get the max score and the files with the max score
    max_score_val = max(plagiarism_result, key=lambda x: x[2])[2]
    max_files = [(t[0], t[1]) for t in plagiarism_result if t[2] == max_score_val]

    with open(os.path.join(app.static_folder, max_files[0][1])) as max_file:
        max_file_content = max_file.read()

    # Read max_file_content and highlight matching content in file1.txt
    with open(os.path.join(app.static_folder, 'file1.txt')) as file1:
        file1_content = file1.read()
        highlighted_content = ""
        for line in file1_content.splitlines():
            highlighted_line = line
            for match in set(max_file_content.split()) & set(line.split()):
                highlighted_line = re.sub(r'\b{}\b'.format(match), '<mark style="background-color:rgba(255, 0, 0, 0.5);">{}</mark>'.format(match), highlighted_line)
            highlighted_content += highlighted_line + "<br>"

    plagiarised_percent = round(max_score_val, 2) * 100 + 30
    if plagiarised_percent >= 100:
        plagiarised_percent = 100
    elif plagiarised_percent <= 50:
        plagiarised_percent -= 30
    non_plagiarised_percent = 100 - plagiarised_percent
    data = [plagiarised_percent, non_plagiarised_percent]

    return (data, highlighted_content)


@app.route("/chart")
def chart():
    global data
    global highlighted_content
    global flag

    n = 2

    # Checking flag so that the function calls are ot done while page is being refreshed
    if flag == 1:        
        return render_template("chart.html", data=data, content=highlighted_content)
    else: 
        # Removing the files created by chatGPT
        for i in range(0, n):
            file_name = f"chatGPT_file{i+1}.txt"
            file_path = os.path.join(app.static_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        new_path = os.path.join(app.static_folder, 'file1.txt')

        #Read the uploaded file in the fold of static
        with open(new_path, 'r') as file:
            text = file.read()

        #invoke openAI api
        openai.api_key = SECRET_KEY
        
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # The first sentence is usually the title
        title = sentences[0]

        # Pass the title as the prompt to OpenAI
        prompt = f"Write an essay on '{title}'."
        
        # Call the completion API and get the response
        for i in range(0, n):
            if i > 0:
                prompt = f"Write an essay on '{title}, give a different content for this'."

            var_name = f"var{i+1}"

            response = openai.Completion.create(engine = 'text-davinci-003',
                                                prompt = prompt,
                                                max_tokens = 3000,
                                                temperature = 0.5,
                                                top_p = 1,
                                                frequency_penalty = 0,
                                                presence_penalty = 0
                                                )

            var_name = response.choices[0].text.strip()
            file_name = f"chatGPT_file{i+1}.txt"
            file_path = 'static/' + file_name
            with open(file_path, 'w') as f:
                f.write(str(var_name))
        data, highlighted_content = check()

        flag = 1

        return render_template("chart.html", data=data, content=highlighted_content)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/terms')
def terms():
    return render_template('terms_and_conditions.html')


@app.route("/teamcot")
def teamcot():
    return render_template("teamcot.html")


@app.route("/cot")
def cot():
    return render_template("cot.html")


@app.route("/aboutplagiarism")
def aboutplagiarism():
    return render_template("aboutplagiarism.html")

if __name__ == '__main__':
    app.run(debug=True)
