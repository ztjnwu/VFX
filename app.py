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
import shutil
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

    #Communicate with stability AI
    #multiple_thread(text)
    
    message = "RESULT"

    #Zip result
    shutil.make_archive('./output/result', 'zip', root_dir = './static/')
    print(message)
    return render_template('index.html', message = message)


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
