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
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
SECRET_KEY = os.getenv('OPENAI_KEY')

data = ""
highlighted_content = ""
flag = 0

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global flag

    # Flag is used to manage the page refresh, when we refresh page
    # flag = 0 --> Computation is done
    # flag = 1 --> Computation is not done and only results is displayed
    flag = 0

    # Get the file from the form
    file = request.files['file']

    # Get textbox from the form
    text = request.form.get('textbox')

    # Checking if content is available for checking
    if not file and not text:
        message = "Error! No file or text provided"
        return render_template('alert.html', message=message)

    # If file is uploaded
    if file:
        filename = file.filename
        old_path = os.path.join(app.static_folder, filename)
        doc_path = os.path.join(app.static_folder, 'file1.docx')
        text_path = os.path.join(app.static_folder, 'file1.txt')

        if os.path.isfile(doc_path):
            os.remove(doc_path)
        if os.path.isfile(text_path):
            os.remove(text_path)

        # Save the file to the static folder
        file.save(os.path.join(app.static_folder, filename))

        # Getting the file extension
        file_extension = os.path.splitext(filename)[1]

        # Renaming file based on the file extension
        if file_extension == ".docx":
            new_filename = r'file1.docx'

        elif file_extension == ".txt":
            new_filename = r'file1.txt'

        else:
            message = 'File format not accepted. Choose .txt or .docx file'
            os.remove(old_path)
            return render_template('alert.html', message=message)

        global new_path
        new_path = os.path.join(app.static_folder, new_filename)
        os.rename(old_path, new_path)

        if file_extension == ".docx":
            new_filename = r'file1.docx'

            # Set the path of the docx file and the output txt file
            docx_file_path = os.path.join('static', 'file1.docx')
            txt_file_path = os.path.join('static', 'file1.txt')

            # Open the docx file and extract its text
            doc = Document(docx_file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

            # Save the text to a txt file with UTF-8 encoding
            with open(txt_file_path, 'w', errors='ignore') as txt_file:
                txt_file.write(text)

        # Counting the number of characters in file
        with open("static/file1.txt", "r") as file:
            content = file.read()
            num_chars = len(content)
            if num_chars > 2048:
                message = 'File size is too big! Upload a file which has 2000 or less characters'
                return render_template('alert.html', message=message)

        message = 'File upload successful!\n\nWait while we calculate the Plagiarism result.'
        return render_template('alerts.html', message=message)

    else:
        # If file1.txt exist open it else create a file1.txt
        with open('static/file1.txt', 'w') as f:
            f.write(str(text))
            message = "Content submittion successful!\n\nWait while we calculate the Plagiarism result."
            return render_template('alerts.html', message=message)


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
