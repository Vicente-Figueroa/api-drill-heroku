import boto3
import pickle
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from flask_cors import CORS
import os
#from dotenv import load_dotenv
#load_dotenv()


key_id = os.environ['KEY_ID']
key = os.environ['KEY']

#key_id = os.getenv('KEY_ID')
#key = os.getenv('KEY')
# Configurar las credenciales de AWS
aws_region = "us-east-1"
s3_bucket = "awsbucket180624"
s3_object_key = "colab/tfidf_retriever.pkl"

# Crear un cliente de S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=key_id,
    aws_secret_access_key=key,
    region_name=aws_region
)

# Especifica el nombre del archivo local donde quieres guardar el archivo descargado
local_file_name = 'tfidf_retriever.pkl'

# Descarga el archivo desde S3
s3_client.download_file(s3_bucket, s3_object_key, local_file_name)
with open(local_file_name, 'rb') as file:
    tfidf_retriever = pickle.load(file)

# Configuración del modelo
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key='AIzaSyAtQwglcD0LvJdhKcbb2KCNjHhiSAepvqQ')
prompt = """Como un asistende de Utem hazte pasar por un asesor experimentado en atención al cliente respondiendo amablemente
y de forma concisa. Recuerda siempre proporcionar la mayor información posible y responder
la siguiente pregunta que te hace el cliente: {question} .
Y este es tu contexto: {context}
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=tfidf_retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Configurar la aplicación Flask
app = Flask(__name__)

CORS(app, resources={
    r"/items": {"origins": "*", "methods": ["POST", "OPTIONS", "GET"], "headers": ["Content-Type"]},
    r"/chat": {"origins": "*", "methods": ["POST", "OPTIONS", "GET"], "headers": ["Content-Type"]},
    r"/send-email": {"origins": "*", "methods": ["POST", "OPTIONS", "GET"], "headers": ["Content-Type"]}
})

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

@app.route('/chat', methods=['GET', 'POST', 'OPTIONS'])
def chat():
    if request.method in ['POST', 'OPTIONS']:
        data = request.get_json()
        query = data.get('query', '')
        if query:
            result = qa_chain({"query": query})
            response = result['result']
        else:
            return jsonify({"response": "Por favor, envía una consulta válida."})
    else:
        # Mensaje de bienvenida para GET
        query = "Hola, presentate"
        result = qa_chain({"query": query})
        response = result['result']

    return jsonify({'message': str(response)})

if __name__ == '__main__':
    app.run()