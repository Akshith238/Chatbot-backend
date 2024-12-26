from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],  # Add your frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Google API
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path):
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""  # Handle empty pages
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_docx(file_path):
        try:
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise

    @staticmethod
    def extract_text_from_txt(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {str(e)}")
            raise

    @staticmethod
    def process_file(file_path):
        try:
            file_extension = Path(file_path).suffix.lower()
            if file_extension == '.pdf':
                return DocumentProcessor.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                return DocumentProcessor.extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                return DocumentProcessor.extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None

    def process_documents(self, texts):
        try:
            # Filter out empty texts
            texts = [text for text in texts if text.strip()]
            if not texts:
                raise ValueError("No valid text content found in documents")

            # Split texts into chunks
            chunks = text_splitter.split_text("\n".join(texts))
            
            # Create vector store
            self.vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings
            )
            
            # Create QA chain
            template = """
            You are an AI assistant analyzing documents. Based on the provided context, 
            give a detailed, structured response. If you find any issues or problems, 
            highlight them clearly.

            Context: {context}
            Question: {question}
            Answer:"""
            
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True
            )
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def get_answer(self, question):
        if not self.qa_chain:
            raise ValueError("Documents haven't been processed yet")
        
        try:
            response = self.qa_chain({"query": question})
            return {
                "answer": response["result"],
                "sources": [doc.page_content for doc in response["source_documents"]]
            }
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise

# Initialize RAG system
rag_system = RAGSystem()

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            logger.error("No files in request")
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files[]')
        if not files or all(file.filename == '' for file in files):
            logger.error("No files selected")
            return jsonify({'error': 'No files selected'}), 400

        texts = []
        processed_files = []

        for file in files:
            if file.filename == '':
                continue

            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            
            try:
                # Save and process file
                file.save(temp_path)
                text = DocumentProcessor.process_file(temp_path)
                texts.append(text)
                processed_files.append(file.filename)
                logger.info(f"Successfully processed {file.filename}")
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                return jsonify({
                    'error': f'Error processing {file.filename}: {str(e)}',
                    'processed_files': processed_files
                }), 400
            finally:
                # Clean up
                try:
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary files: {str(e)}")

        # Process documents with RAG system
        try:
            rag_system.process_documents(texts)
            return jsonify({
                'message': 'Documents processed successfully',
                'processed_files': processed_files
            }), 200
        except Exception as e:
            logger.error(f"Error in RAG system: {str(e)}")
            return jsonify({
                'error': f'Error processing documents: {str(e)}',
                'processed_files': processed_files
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        question = data.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        response = rag_system.get_answer(question)
        return jsonify(response), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summary', methods=['GET'])
def get_summary():
    try:
        summary = rag_system.get_answer("Generate a comprehensive summary of all the documents.")
        return jsonify(summary), 200
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_specific():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        aspect = data.get('aspect', 'general')
        
        prompts = {
            'issues': "List all problems, issues, and concerns found in the documents.",
            'recommendations': "Provide recommendations based on the document content.",
            'general': "Analyze the key points and provide a structured analysis."
        }
        
        analysis = rag_system.get_answer(prompts.get(aspect, prompts['general']))
        return jsonify(analysis), 200
    except Exception as e:
        logger.error(f"Error in analyze_specific: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)