from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma


resume = "../data/CV.md"
with open(resume) as f:
    resume_data = f.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(resume_data)
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_texts(
    texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
)

query = "What are skills, experiences, education and abilities"
docs = docsearch.similarity_search(query)


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

bio_template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a summary of their experiences for a cover letter.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=bio_template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(
    OpenAI(temperature=0.2), chain_type="stuff", memory=memory, prompt=prompt
)

query = "What is a good discription of Evelyn BOettcher"
chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)

{'output_text': ' Evelyn Boettcher is a physicist and founder with a Master of Science in Physics from the University of Maryland and a Bachelor of Science in Physics with Highest Honors from the University of Florida. She has a wide range of technical skills, including data science, machine learning, and sensor system development. She has published two journal papers, acquired two patents, and is the head of the Machine Learning Community group. She also enjoys teaching, promoting STEM, and creating things with sewing, 3D printing, and hiking.'}

 query = "What should Evelyn BOettcher one page resume should look like for a new Sr. Scientist position"
>>> chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
{'output_text': ' Evelyn Boettcher is a physicist and founder with a Master of Science in Physics from the University of Maryland and a Bachelor of Science in Physics with Highest Honors from the University of Florida. She has a wide range of technical skills, including data science, machine learning, and sensor system development. She has published two journal papers, acquired two patents, and is the head of the Machine Learning Community group. She has experience in designing and executing LabVIEW and Visual Basic programs, integrating systems, designing and testing mechanical tunable filter dispersion compensating devices, and designing magnetic sensor systems. She also enjoys teaching, promoting STEM, and creating things with sewing, 3D printing, and hiking.'}

resume_template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a document and a question, can you create a professional modern resume with bulleted key skills. The resume should be as a markdown document.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""


prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=resume_template
)

chain = load_qa_chain(
    OpenAI(temperature=0.2), chain_type="stuff", memory=memory, prompt=prompt
)

query = "What is a good resume for senior scientist position for Evelyn BOettcher "
chain({"input_documents": resume_data, "human_input": query}, return_only_outputs=True)



resume_template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a document and a question, can you convey their adaptability and quick learning skills on your resume without sounding arrogant or dismissive. 

Here's a suggestion on how to incorporate that sentiment into your resume in a professional manner:
Previous Experience:
Senior [Your Industry] Professional | [Company Name] | [Dates of Employment]
Spearheaded the successful adoption of new technologies, demonstrating the ability to rapidly learn and apply unfamiliar concepts within a tight timeframe.
Demonstrated a passion for continuous learning, consistently adapting to new challenges and expanding skillset to meet evolving industry demands.


{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=resume_template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(
    OpenAI(temperature=0.2), chain_type="stuff", memory=memory, prompt=prompt
)

query = "What is a good resume for senior scientist position for Evelyn BOettcher, please show as a markdown document"
chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
