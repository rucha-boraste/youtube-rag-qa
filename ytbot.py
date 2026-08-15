import gradio as gr
import re 
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_video_id(url):
    pattern = r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url):
    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        if t.language_code == 'en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break
    
    return transcript if transcript else None


def process(transcript):
    txt=""
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass
    
    return txt

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


def initialize_groq_llm(model_id, temperature, max_tokens):
    llm = ChatGroq(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm

def setup_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    
def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index , query, k=3):
    results = faiss_index.similarity_search(query, k=k)
    return results


def create_summary_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant tasked with summarizing YouTube video transcripts.

                Provide a concise and informative summary that captures the main points of the video.

                Instructions:
                1. Summarize the transcript in a single concise paragraph.
                2. Ignore timestamps.
                3. Focus only on the spoken content of the video."""
            ),
            (
                "human",
                """Please summarize the following YouTube video transcript:
                {transcript}"""
            )
        ]
    )
    
    return prompt

def create_summary_chain(llm, prompt):
    return prompt | llm


def retrieve(query, faiss_index, k=7):
    revelant_context = faiss_index.similarity_search(query, k=k)
    return revelant_context #Returns a list of k most relevant documents

def create_qa_prompt_template():
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert assistant Providing detailed answers based on the following video content. """
            ),
            (
                "human",
                """Relevant video context: {context}
                Based on above context, please answer the following question:
                Question: {question}"""
            )
        ]
    )
    return template


def create_qa_chain(llm, prompt_template):
    return prompt_template | llm


def generate_answer(question, faiss_index, qa_chain, k=7):
    relevant_context = retrieve(question, faiss_index, k=k)
    response = qa_chain.invoke(
        {
            "context": relevant_context,
            "question": question
        }
    )
    
    return response.content



processed_transcript=""

def summarize_video(video_url):
    global fetched_transcript, processed_transcript
    
    if not video_url:
        return "Please provide a valid video URL."
    
    fetched_transcript = get_transcript(video_url)
    
    if not fetched_transcript:
        return "No transcript available for this video."
    
    processed_transcript = process(fetched_transcript)
    
    if not processed_transcript:
        return "No transcript available after processing."
    
    summary_prompt = create_summary_prompt()
    llm = initialize_groq_llm("llama-3.3-70b-versatile", 0, 400)
    summary_chain = create_summary_chain(llm, summary_prompt)
    
    response = summary_chain.invoke(
        {
            "transcript": processed_transcript
        }
    )
    
    return response.content


def answer_question(video_url, user_question):
    global fetched_transcript, processed_transcript
    
    if not video_url:
        return "Please provide valid youtube URL."
    
    if not user_question:
        return "Please provide a valid question."
    
    if not processed_transcript:
        fetched_transcript = get_transcript(video_url)
        
        if not fetched_transcript:
            return "No transcript available for this video."
        
        processed_transcript = process(fetched_transcript)
        
    if not processed_transcript:
        return "No transcript available for this video."
    
    chunks = chunk_transcript(processed_transcript)
    llm = initialize_groq_llm("llama-3.3-70b-versatile", 0, 400)
    embedding_model = setup_embedding_model()
    faiss_index = create_faiss_index(
        chunks,
        embedding_model
    )
    
    qa_prompt = create_qa_prompt_template()
    qa_chain = create_qa_chain(llm, qa_prompt)
    
    answer = generate_answer(
        user_question,
        faiss_index,
        qa_chain
    )
    
    return answer

# url = "https://youtu.be/gset79KMmt0?si=poPiUU98FMSNzaSk"
# transcript = get_transcript(url)
# formatted_transcript = process(transcript)
# chunks = chunk_transcript(formatted_transcript)
# for i in chunks:
#     print("---------------")
#     print(i)


with gr.Blocks(title="YouTube RAG Assistant") as interface:
    video_url = gr.Textbox(
        label="YouTube Video URL",
        placeholder="Enter the YouTube Video URL"
    )
    
    summary_output = gr.Textbox(
        label="Video Summary",
        lines=5
    )
    
    question_input = gr.Textbox(
        label="Ask question about the video",
        placeholder="Ask your question"
    )
    
    answer_output = gr.Textbox(
        label="Answer to your question",
        lines=5
    )
    
    summarize_btn = gr.Button(
        "Summarize Video"
    )
    
    question_btn = gr.Button(
        "Ask a question"
    )
    
    transcript_status = gr.Textbox(
        label="Transcript Status",
        interactive=False
    )
    
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)
    
interface.launch(server_name="0.0.0.0", server_port=7860)