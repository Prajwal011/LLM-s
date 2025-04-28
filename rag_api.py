from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from rag_pipeline import RAGPipeline

app = Flask(__name__)
CORS(app)

# Groq API configuration
GROQ_API_KEY = "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"

# Initialize ChatGroq
chat_model = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Store chat histories for different sessions
chat_histories = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')

        # Detect intent
        intent = detect_intent(user_message)

        if intent == "ask_user_info":
            assistant_message = (
                "I'm glad to hear you're interested! Could you please share your name and email so we can assist you further?"
            )
        elif intent == "career_guidance":
            assistant_message = (
                "For personalized career guidance, please reach out to our office. Our advisors will be happy to help you!"
            )
        else:
            # Get relevant context from RAG pipeline
            context = rag_pipeline.get_relevant_context(user_message)
            
            # Create system message with context
            system_message = f"""You are a helpful AI assistant for AI Adventures Pune. 
            Use the following context to answer questions about AI Adventures Pune:
            
            {context}
            
            Also highlight the important information in the response.
            If the question is not related to AI Adventures Pune, you can answer generally.
            Always be helpful and informative."""

            # Prepare messages for ChatGroq
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ]

            # Add chat history if available
            if session_id in chat_histories:
                for msg in chat_histories[session_id]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(SystemMessage(content=msg["content"]))

            # Get response from ChatGroq
            response = chat_model.invoke(messages)
            assistant_message = response.content

        # Save to chat history
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ])
        print("user query--------------------------------",user_message)
        print('assistant_message------------:',assistant_message)
        return jsonify({
            'message': assistant_message,
            'status': 'success'
        })
        
    except Exception as e:
        print('Error:', e)
        return jsonify({
            'message': str(e),
            'status': 'error'
        }), 500

def detect_intent(user_message):
    message = user_message.lower()
    # Intent 1: Interested in course/info
    interested_keywords = [
        "interested", "enroll", "join", "sign up", "register", "admission", "apply"
    ]
    for word in interested_keywords:
        if word in message:
            return "ask_user_info"
    # Intent 2: Career guidance
    career_keywords = [
        "career guidance", "career advice", "career help", "career support", "career counseling"
    ]
    for word in career_keywords:
        if word in message:
            return "career_guidance"
    return None

if __name__ == '__main__':
    app.run(debug=True, load_dotenv=False) 
