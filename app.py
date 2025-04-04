from flask import Flask, render_template, request, jsonify
from Redis_code import RAGChatbot
import json
import ollama
from datetime import datetime

app = Flask(__name__)

# Initialize the chatbot
chatbot = RAGChatbot()

def generate_conversation_summary(history):
    """Generate a summary of the conversation using Ollama."""
    try:
        # Format the conversation history
        formatted_history = "\n\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in history
        ])
        
        # Create the summary prompt
        prompt = f"""Please analyze this onboarding conversation and create a detailed summary. 
        Focus on:
        1. Key topics discussed
        2. Important questions answered
        3. Action items or next steps
        4. Areas that might need follow-up

        Format the response in HTML with appropriate sections and bullet points.

        Conversation:
        {formatted_history}
        """
        
        # Get summary from Ollama using a model good at summarization
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled assistant that creates clear, well-structured summaries of conversations. Format your response in HTML with proper sections and bullet points."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response["message"]["content"]
    except Exception as e:
        return f"<p>Error generating summary: {str(e)}</p>"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        # Get response from the chatbot
        result = chatbot.perform_knn_search(question)
        
        # Generate suggested follow-up questions
        suggested_questions = chatbot.generate_suggested_questions(result['response'])
        
        return jsonify({
            'response': result['response'],
            'matching_chunks': result['matching_chunks'],
            'suggested_questions': suggested_questions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()
        history = data.get('history', [])
        
        if not history:
            return jsonify({'error': 'No conversation history provided'}), 400
        
        # Generate the summary
        summary = generate_conversation_summary(history)
        
        # Store the summary in the session
        app.config['CURRENT_SUMMARY'] = {
            'content': summary,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summary')
def show_summary():
    summary_data = app.config.get('CURRENT_SUMMARY', {
        'content': '<p>No summary available.</p>',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    return render_template(
        'summary.html',
        summary=summary_data['content'],
        timestamp=summary_data['timestamp']
    )

if __name__ == '__main__':
    app.run(debug=True, port=5002) 