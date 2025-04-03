from flask import Flask, render_template, request, jsonify
from Redis_code import RAGChatbot
import json

app = Flask(__name__)

# Initialize the chatbot
chatbot = RAGChatbot()

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

if __name__ == '__main__':
    app.run(debug=True, port=5002) 