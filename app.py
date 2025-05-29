"""
Web interface for the nanozyme multi-agent framework.
"""
from flask import Flask, render_template, request, jsonify, session
from src.framework import SimpleReActFramework
from src.utils.logger import setup_logger
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)
logger = setup_logger("nanozyme_framework.web")

conversation_history = {}

def format_result(result_text):
    if not result_text:
        return ""
        
    lines = result_text.split('\n')
    formatted_lines = [line.strip() for line in lines]
    
    formatted_result = ""
    prev_empty = False
    for line in formatted_lines:
        if not line:
            if not prev_empty:
                formatted_result += "\n"
                prev_empty = True
        else:
            formatted_result += line + "\n"
            prev_empty = False
            
    return formatted_result

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        conversation_history[session['session_id']] = []
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form.get('query', '')
    api_key = request.form.get('api_key', '').strip()
    session_id = session.get('session_id')
    
    if not api_key:
        return jsonify({'error': "API key is required to use this system."})
    
    if not query:
        return jsonify({'error': "Please enter a query."})
    
    try:
        history = conversation_history.get(session_id, [])
        
        react_framework = SimpleReActFramework(openai_api_key=api_key)
        
        if history:
            context = "\n".join([f"Previous conversation:\nQ: {h['query']}\nA: {h['result']}" for h in history[-10:]])
            query = f"{context}\n\nCurrent query: {query}"
        
        result = react_framework.run(query)
        
        formatted_result = format_result(result)
        
        history.append({
            'query': query,
            'result': formatted_result
        })
        conversation_history[session_id] = history
        
        logger.info(f"Query processed: {query[:50]}...")
        
        return jsonify({'result': formatted_result})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f"Error during processing: {str(e)}"})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history for current session."""
    session_id = session.get('session_id')
    if session_id in conversation_history:
        conversation_history[session_id] = []
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    app.run(debug=True)