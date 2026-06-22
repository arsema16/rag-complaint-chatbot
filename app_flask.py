from flask import Flask, request, jsonify, render_template_string
import json

app = Flask(__name__)

# HTML template (simple version)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Complaint Analysis RAG</title></head>
<body>
    <h1>🏦 CrediTrust Complaint Analysis</h1>
    <form id="queryForm">
        <textarea id="question" rows="4" cols="50">Why are customers unhappy with credit card fees?</textarea><br>
        <button type="submit">Ask</button>
    </form>
    <div id="result"></div>
    <script>
        document.getElementById('queryForm').onsubmit = async function(e) {
            e.preventDefault();
            const question = document.getElementById('question').value;
            const response = await fetch('/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question, k: 3})
            });
            const data = await response.json();
            document.getElementById('result').innerHTML = '<h3>Answer:</h3><p>' + data.answer + '</p>';
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    from src.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    result = rag.query(data.get('question', ''), k=data.get('k', 3))
    return jsonify({'answer': result['answer']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
