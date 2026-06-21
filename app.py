\"\"\"
Interactive RAG Chatbot UI with Gradio
\"\"\"

import gradio as gr
from src.rag_pipeline import RAGPipeline

# Initialize RAG
print(\"Loading RAG pipeline...\")
rag = RAGPipeline()

def answer_question(question, k=3):
    \"\"\"Process question and return answer with sources\"\"\"
    if not question:
        return \"Please enter a question.\", \"\"
    
    result = rag.query(question, k=k)
    
    answer = f\"**Answer:**\\n{result['answer']}\\n\\n\"
    
    sources = \"\\n**Sources:**\\n\"
    for i, source in enumerate(result['sources'], 1):
        sources += f\"\\n{i}. {source['text'][:200]}...\\n\"
        sources += f\"   Product: {source['metadata']['product_category']}\\n\"
        sources += f\"   Similarity: {source['score']:.3f}\\n\"
    
    return answer, sources

# Create Gradio interface
with gr.Blocks(title=\"Complaint Analysis RAG\", theme=gr.themes.Soft()) as demo:
    gr.Markdown(\"\"\"
    # 🏦 CrediTrust Financial - Complaint Analysis
    
    Ask questions about customer complaints across Credit Cards, Personal Loans, 
    Savings Accounts, and Money Transfers.
    \"\"\")
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label=\"Your Question\",
                placeholder=\"e.g., Why are customers unhappy with credit card fees?\",
                lines=2
            )
            k_slider = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label=\"Number of sources to retrieve (k)\"
            )
            submit_btn = gr.Button(\"🔍 Ask\", variant=\"primary\")
            clear_btn = gr.Button(\"🗑️ Clear\")
        
        with gr.Column(scale=3):
            answer_output = gr.Markdown(label=\"Answer\")
            sources_output = gr.Markdown(label=\"Sources\")
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, k_slider],
        outputs=[answer_output, sources_output]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, k_slider],
        outputs=[answer_output, sources_output]
    )
    
    clear_btn.click(
        fn=lambda: (\"\", \"\", \"\"),
        inputs=[],
        outputs=[question_input, answer_output, sources_output]
    )

if __name__ == \"__main__\":
    demo.launch(share=True)
