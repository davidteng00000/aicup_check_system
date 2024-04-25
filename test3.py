import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        delete_btn = gr.Button("Delete")
        confirm_btn = gr.Button("Confirm delete", variant="stop", visible=False)
        cancel_btn = gr.Button("Cancel", visible=False)
        
        delete_btn.click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [delete_btn, confirm_btn, cancel_btn])
        cancel_btn.click(lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [delete_btn, confirm_btn, cancel_btn])
    
demo.launch()