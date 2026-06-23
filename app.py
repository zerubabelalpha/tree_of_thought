import os
import gradio as gr
from dotenv import load_dotenv
from tot.llm_client import LLMClient
from tot.arxiv_client import ArxivClient
from tot.tot_search import TreeOfThoughts

# Load environmental variables
load_dotenv()

def run_deep_research(query, max_depth, branching_factor, score_threshold):
    """
    Triggers the Tree of Thoughts research process, reloading environment configs
    dynamically on each execution run.
    """
    # Force reload of .env file in case the user edited it while the server is active
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    
    if not api_key:
        yield (
            "### Error\n`OPENROUTER_API_KEY` is not set.\n\nPlease define it in a `.env` file in the project folder or in your environment variables, then click 'Start Deep Research' again.",
            ""
        )
        return
        
    if not model:
        # Default fallback model
        model = "google/gemini-2.5-flash"
        
    if not query.strip():
        yield "### Error\nPlease enter a valid query to begin research.", ""
        return
        
    # Initialize clients with current config
    llm_client = LLMClient(api_key=api_key, model=model)
    arxiv_client = ArxivClient()
    tot_solver = TreeOfThoughts(llm_client, arxiv_client)
    
    # Run solver and stream output
    try:
        for status, report in tot_solver.solve(
            query=query,
            max_depth=int(max_depth),
            branching_factor=int(branching_factor),
            score_threshold=float(score_threshold)
        ):
            yield status, report
    except Exception as e:
        yield f"### Execution Exception\nAn unexpected error occurred:\n`{str(e)}`", ""


# Custom CSS styling for premium minimalistic aesthetic
css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

.gradio-container {
    font-family: 'Outfit', sans-serif !important;
    max-width: 1100px !important;
    margin: 30px auto !important;
    padding: 20px !important;
}

/* Header styling */
.header-container {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-color-primary, var(--block-border-color, #e2e8f0));
}

.header-container h1 {
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.5rem !important;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.header-container p {
    font-size: 1.05rem !important;
    color: var(--body-text-color-subdued, #64748b) !important;
    max-width: 600px;
    margin: 0 auto !important;
}

/* Status log styling */
.status-log {
    background-color: var(--block-background-fill, #f8fafc) !important;
    border: 1px solid var(--block-border-color, #e2e8f0) !important;
    border-radius: var(--block-radius, 12px) !important;
    padding: 20px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92em !important;
    max-height: 420px;
    overflow-y: auto;
}

/* Final report styling */
.final-report {
    background-color: var(--block-background-fill, #f8fafc) !important;
    border: 1px solid var(--block-border-color, #e2e8f0) !important;
    border-radius: var(--block-radius, 12px) !important;
    padding: 28px !important;
    margin-top: 15px !important;
}

.final-report h1, .final-report h2, .final-report h3 {
    color: var(--primary-color, #6366f1) !important;
    margin-top: 1.6em !important;
    margin-bottom: 0.6em !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
}

.final-report p, .final-report li {
    line-height: 1.7 !important;
    font-size: 1.02rem !important;
}

.final-report hr {
    border: 0;
    height: 1px;
    background: var(--block-border-color, #e2e8f0);
    margin: 24px 0;
}

/* Parameters block */
.parameter-card {
    background-color: var(--block-background-fill, #f8fafc) !important;
    border: 1px solid var(--block-border-color, #e2e8f0) !important;
    border-radius: var(--block-radius, 12px) !important;
    padding: 16px !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=theme, title="Tree of Thoughts Deep Researcher") as demo:
    gr.HTML("""
        <div class="header-container">
            <h1>Tree of Thoughts Deep Researcher</h1>
        </div>
    """)
        
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Research Topic / Query",
                lines=3
            )
            
            with gr.Accordion("Advanced Reasoning Configuration", open=False):
                with gr.Group(elem_classes=["parameter-card"]):
                    max_depth_slider = gr.Slider(
                        minimum=1,
                        maximum=3,
                        value=2,
                        step=1,
                        label="Search Depth "# Layers of thought exploration
                    )
                    branching_slider = gr.Slider(
                        minimum=2,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Branching Factor" # Number of directions to explore
                    )
                    threshold_slider = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=5.0,
                        step=0.5,
                        label="Score Threshold" # Minimum node evaluation score
                    )
            
            run_btn = gr.Button("Start Deep Research", variant="primary")

        with gr.Column(scale=3):
            # Agent activity log
            status_box = gr.Markdown(
                value="*Agent is idle. Enter a query and click 'Start Deep Research' above to begin.*",
                label="Agent Activity Log",
                elem_classes=["status-log"]
            )

    with gr.Row(variant="panel"):
        with gr.Column():
            # Final synthesized report output
            report_box = gr.Markdown(
                value="*The final compiled literature report will be rendered here once research completes.*",
                label="Synthesized Research Paper",
                elem_classes=["final-report"]
            )

    # Event handlers
    run_btn.click(
        fn=run_deep_research,
        inputs=[query_input, max_depth_slider, branching_slider, threshold_slider],
        outputs=[status_box, report_box]
    )

if __name__ == "__main__":
    # Launch locally on default port
    demo.launch(server_name="127.0.0.1", share=False, css=css)
