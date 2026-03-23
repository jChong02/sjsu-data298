"""
TokenSHAP QA Streamlit Interface

A web-based interface for analyzing medical question-answering models
using TokenSHAP explainability methods.

Usage:
    streamlit run tokenshap_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
from pathlib import Path
import sys
import streamlit.components.v1 as components

# Import your custom modules
from medical_llm_wrapper import MedicalLLMWrapper, load_medical_llm
from TokenSHAP_QA.tokenshap_extensions import QATokenSHAP
from TokenSHAP_QA.tokenshap_extensions.value_functions import CorrectnessValueFunction
from TokenSHAP_QA.token_shap.token_shap import StringSplitter
from TokenSHAP_QA.token_shap.base import TfidfTextVectorizer, HuggingFaceEmbeddings, OpenAIEmbeddings

# Page configuration
st.set_page_config(
    page_title="TokenSHAP QA Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .token-box {
        display: inline-block;
        padding: 5px 10px;
        margin: 2px;
        border-radius: 5px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
    }
    .token-box:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        z-index: 100;
    }
    .tooltip {
        pointer-events: none;
        font-family: monospace;
        line-height: 1.6;
    }
    .tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: rgba(0,0,0,0.9);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'shapley_values' not in st.session_state:
    st.session_state.shapley_values = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'baseline_text' not in st.session_state:
    st.session_state.baseline_text = None

def get_color_for_value(value, min_val, max_val):
    """Generate color based on normalized Shapley value"""
    if max_val == min_val:
        norm_value = 0.5
    else:
        norm_value = (value - min_val) / (max_val - min_val)
    
    # Red (low) to Yellow (mid) to Green (high)
    if norm_value < 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (norm_value * 2))
        b = 0
    else:
        # Yellow to Green
        r = int(255 * (2 - norm_value * 2))
        g = 255
        b = 0
    
    return f"rgb({r},{g},{b})"

def create_interactive_visualization(shapley_values):
    """Create interactive Plotly visualization of token importance"""
    if not shapley_values:
        return None
    
    # Extract tokens and values
    tokens = [k.rsplit('_', 1)[0] for k in shapley_values.keys()]
    values = list(shapley_values.values())
    indices = list(range(len(tokens)))
    
    min_val = min(values)
    max_val = max(values)
    
    # Create colors
    colors = [get_color_for_value(v, min_val, max_val) for v in values]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=indices,
            y=values,
            text=tokens,
            textposition='outside',
            marker=dict(
                color=values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Shapley Value")
            ),
            hovertemplate='<b>%{text}</b><br>Shapley Value: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Token Importance (Shapley Values)",
        xaxis_title="Token Position",
        yaxis_title="Shapley Value",
        height=400,
        hovermode='closest',
        showlegend=False
    )
    
    return fig

def create_text_visualization(shapley_values):
    """Create colored text visualization with interactive hover"""
    if not shapley_values:
        return ""
    
    min_val = min(shapley_values.values())
    max_val = max(shapley_values.values())
    
    # Build complete HTML with inline styles and JavaScript
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                margin: 0;
                padding: 20px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #0e1117;
                color: #fafafa;
            }
            .token-container {
                line-height: 3;
                padding-bottom: 80px;
            }
            .token-box {
                display: inline-block;
                padding: 8px 14px;
                margin: 4px;
                border-radius: 5px;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                position: relative;
                color: white;
                font-weight: 600;
                font-size: 15px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }
            .token-box:hover {
                transform: scale(1.15);
                box-shadow: 0 6px 16px rgba(0,0,0,0.5);
                z-index: 100;
            }
            .tooltip {
                display: none;
                position: absolute;
                top: calc(100% + 12px);
                left: 50%;
                transform: translateX(-50%);
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                white-space: nowrap;
                font-size: 14px;
                z-index: 1000;
                box-shadow: 0 8px 24px rgba(0,0,0,0.6);
                font-family: 'Courier New', monospace;
                line-height: 1.8;
                pointer-events: none;
                border: 2px solid rgba(255,255,255,0.3);
                min-width: 180px;
            }
            .tooltip::before {
                content: '';
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                border: 8px solid transparent;
                border-bottom-color: #667eea;
            }
            .tooltip-visible {
                display: block;
                animation: fadeIn 0.2s ease-in;
            }
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateX(-50%) translateY(-5px);
                }
                to {
                    opacity: 1;
                    transform: translateX(-50%) translateY(0);
                }
            }
            .tooltip strong {
                color: #ffd700;
                font-weight: 700;
            }
        </style>
    </head>
    <body>
        <div class="token-container">
    """
    
    # Generate tokens
    for idx, (token_key, value) in enumerate(shapley_values.items()):
        token = token_key.rsplit('_', 1)[0]
        position = token_key.rsplit('_', 1)[1]
        color = get_color_for_value(value, min_val, max_val)
        
        html_content += f'''
            <span class="token-box" id="token-{idx}" style="background-color: {color};">
                {token}
                <span class="tooltip" id="tooltip-{idx}">
                    <strong>Token:</strong> {token}<br>
                    <strong>Position:</strong> {position}<br>
                    <strong>Shapley:</strong> {value:.4f}
                </span>
            </span>
        '''
    
    html_content += """
        </div>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const tokens = document.querySelectorAll('.token-box');
                
                tokens.forEach(token => {
                    const tooltip = token.querySelector('.tooltip');
                    
                    token.addEventListener('mouseenter', function() {
                        if (tooltip) {
                            tooltip.classList.add('tooltip-visible');
                        }
                    });
                    
                    token.addEventListener('mouseleave', function() {
                        if (tooltip) {
                            tooltip.classList.remove('tooltip-visible');
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    
    return html_content

# Sidebar - Model Selection
st.sidebar.header("🤖 Model Configuration")

# Hugging Face token input (for gated models like MedGemma)
with st.sidebar.expander("🔑 Hugging Face Token (for gated models)", expanded=False):
    hf_token = st.text_input(
        "HF Token",
        type="password",
        help="Required for gated models like MedGemma. Get yours at https://huggingface.co/settings/tokens",
        key="hf_token_input"
    )
    st.caption("⚠️ MedGemma requires authentication. Create a token and accept the model terms at [MedGemma model page](https://huggingface.co/google/medgemma-2b-it)")

model_options = {
    "MedGemma 4B": "google/medgemma-4b-it",
    "Apollo 2B": "FreedomIntelligence/Apollo-2B",
    "BioMistral 7B": "BioMistral/BioMistral-7B"
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    help="Choose the medical language model to analyze"
)

# Show warning for MedGemma if no token provided
if "MedGemma" in selected_model_name and not hf_token:
    st.sidebar.warning("⚠️ MedGemma requires a Hugging Face token. Please provide one above.")

task_type = st.sidebar.selectbox(
    "Task Type",
    options=["yn", "mcq", "free"],
    index=0,
    help="Select the type of question-answering task"
)

generation_mode = st.sidebar.selectbox(
    "Generation Mode",
    options=["answer_rationale", "answer_only"],
    index=0,
    help="Choose whether to generate full rationale or answer only"
)

# Apply task/mode changes to existing model if loaded
if st.session_state.model_loaded and st.session_state.model is not None:
    # Check if task type or mode changed
    current_task = getattr(st.session_state.model, 'task_type', None)
    current_mode = getattr(st.session_state.model, 'mode', None)
    
    if current_task != task_type or current_mode != generation_mode:
        st.session_state.model.set_task(task_type)
        st.session_state.model.set_mode(generation_mode)
        st.sidebar.info(f"✅ Updated to {task_type.upper()} task, {generation_mode} mode")

# Load Model Button
if st.sidebar.button("🔄 Load Model", type="primary"):
    # Check for token requirement
    if "MedGemma" in selected_model_name and not hf_token:
        st.sidebar.error("❌ MedGemma requires a Hugging Face token. Please provide one in the 🔑 section above.")
        st.sidebar.info("📋 Steps:\n1. Go to https://huggingface.co/settings/tokens\n2. Create a token\n3. Accept model terms at https://huggingface.co/google/medgemma-4b-it")
    else:
        with st.spinner(f"Loading {selected_model_name}..."):
            try:
                # Clean up old model from memory before loading new one
                if st.session_state.model_loaded and st.session_state.model is not None:
                    st.sidebar.info("🗑️ Cleaning up previous model from memory...")
                    
                    # Delete model and clear GPU cache
                    del st.session_state.model
                    st.session_state.model = None
                    
                    # Clear CUDA cache if using GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Python garbage collection
                    import gc
                    gc.collect()
                    
                    st.sidebar.success("✅ Previous model cleared from memory")
                
                # Load new model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_path = model_options[selected_model_name]
                
                # Pass token if provided
                token_to_use = hf_token if hf_token else None
                
                st.session_state.model = load_medical_llm(
                    model_path, 
                    device=device,
                    token=token_to_use
                )
                st.session_state.model.set_task(task_type)
                st.session_state.model.set_mode(generation_mode)
                st.session_state.model_loaded = True
                st.session_state.current_model_name = selected_model_name
                
                # Clear previous analysis results since model changed
                st.session_state.analysis_complete = False
                st.session_state.shapley_values = None
                st.session_state.results_df = None
                st.session_state.baseline_text = None
                
                st.sidebar.success(f"✅ {selected_model_name} loaded successfully!")
                st.sidebar.info(f"Device: {device.upper()}")
                
                # Show memory info if CUDA available
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    st.sidebar.info(f"GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error loading model: {str(e)}")
                
                # Provide helpful error messages
                error_msg = str(e)
                if "authentication" in error_msg.lower() or "401" in error_msg:
                    st.sidebar.error("🔒 Authentication failed. Please check your HF token.")
                elif "403" in error_msg or "gated" in error_msg.lower():
                    st.sidebar.error("🚫 Access denied. Make sure you've accepted the model terms at the HuggingFace model page.")
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    st.sidebar.error("🌐 Network error. Check your internet connection.")
                
                st.session_state.model_loaded = False

# Display model status
if st.session_state.model_loaded:
    st.sidebar.success(f"✅ Model Ready: {st.session_state.current_model_name}")
    
    # Add a clear model button
    if st.sidebar.button("🗑️ Clear Model from Memory"):
        if st.session_state.model is not None:
            del st.session_state.model
            st.session_state.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            import gc
            gc.collect()
            
            st.session_state.model_loaded = False
            st.session_state.current_model_name = None
            st.session_state.analysis_complete = False
            st.sidebar.success("✅ Model cleared from memory")
            st.rerun()
else:
    st.sidebar.warning("⚠️ No model loaded")

# Main content
st.title("🔬 TokenSHAP QA Analyzer")
st.markdown("Explainable AI for Medical Question-Answering Models")

# Question Input Section
st.header("📝 Question Input")

# Provide example questions
example_questions = {
    "Septoplasty (Y/N)": """Question:
Does septoplasty change the dimensions of compensatory hypertrophy of the middle turbinate?

Answer Choices:
A. Yes
B. No""",
    
    "Laparoscopic Cholecystectomy (Y/N)": """Question:
Is laparoscopic cholecystectomy not safe as a day case procedure?

Answer Choices:
A. Yes
B. No""",
    
    "Potassium Treatment (MCQ)": """Question:
A 9 year old girl was admitted for dialysis. On laboratory examination her potassium levels were 7.8 mEq/L. Which of the following would quickly lower her increased potassium levels?

Answer Choices:
A. IV calcium gluconate
B. IV Glucose and insulin
C. Oral kayexalate in sorbitol
D. IV NaHCO3"""
}

col1, col2 = st.columns([3, 1])

with col1:
    selected_example = st.selectbox(
        "Load Example Question",
        options=["Custom"] + list(example_questions.keys()),
        help="Select a pre-defined example or enter your own"
    )

with col2:
    if selected_example != "Custom":
        if st.button("📋 Load Example"):
            st.session_state.question_text = example_questions[selected_example]

question_prompt = st.text_area(
    "Enter Question Prompt",
    value=st.session_state.get('question_text', ''),
    height=200,
    placeholder="Enter your medical question here...",
    help="Format: Question + Answer Choices (for Y/N or MCQ tasks)"
)

# Optional: Ground truth answer
ground_truth = st.text_input(
    "Ground Truth Answer (Optional)",
    placeholder="e.g., 'A' or 'B' for validation",
    help="Provide the correct answer if known"
)

# TokenSHAP Configuration Section
st.header("⚙️ TokenSHAP Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    vectorizer_type = st.selectbox(
        "Vectorizer",
        options=["TF-IDF", "HuggingFace", "OpenAI"],
        index=0,
        help="Method for text vectorization"
    )
    
    if vectorizer_type == "HuggingFace":
        hf_model = st.text_input(
            "HF Model",
            value="sentence-transformers/all-MiniLM-L6-v2",
            help="HuggingFace model for embeddings"
        )
    elif vectorizer_type == "OpenAI":
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key"
        )

with col2:
    sampling_ratio = st.slider(
        "Sampling Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Ratio of token combinations to sample (0=essential only, 1=all)"
    )

with col3:
    max_combinations = st.number_input(
        "Max Combinations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Maximum number of token combinations to analyze"
    )

debug_mode = st.checkbox(
    "Debug Mode",
    value=False,
    help="Show detailed debug information during analysis"
)

# Analysis Button
if st.button("🔍 Analyze with TokenSHAP", type="primary", disabled=not st.session_state.model_loaded):
    if not question_prompt.strip():
        st.error("❌ Please enter a question prompt!")
    else:
        try:
            # Initialize vectorizer
            if vectorizer_type == "TF-IDF":
                vectorizer = TfidfTextVectorizer()
            elif vectorizer_type == "HuggingFace":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                vectorizer = HuggingFaceEmbeddings(model_name=hf_model, device=device)
            elif vectorizer_type == "OpenAI":
                if not openai_key:
                    st.error("❌ Please provide OpenAI API key!")
                    st.stop()
                vectorizer = OpenAIEmbeddings(api_key=openai_key)
            
            # Initialize TokenSHAP
            splitter = StringSplitter()
            token_shap = QATokenSHAP(
                model=st.session_state.model,
                splitter=splitter,
                vectorizer=vectorizer,
                debug=debug_mode
            )
            
            # Run analysis with progress bar
            st.info("🔄 Running TokenSHAP analysis...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Analyzing token importance..."):
                status_text.text("Extracting baseline response...")
                progress_bar.progress(10)
                
                status_text.text("Generating token combinations...")
                progress_bar.progress(30)
                
                # Run analysis
                results_df = token_shap.analyze(
                    question_prompt,
                    sampling_ratio=sampling_ratio,
                    max_combinations=max_combinations,
                    print_highlight_text=False
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
            
            # Store results in session state
            st.session_state.analysis_complete = True
            st.session_state.shapley_values = token_shap.shapley_values
            st.session_state.results_df = results_df
            st.session_state.baseline_text = token_shap.baseline_text
            
            st.success("✅ TokenSHAP analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            if debug_mode:
                st.exception(e)

# Results Section
if st.session_state.analysis_complete:
    st.header("📊 Analysis Results")
    
    # Baseline Response
    with st.expander("🎯 Baseline Model Response", expanded=True):
        st.markdown("**Model Output:**")
        # Use markdown with proper formatting to respect newlines
        formatted_output = st.session_state.baseline_text.replace('\n', '  \n')
        st.markdown(f"```\n{st.session_state.baseline_text}\n```")
        
        # Also show it in a nicer box
        st.markdown("---")
        st.markdown(formatted_output)
    
    # Interactive Visualizations
    st.subheader("📈 Token Importance Visualization")
    
    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🎨 Colored Text", "📋 Data Table"])
    
    with tab1:
        fig = create_interactive_visualization(st.session_state.shapley_values)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 Hover over bars to see token details")
    
    with tab2:
        st.markdown("### Token Importance (Hover for detailed values)")
        
        # Use Streamlit's components.html to render interactive HTML
        colored_html = create_text_visualization(st.session_state.shapley_values)
        components.html(colored_html, height=500, scrolling=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Legend with better formatting
        st.markdown("#### Color Scale Legend")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🔴 <span style='background-color: rgb(255,0,0); color: white; padding: 3px 8px; border-radius: 3px;'>Low Importance</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("🟡 <span style='background-color: rgb(255,255,0); color: black; padding: 3px 8px; border-radius: 3px;'>Medium Importance</span>", unsafe_allow_html=True)
        with col3:
            st.markdown("🟢 <span style='background-color: rgb(0,255,0); color: white; padding: 3px 8px; border-radius: 3px;'>High Importance</span>", unsafe_allow_html=True)
        
        st.info("💡 Hover over any token to see its exact Shapley value and position")
    
    with tab3:
        # Create DataFrame for display
        shapley_df = pd.DataFrame([
            {
                'Token': k.rsplit('_', 1)[0],
                'Position': k.rsplit('_', 1)[1],
                'Shapley Value': v
            }
            for k, v in st.session_state.shapley_values.items()
        ]).sort_values('Shapley Value', ascending=False)
        
        st.dataframe(
            shapley_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = shapley_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="tokenshap_results.csv",
            mime="text/csv"
        )
    
    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    values = list(st.session_state.shapley_values.values())
    
    with col1:
        st.metric("Total Tokens", len(values))
    with col2:
        st.metric("Max Importance", f"{max(values):.4f}")
    with col3:
        st.metric("Min Importance", f"{min(values):.4f}")
    with col4:
        st.metric("Mean Importance", f"{np.mean(values):.4f}")
    
    # Detailed Results DataFrame
    with st.expander("🔍 Detailed Combination Results"):
        st.dataframe(
            st.session_state.results_df,
            use_container_width=True,
            hide_index=True
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | TokenSHAP QA Analyzer v1.0</p>
        <p><small>For medical AI explainability research</small></p>
    </div>
    """,
    unsafe_allow_html=True
)