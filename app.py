import streamlit as st
import requests
import io
import json
import plotly.graph_objects as go
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Opoku ML | GeoSight",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: INFO & LINKS ---
with st.sidebar:
    # UPDATED: Using your specific logo file
    try:
        st.image("opoku_logo.jpeg", width=250)
    except:
        st.warning("Logo not found. Make sure 'opoku_logo.jpeg' is in the folder.")
        st.title("Opoku ML")

    st.caption("v1.0.0 | By Kofi Opoku")
    
    st.markdown("---")
    
    st.subheader("📖 Project Description")
    st.markdown("""
    **GeoSight** is a satellite terrain intelligence system built for the **Opoku ML** portfolio.
    
    It uses a **ConvNeXt Tiny** Deep Learning model (ONNX) running on **AWS Lambda (Docker)** to classify satellite imagery into 10 distinct terrain types for urban planning and environmental monitoring.
    """)
    
    st.subheader("📝 Instructions")
    st.markdown("""
    1. Upload a satellite image (EuroSAT format).
    2. Click **Analyze Terrain**.
    3. View real-time inference results.
    """)
    
    st.markdown("---")
    
    st.subheader("🔗 Connect")
    st.markdown(
        """
        <div style='display: flex; gap: 10px;'>
            <a href='https://github.com/K-Opoku' target='_blank'>
                <img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' width='100' />
            </a>
            <a href='https://www.linkedin.com/in/kofi-opoku-98b903301/' target='_blank'>
                <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' width='100' />
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- MAIN PAGE: BRANDING ---
st.title("Opoku ML | GeoSight Intelligence")
st.markdown("##### 🌍 Autonomous Satellite Terrain Classification System")
st.divider()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 1. Upload Imagery")
    uploaded_file = st.file_uploader("Drop satellite feed here...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Source Image", use_container_width=True)
        
        # Centered Button
        if st.button(" Analyze Terrain", type="primary", use_container_width=True):
            
            with st.spinner("Processing via AWS Lambda Container..."):
                try:
                    # Prepare image
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    img_bytes = list(img_byte_arr.getvalue())

                    # Send to Docker
                    payload = {"image_bytes": img_bytes}
                    
                    # NOTE: Connects to local Docker container
                    response = requests.post(
                        "http://localhost:9000/2015-03-31/functions/function/invocations",
                        json=payload
                    )
                    
                    result = response.json()
                    if 'body' in result:
                        data = json.loads(result['body'])
                    else:
                        data = result

                    # --- SUCCESS STATE ---
                    if response.status_code == 200 and 'error' not in data:
                        st.session_state['last_result'] = data
                    else:
                        st.error(f"Analysis Failed: {data.get('error', 'Unknown Error')}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- RESULTS COLUMN (RIGHT) ---
with col2:
    if 'last_result' in st.session_state:
        data = st.session_state['last_result']
        
        st.markdown("### 2. Analysis Results")
        
        # Top Metrics Row
        m1, m2 = st.columns(2)
        with m1:
            st.info(f"**Detected Class:** {data['class']}")
        with m2:
            st.success(f"**Confidence:** {data['confidence']:.1%}")

        # Natural Language Context
        st.markdown(f"""
        > *AI Analysis Report:*
        > The system has identified this terrain as **{data['class']}** with a confidence score of **{data['confidence']:.2%}**. 
        > {data['description']}
        """)

        # Gauge Chart (Plotly)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = data['confidence'] * 100,
            title = {'text': "Confidence Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00CC96"}, # Opoku ML Green
                'steps': [
                    {'range': [0, 50], 'color': "#2b2b2b"},
                    {'range': [50, 80], 'color': "#404040"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Detailed Probabilities (Bar Chart)
        with st.expander("📊 View Full Class Probabilities"):
            st.bar_chart(data['chart_data'])
            
        # Action Item
        st.warning(f"💡 **Recommended Action:** {data['recommendation']}")

    else:
        # Placeholder when no data is loaded yet
        st.info("👈 Upload an image and click 'Analyze' to see intelligence data here.")