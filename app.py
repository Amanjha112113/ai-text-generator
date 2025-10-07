import streamlit as st
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv
import os
import time  # For retries
import json  # Fixes UnboundLocalError
from tenacity import retry, stop_after_attempt, wait_exponential  # tenacity>=8.0.0

# Load .env (local fallback)
load_dotenv()

# Best practice: Prioritize st.secrets for deployment, fallback to .env
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing GEMINI_API_KEY. Set it in .env (local) or Streamlit Secrets (deploy).")
    st.stop()  # Halt app on missing key

genai.configure(api_key=api_key)

# Dynamic model selection (defaults to 2.5 Flash for speed/stability)
model_name = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
model = genai.GenerativeModel(model_name)

@st.cache_data(ttl=300)  # 5-min TTL for freshness
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_sentiment(text: str) -> str:
    """
    Analyze sentiment using enforced JSON mode for 99% reliability.
    """
    if len(text.strip()) < 3:
        return 'neutral'
    
    sentiment_prompt = f"""
    Classify the sentiment of the following text as exactly one of: positive, negative, neutral.
    
    Text: {text}
    """
    
    # JSON Schema for enforced structure
    json_schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            }
        },
        "required": ["sentiment"]
    }
    
    try:
        response = model.generate_content(
            sentiment_prompt,
            generation_config={
                "response_mime_type": "application/json",  # Forces JSON
                "response_schema": json_schema  # Enforces exact format
            }
        )
        if not response.text.strip():
            raise ValueError("Empty response from Gemini")
        result = json.loads(response.text.strip())
        return result.get("sentiment", "neutral").lower()
    except (json.JSONDecodeError, ValueError, Exception) as e:
        st.error(f"Error in sentiment analysis: {e}")
        return 'neutral'

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_sentiment_aligned_text(sentiment: str, original_prompt: str, word_count: int = 200, style_mode: str = "ELI10") -> str:
    """
    Enhanced with structured constraints in prompt, including ELI10 style for simple, fun explanations.
    Adds sentiment-specific flair: jokes & emojis for positive; appropriate emojis for negative/neutral.
    Supports 'ELI10' (fun, kid-friendly with jokes/emojis) or 'Normal' (formal, engaging without extras).
    """
    # Normalize mode (fix for lowercase/mismatch)
    style_mode = style_mode.upper().replace(" (FUN & EMOJIS)", "").strip()  # Handles "eli10" → "ELI10", etc.
    
    # Base instructions common to both modes
    base_instructions = f"""
    Task: Generate a coherent { 'paragraph' if word_count < 150 else 'short essay' } based on the original prompt.
    - Strictly maintain {sentiment} tone.
    - Exactly {word_count} words (±20).
    - Engaging and natural; structure: Start with a hook related to the prompt, develop 2-3 key ideas, and end with a reflective close.
    """

    if style_mode == "ELI10":
        # ELI10 mode: Kid-friendly, fun, with jokes/emojis
        flair = f"""
        - In ELI10 style: Explain like you're talking to a 10-year-old—use simple words, short sentences, fun examples, and easy ideas anyone can get. Avoid big words or boring facts; make it feel like a story or chat with a kid.
        """
        # Sentiment-specific enhancements
        if sentiment == "positive":
            extra_flair = "Add 3-5 fun jokes or puns to make it hilarious, and sprinkle in lots of excited emojis (like 😄, 🚀, 🎉) throughout to amp up the joy!"
        elif sentiment == "negative":
            extra_flair = "Weave in 2-3 dark humor jokes or ironic twists to heighten the gloom, and use grumpy/sad emojis (like 😞, 🌧️, 💔) to match the vibe."
        else:  # neutral
            extra_flair = "Keep it balanced with 2-3 light, observational jokes if they fit naturally, and add neutral emojis (like 📖, 🌤️, 🤔) sparingly for visual pop."
        
        generation_prompt = base_instructions + flair + f" - {extra_flair}\n\nExamples:\n" + get_eli10_examples(sentiment)
    else:
        # Normal mode: Formal, professional tone without ELI10/jokes/emojis
        extra_flair = """
        - Use a formal, professional tone suitable for adults: Clear, concise language with varied sentence structure. No slang, jokes, or emojis.
        """
        generation_prompt = base_instructions + extra_flair + "\n\nExamples:\n" + get_normal_examples(sentiment)

    generation_prompt += f"\nOriginal Prompt: {original_prompt}"
    
    try:
        response = model.generate_content(generation_prompt)
        if not response.text.strip():
            raise ValueError("Empty response from Gemini")
        return response.text.strip()
    except (ValueError, Exception) as e:
        st.error(f"Error in text generation: {e}")
        return "Generation failed. Please try again."

def get_eli10_examples(sentiment: str) -> str:
    """Return ELI10 examples based on sentiment."""
    examples = {
        "positive": """
    - Original (positive): "I love sunny days at the beach."
      Generated (positive): "Imagine the sun smiling down on the beach, making everything warm and sparkly like a giant hug from the sky 😄! You can splash in the waves that tickle your toes and build the tallest sandcastles ever—watch out, or the tide might say, 'Hey, that's my moat!' 😂. Kids everywhere are giggling and chasing seagulls—it's like the world's best playground 🎉! And at the end of the day, as the sun says goodnight with pretty colors, you feel super happy inside, ready for more adventures tomorrow 🚀. Why did the beach ball go to school? To get a little 'shore' education! 🌅"
        """,
        "negative": """
    - Original (negative): "Traffic jams ruin my commute."
      Generated (negative): "Ugh, picture being stuck in a huge line of cars, like a boring game where nobody moves 😞. The engine grumbles like a grumpy monster, and the clock just laughs at you while time drags on forever—talk about a 'traffic' light that never turns green! 🌧️. Horns beep like angry birds, and you're trapped with nowhere to go, making everything feel extra yucky and slow 💔. It's like the cars are having a never-ending staring contest. Why don't traffic jams ever break up? They're just too 'jammed' together! 😤 At least when you finally get home, you can sigh and say, 'Better luck tomorrow... maybe.'"
        """,
        "neutral": """
    - Original (neutral): "The weather today is mild."
      Generated (neutral): "Today's weather is just right, not too hot or too cold—like wearing your favorite cozy sweater outside 🌤️. Clouds float by like fluffy sheep, and the breeze is gentle, perfect for kicking a ball or reading a book under a tree 📖. It's a normal day where you can do whatever you want without sweating or shivering, keeping things simple and easy 🤔. Why did the cloud go to school? To get a little 'higher' education! ☁️ Either way, it's a chill day—no drama, just steady and nice, like your best buddy who's always there without stealing the show."
        """
    }
    return examples.get(sentiment, examples["neutral"])

def get_normal_examples(sentiment: str) -> str:
    """Return Normal mode examples based on sentiment."""
    examples = {
        "positive": """
    - Original (positive): "I love sunny days at the beach."
      Generated (positive): "Sunny days at the beach evoke a profound sense of joy and relaxation. The golden sunlight bathes the shoreline in warmth, creating an ideal setting for leisurely walks along the water's edge. Families gather to enjoy the rhythmic crash of waves, while the gentle sea breeze carries the faint scent of salt and sunscreen. Such moments remind us of nature's ability to recharge the spirit, fostering connections with loved ones and inspiring a deeper appreciation for life's simple pleasures."
        """,
        "negative": """
    - Original (negative): "Traffic jams ruin my commute."
      Generated (negative): "Traffic jams transform an ordinary commute into an exasperating ordeal, marked by stagnation and mounting irritation. Vehicles inch forward amid the cacophony of impatient horns, while exhaust fumes thicken the air, amplifying the sense of confinement. Precious time slips away in this involuntary standstill, disrupting schedules and eroding productivity. Ultimately, these delays underscore the vulnerabilities of urban mobility, leaving commuters drained and resentful upon arrival."
        """,
        "neutral": """
    - Original (neutral): "The weather today is mild."
      Generated (neutral): "The current weather conditions are mild, characterized by moderate temperatures and low humidity. This equilibrium supports a variety of daily activities, from outdoor errands to indoor pursuits, without the discomfort of extremes. Light cloud cover provides intermittent shade, and a subtle breeze maintains comfort levels. Overall, it presents an unremarkable yet conducive environment for routine proceedings."
        """
    }
    return examples.get(sentiment, examples["neutral"])

# Streamlit UI
st.set_page_config(page_title="AI Sentiment Text Generator", page_icon="🤖", layout="wide")

# Custom CSS for attractiveness
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 1rem;
    }
    .stButton > button {
        border-radius: 1rem;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stTextArea > label {
        font-weight: bold;
        color: #1f77b4;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
    .stSlider > label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Sentiment Text Generator")
st.markdown("Enter a prompt. Auto-detects sentiment and generates aligned text via Gemini 2.5 (updated for Oct 2025).")

# Sidebar
st.sidebar.header("⚙️ Options")
manual_sentiment = st.sidebar.selectbox("Sentiment Mode", ["Auto-detect", "Manual"], index=0)
if manual_sentiment == "Manual":
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", ["positive", "negative", "neutral"])
else:
    selected_sentiment = None

word_count = st.sidebar.slider("Word Count", 50, 500, 200, 50)

# Style Mode Selector
style_mode = st.sidebar.selectbox("Style Mode", ["ELI10 (Fun & Emojis)", "Normal (Formal)"], index=0)
st.sidebar.info("ELI10: Kid-friendly stories with jokes! | Normal: Professional tone.")

# Main input
original_prompt = st.text_area(
    "✍️ Enter your prompt",
    placeholder="e.g., 'I love exploring new cities!'",
    height=100,
    key="prompt_input"  # For clear functionality
)

# Clear button
if st.button("🗑️ Clear Prompt", type="secondary"):
    st.rerun()

# Generate button with Style Mode integration
col_gen1, col_gen2 = st.columns([3, 1])
with col_gen1:
    if st.button("🚀 Generate Text", type="primary", use_container_width=True, key="generate"):
        if original_prompt.strip():
            # Sentiment logic
            if manual_sentiment == "Manual":
                sentiment = selected_sentiment
                st.info(f"Using manual sentiment: **{sentiment}**")
            else:
                with st.spinner("🔍 Analyzing sentiment..."):
                    sentiment = get_sentiment(original_prompt)
                st.success(f"Detected Sentiment: **{sentiment}**")

            with st.spinner("✨ Generating aligned text..."):
                # Fixed: Pass cleaned style_mode
                clean_style = "ELI10" if "ELI10" in style_mode.upper() else "Normal"
                generated_text = generate_sentiment_aligned_text(sentiment, original_prompt, word_count, clean_style)

            # Display
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("### 📝 Original Prompt")
                st.write(original_prompt)
                st.markdown(f"**Sentiment:** {sentiment}")
                st.markdown(f"**Style:** {style_mode}")
            with col2:
                st.markdown("### 🎨 Generated Text")
                st.write(generated_text)
            
            gen_word_count = len(generated_text.split())
            st.caption(f"Generated ~{gen_word_count} words")
            
            # Export/Share Feature
            st.markdown("### 📤 Export Options")
            
            # Download as PDF (robust: try import, fallback to TXT)
            def create_export():
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.lib.colors import HexColor
                    import io
                
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                
                    # Custom style for title with color
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=20,
                        spaceAfter=30,
                        alignment=1,  # Center
                        textColor=HexColor('#1f77b4')  # Blue color
                    )
                
                    # Custom normal style with better font
                    normal_style = ParagraphStyle(
                        'CustomNormal',
                        parent=styles['Normal'],
                        fontSize=11,
                        spaceAfter=12,
                        textColor=HexColor('#333333')
                    )
                
                    story = []
                    story.append(Paragraph("🤖 AI Sentiment Text Generator", title_style))
                    story.append(Spacer(1, 0.3 * inch))
                    story.append(Paragraph(f"<b>Original Prompt:</b> {original_prompt}", normal_style))
                    story.append(Spacer(1, 0.15 * inch))
                    story.append(Paragraph(f"<b>Sentiment:</b> {sentiment}", normal_style))
                    story.append(Spacer(1, 0.15 * inch))
                    story.append(Paragraph(f"<b>Style:</b> {style_mode}", normal_style))
                    story.append(Spacer(1, 0.3 * inch))
                    story.append(Paragraph("<b>Generated Text:</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.15 * inch))
                    story.append(Paragraph(generated_text, normal_style))
                    story.append(Spacer(1, 0.4 * inch))
                    story.append(Paragraph(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} | ~{gen_word_count} words", styles['Italic']))
                
                    doc.build(story)
                    buffer.seek(0)
                    return buffer.getvalue(), "application/pdf", f"sentiment_text_{sentiment}_{style_mode.lower().replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
                except ImportError:
                    # Fallback: Plain text file
                    fallback_text = f"AI Sentiment Text Generator\n\nOriginal Prompt: {original_prompt}\nSentiment: {sentiment}\nStyle: {style_mode}\n\nGenerated Text:\n{generated_text}\n\nGenerated on {time.strftime('%Y-%m-%d %H:%M:%S')} | ~{gen_word_count} words"
                    return fallback_text.encode('utf-8'), "text/plain", f"sentiment_text_{sentiment}_{style_mode.lower().replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            data, mime, filename = create_export()
            st.download_button(
                label=f"💾 Download as PDF 📄" if "pdf" in filename.lower() else "💾 Download as TXT 📄",
                data=data,
                file_name=filename,
                mime=mime,
                type="primary",
                use_container_width=True
            )
            if "pdf" not in filename.lower():
                st.warning("💡 PDF needs 'reportlab'—installed? Falling back to TXT for now.")
        else:
            st.warning("Please enter a prompt.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & Gemini. Updated for 2025 model availability.")