from openai import OpenAI
import streamlit as st
from io import BytesIO

# Set your OpenAI API key
openai_api_key = 'sk-proj-Nkfn2KX6fO9X15iU0f2eT3BlbkFJC604Ir7QpY92I6v3Ydnk'
client = OpenAI(api_key=openai_api_key)

def gpt4_analysis(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        max_tokens=1000,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def gpt4_insights(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def analyze_sentiment(text):
    prompt = f"Perform sentiment analysis on the following text:\n\n{text}\n\nSentiment:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def perform_topic_modeling(text):
    prompt = f"Identify the main topics discussed in the following text:\n\n{text}\n\nTopics:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def extract_entities(text):
    prompt = f"Extract entities from the following text and categorize them as names, organizations, locations, or industry-specific terms:\n\n{text}\n\nEntities:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def extract_keywords(text):
    prompt = f"Extract important keywords and phrases from the following text:\n\n{text}\n\nKeywords:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def generate_insights(sentiment, topics, entities, keywords):
    prompt = f"Generate insights based on the following analyses:\n\nSentiment: {sentiment}\n\nTopics: {topics}\n\nEntities: {entities}\n\nKeywords: {keywords}\n\nInsights:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_insights(messages)

def summarize_insights(insights):
    prompt = f"Summarize the following insights in 4096 tokens or less:\n\n{insights}\n\nSummary:"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_insights(messages)

def transcribe_audio(file):
    response = client.audio.transcriptions.create(
        file=file,
        model="whisper-1",
        response_format="text",
    )
    return response['text']

def text_to_speech(text, filename):
    response = client.audio.speech.create(
        model="tts-1",
        input=text,
        voice="shimmer",
    )
    response.write_to_file(filename)
    return filename

# Apply custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    local_css("style.css")
    st.markdown("<header>🎙️ Radio Interview Transcript Analyzer</header>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a transcript or an audio file", type=["txt", "wav", "mp3"])
    
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            transcript = uploaded_file.read().decode("utf-8")
        else:
            transcript = transcribe_audio(uploaded_file)
        
        st.text_area("Transcript", transcript, height=200)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if st.button("Analyze"):
            with st.spinner('Analyzing the transcript...'):
                sentiment = analyze_sentiment(transcript)
                topics = perform_topic_modeling(transcript)
                entities = extract_entities(transcript)
                keywords = extract_keywords(transcript)
                insights = generate_insights(sentiment, topics, entities, keywords)
                summary = summarize_insights(insights)
                st.session_state['results'] = (sentiment, topics, entities, keywords, insights, summary)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            show_results()
            
        if 'results' in st.session_state:
            show_results()

def show_results():
    sentiment, topics, entities, keywords, insights, summary = st.session_state['results']
    
    st.subheader("Analysis Results")
    with st.expander("Sentiment Analysis"):
        st.write(sentiment)
    with st.expander("Topic Modeling"):
        st.write(topics)
    with st.expander("Entity Extraction"):
        st.write(entities)
    with st.expander("Keyword Analysis"):
        st.write(keywords)
    
    st.subheader("Generated Insights")
    st.write(insights)
    
    # Generate speech output for summary
    summary_audio = text_to_speech(summary, "summary.mp3")
    
    # Sidebar with images and download buttons
    st.sidebar.markdown("<h2>Resources</h2>", unsafe_allow_html=True)
    st.sidebar.image("radio_insights.webp", use_column_width=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2>Download</h2>", unsafe_allow_html=True)
    
    insights_file = BytesIO(insights.encode())
    summary_audio_file = BytesIO(open("summary.mp3", "rb").read())
    
    st.sidebar.download_button(
        label="📄 Download Insights",
        data=insights_file,
        file_name="insights.txt",
        mime="text/plain",
        help="Download the generated insights"
    )
    
    st.sidebar.download_button(
        label="🔊 Download Audio",
        data=summary_audio_file,
        file_name="summary.mp3",
        mime="audio/mpeg",
        help="Download the audio summary"
    )

if __name__ == "__main__":
    app()
