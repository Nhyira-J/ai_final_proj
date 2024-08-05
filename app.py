import os
import requests
import streamlit as st

from openai import OpenAI
from io import BytesIO


from dotenv import load_dotenv
from pydub import AudioSegment


# Set your OpenAI API key
load_dotenv()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

openai_api_key = os.getenv("OPENAI_API_KEY") # or st.secrets["OPERNAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Maximum audio file size (30 MB)
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 30 MB in bytes

def gpt4_analysis(prompt):
    """
    Analyzes the given prompt using the GPT-4o-mini model.

    Parameters:
    - prompt (str): The prompt to be analyzed.

    Returns:
    - str: The generated response from the GPT-4o-mini model.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        max_tokens=1500,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def gpt4_insights(prompt):
    """
    Generates insights using GPT-4o Mini model.

    Args:
        prompt (str): The prompt for generating insights.

    Returns:
        str: The generated insights.

    Raises:
        None

    Example:
        >>> prompt = "What are the benefits of exercise?"
        >>> insights = gpt4_insights(prompt)
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text.

    Parameters:
    text (str): The text to perform sentiment analysis on.

    Returns:
    str: The sentiment analysis result.
    """
    prompt = f"Perform sentiment analysis on the following text:\n\n{text}\n\nSentiment:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def perform_topic_modeling(text):
    """
    Perform topic modeling on the given text.

    Args:
        text (str): The text to analyze.

    Returns:
        str: The identified main topics discussed in the text.
    """
    prompt = f"Identify the main topics discussed in the following text:\n\n{text}\n\nTopics:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def extract_entities(text):
    """
    Extracts entities from the given text and categorizes them as names, organizations, locations, or industry-specific terms.

    Args:
        text (str): The text from which entities need to be extracted.

    Returns:
        list: A list of extracted entities.

    """
    prompt = f"Extract entities from the following text and categorize them as names, organizations, locations, or industry-specific terms:\n\n{text}\n\nEntities:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def extract_keywords(text):
    """
    Extracts important keywords and phrases from the given text.

    Parameters:
    text (str): The text from which keywords and phrases need to be extracted.

    Returns:
    list: A list of important keywords and phrases extracted from the text.
    """
    prompt = f"Extract important keywords and phrases from the following text:\n\n{text}\n\nKeywords:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_analysis(messages)

def generate_insights(sentiment, topics, entities, keywords, custom_insights=None):
    """
    Generate compelling and actionable insights from the given interview analysis.

    Parameters:
    - sentiment (str): The sentiment analysis of the interview.
    - topics (str): The topics identified in the interview.
    - entities (str): The entities identified in the interview.
    - keywords (str): The keywords identified in the interview.
    - custom_insights (str, optional): Custom insights provided by the user.

    Returns:
    - insights (str): The generated insights.

    The function generates insights based on the provided interview analysis. If custom insights are provided, they will be used exclusively. Otherwise, a base prompt template will be used. The insights should be tailored to a journalist aiming to produce a captivating and informative article. The generated insights prioritize newsworthy events, human-interest stories, controversial topics, and expert-driven quotes and insights. Additional considerations include the target audience, story angle, visuals, and fact-checking.

    Example usage:
    ```
    sentiment = "Positive"
    topics = "Technology"
    entities = "Apple, iPhone"
    keywords = "innovation, market share"
    custom_insights = "Focus on Apple's latest product launch and its impact on the smartphone market."

    insights = generate_insights(sentiment, topics, entities, keywords, custom_insights)
    print(insights)
    ```
    """
    
    # Check if custom insights are provided
    if custom_insights:
        # Use custom insights exclusively if provided
        prompt = (
            f"Generate compelling and actionable insights from the following interview analysis. **Focus on uncovering hidden narratives, potential controversies, and expert opinions.\n\n"
            f"User-specified Insights: {custom_insights}\n\n"
            f"Generate insights using these instructions provided. Do not be suggestive but be accurate and concise in your responses."
        )
    else:
        # Use base prompt template if no custom insights are provided
        prompt = (
            f"Generate compelling and actionable insights from the following interview analysis. **Focus on uncovering hidden narratives, potential controversies, and expert opinions.** "
            f"The insights should be tailored to a journalist aiming to produce a captivating and informative article. Prioritize insights that are:\n"
            f"* **Newsworthy:** Identify significant events, trends, or statements.\n"
            f"* **Human-interest:** Highlight relatable stories or emotional appeals.\n"
            f"* **Controversial:** Point out potential conflicts or opposing viewpoints.\n"
            f"* **Expert-driven:** Leverage quotes and insights from knowledgeable sources.\n"
            f"\n**Key Analysis:**\n"
            f"Sentiment Analysis: {sentiment}\n"
            f"Topic Modeling: {topics}\n"
            f"Entities: {entities}\n"
            f"Keywords: {keywords}\n\n"
            f"**Additional Considerations:**\n"
            f"* **Target Audience:** Consider the interests and knowledge level of the target audience.\n"
            f"* **Story Angle:** Suggest potential angles or focus areas for the article.\n"
            f"* **Visuals:** Identify opportunities for incorporating images or multimedia.\n"
            f"* **Fact-checking:** Indicate potential areas requiring further verification.\n"
            f"\nGenerate insights using these instructions provided. Do not be suggestive but be accurate and concise in your responses."
        )

    # Construct the message to be sent to the API
    messages = [
        {"role": "system", "content": "You are a Interview Insight Generator.Follow the prompts to give Journalists goo results:"},
        {"role": "user", "content": prompt}
    ]

    return gpt4_insights(messages)


def summarize_insights(insights):
    """
    Summarizes the given insights in 4096 tokens or less.

    Args:
        insights (str): The insights to be summarized.

    Returns:
        str: The summary of the insights.
    """
    prompt = f"Summarize the following insights in 4096 tokens or less:\n\n{insights}\n\nSummary:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return gpt4_insights(messages)

def transcribe_audio(file_path):
    """
    Transcribes the audio file located at the given file path.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        str: The transcribed text from the audio file.
    """
    with open(file_path, "rb") as file:
        response = client.audio.transcriptions.create(
            file=file,
            model="whisper-1",
            response_format="text",
        )
    return response

def text_to_speech(text, filename):
    """
    Converts the given text to speech and saves it as an audio file.

    Args:
        text (str): The text to be converted to speech.
        filename (str): The name of the audio file to be saved.

    Returns:
        str: The filename of the saved audio file.

    Raises:
        None
    """
    if not os.path.exists(filename):
        response = client.audio.speech.create(
            model="tts-1",
            input=text,
            voice="shimmer",
        )
        response.write_to_file(filename)
    return filename

def download_audio(url):
    """
    Downloads an audio file from the given URL.

    Args:
        url (str): The URL of the audio file to be downloaded.

    Returns:
        str: The filename of the downloaded audio file.

    Raises:
        requests.HTTPError: If the request to the URL fails or returns an error status code.
    """
    headers = {
        "User-Agent": "RadioInterviewAnalyzer/1.0 (jemima.nhyira@ashesi.edu.gh) Script for educational purposes"
    }
    local_filename = url.split('/')[-1]
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded audio file: {local_filename}")
    return local_filename

def convert_to_mp3(input_file):
    """
    Converts an audio file to MP3 format.

    Parameters:
    input_file (str): The path to the input audio file.

    Returns:
    str: The path to the converted MP3 audio file.
    """

    audio = AudioSegment.from_file(input_file)
    output_file = "converted_audio.mp3"
    audio.export(output_file, format="mp3")
    return output_file


def local_css(file_name):
    """
    Applies local CSS styles to the Streamlit app.

    Parameters:
    - file_name (str): The path to the CSS file.

    Returns:
    None
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def reset_state():
    """
    Resets the state of the session by deleting all keys in the session_state dictionary.
    """
   
    for key in st.session_state.keys():
        del st.session_state[key]
    
    for key in st.session_state.keys():
        print(st.session_state[key])

def sidebar_content():
    """
    Function to generate the content for the sidebar in the application.

    This function adds various elements to the sidebar, including headings, images, and download buttons.

    Returns:
        None
    """
    st.sidebar.markdown("<h2>Resources</h2>", unsafe_allow_html=True)
    st.sidebar.image("radio_insights.webp", caption="Professional Radio Studio", use_column_width=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2>Download</h2>", unsafe_allow_html=True)

    # Ensure results are ready before allowing download
    if st.session_state['results']:
        insights_file = BytesIO(st.session_state['results'][4].encode())
        summary_audio_file = BytesIO(open(st.session_state['summary_audio_file'], "rb").read())

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

def app():
    """
    This is a Streamlit application for analyzing radio interview transcripts.
    Functions:
    - app(): Main function that runs the Flask application.
    - sidebar_content(): Function to handle the sidebar content.
    - reset_state(): Function to reset the session state.
    - local_css(file_path): Function to load local CSS file.
    - download_audio(audio_url): Function to download audio file from a URL.
    - convert_to_mp3(file_path): Function to convert audio file to MP3 format.
    - transcribe_audio(file_path): Function to transcribe audio file.
    - analyze_sentiment(transcript): Function to analyze sentiment of a transcript.
    - perform_topic_modeling(transcript): Function to perform topic modeling on a transcript.
    - extract_entities(transcript): Function to extract entities from a transcript.
    - extract_keywords(transcript): Function to extract keywords from a transcript.
    - generate_insights(sentiment, topics, entities, keywords, custom_insights): Function to generate insights from analysis results.
    - summarize_insights(insights): Function to summarize insights.
    - text_to_speech(text, file_name): Function to convert text to speech and save as an audio file.
    """
    local_css("style.css")
    
    st.markdown("<header>🎙️ Radio Interview Transcript Analyzer</header>", unsafe_allow_html=True)

    # Initialize session state
    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = None
    if 'audio_file' not in st.session_state:
        st.session_state['audio_file'] = None
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'summary_audio_file' not in st.session_state:
        st.session_state['summary_audio_file'] = None

    # Tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["Upload Interview", "Analyse Transcript", "View Insights"])

    with tab1:
        st.header("Submit Interview Files")
        st.write("Provide an audio file URL or upload an audio file (Max size: 50 MB).")

        # Input for audio URL or file upload
        audio_url = st.text_input("Enter the URL of an audio file (optional):")
        uploaded_file = st.file_uploader("Or upload an audio file (Max size: 50MB)", type=["mp3", "wav", "ogg", "flac"], accept_multiple_files=False)

        # Handle audio URL download
        if audio_url:
            try:
                audio_file = download_audio(audio_url)
                if os.path.getsize(audio_file) > MAX_AUDIO_SIZE:
                    st.error("The audio file size exceeds the maximum limit of 30 MB.")
                    os.remove(audio_file)
                else:
                    st.session_state['audio_file'] = audio_file
            except Exception as e:
                st.error(f"Error downloading the audio: {e}")
                return

        # Handle uploaded file
        elif uploaded_file:
            if uploaded_file.size > MAX_AUDIO_SIZE:
                st.error("The audio file size exceeds the maximum limit of 30 MB.")
            else:
                audio_bytes = uploaded_file.read()
                with open(uploaded_file.name, "wb") as f:
                    f.write(audio_bytes)
                st.session_state['audio_file'] = uploaded_file.name
                st.success("Audio Uploaded Successfully!")

        # Convert non-MP3 files to MP3 if needed
        if st.session_state['audio_file']:
            if not st.session_state['audio_file'].endswith(".mp3"):
                try:
                    st.session_state['audio_file'] = convert_to_mp3(st.session_state['audio_file'])
                    st.success("Converted audio to MP3 format.")
                except Exception as e:
                    st.error(f"Error converting audio: {e}")
                    return

            # Transcribe the audio
            try:
                st.session_state['transcript'] = transcribe_audio(st.session_state['audio_file'])
                os.remove(st.session_state['audio_file'])  # Remove the downloaded/converted file after transcription
                st.success("Audio Transcribed Successfully! Click on the next tab to analyze the transcript.")
            except Exception as e:
                st.error(f"Error transcribing the audio: {e}")
                return

    with tab2:
        if st.session_state['transcript']:
            st.header("Analyse Interview Transcript")
            st.text_area("Transcript", st.session_state['transcript'], height=200)
            st.markdown("<hr>", unsafe_allow_html=True)

            # Custom insights input
            custom_insights = st.text_input("Enter specific insights you want to be included (optional):")

            # Analyze button
            if st.button("Analyze"):
                with st.spinner('Analyzing the transcript...'):
                    sentiment = analyze_sentiment(st.session_state['transcript'])
                    topics = perform_topic_modeling(st.session_state['transcript'])
                    entities = extract_entities(st.session_state['transcript'])
                    keywords = extract_keywords(st.session_state['transcript'])
                    insights = generate_insights(sentiment, topics, entities, keywords, custom_insights)
                    summary = summarize_insights(insights)
                    st.session_state['results'] = (sentiment, topics, entities, keywords, insights, summary)

                    # Generate audio for the summary only once and store it in session state
                    st.session_state['summary_audio_file'] = text_to_speech(summary, "summary.mp3")                   
                # Add a button to navigate to the "Show Results" tab
                st.success("Analysis complete! Click on View Insights to see the results.")

    with tab3:
        if st.session_state['results']:
            st.header("View Insights")

            # Section dividers for a better visual appearance
            divider_html = "<hr style='border:1px solid #ff5733' />"  # Changed color and thickness for a professional look
            st.markdown(divider_html, unsafe_allow_html=True)
            
            sentiment, topics, entities, keywords, insights, summary = st.session_state['results']

            st.subheader("Analysis Results")
            st.markdown(divider_html, unsafe_allow_html=True)
            
            with st.expander("Sentiment Analysis"):
                st.write(sentiment)
            
            with st.expander("Topic Modeling"):
                st.write(topics)
            
            with st.expander("Entity Extraction"):
                st.write(entities)
           
            with st.expander("Keyword Analysis"):
                st.write(keywords)
            
            st.markdown(divider_html, unsafe_allow_html=True)

            st.subheader("Generated Insights")
            st.write(insights)

            # Button to reset the app
            if st.button("Start New Analysis"):
                reset_state()
                st.rerun()

   
    # Call sidebar content function
    sidebar_content()

    st.markdown("<footer>🎙️ Radio Interview Transcript Analyzer. Made by Nhyira and Konadu: We Love AI 🤖</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    app()


