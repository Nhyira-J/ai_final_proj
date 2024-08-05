# 🎙️ Radio Interview Transcript Analyzer

Welcome to the Radio Interview Transcript Analyzer! This application leverages advanced AI models to transform radio interviews into actionable insights, helping journalists, media professionals, and researchers extract valuable information quickly and efficiently.

## Team Info
We are Group 20. The team is made up of:
- Jemima Nhyira Antwi
- Jemima Konadu Antwi

## Link to the video
https://youtu.be/fG3-kAtKX2I

## 🖥️ Link to Deployed App
https://aifinalproj-7cekd82swfdluhqm9a4834.streamlit.app

## 🌟 Overview

The Radio Interview Transcript Analyzer is a powerful tool that automates the process of analyzing radio interviews. By combining state-of-the-art natural language processing techniques with user-friendly design, the app provides comprehensive analyses, including sentiment analysis, topic modeling, entity extraction, and keyword analysis. These features enable users to uncover hidden narratives, identify key themes, and generate insights tailored for impactful journalism.

## 🧩 Features

- **Audio File Handling:** Supports uploading audio files in various formats (MP3, WAV, OGG, FLAC) or providing a URL for analysis. The maximum supported file size is 50 MB.
- **Automatic Conversion:** Converts non-MP3 audio files to MP3 format using `pydub`.
- **Speech-to-Text:** Transcribes audio files into text using OpenAI's Whisper model.
- **Sentiment Analysis:** Analyzes the sentiment expressed in the interview, providing insights into opinions and emotions.
- **Topic Modeling:** Identifies and categorizes the main topics discussed in the interview.
- **Entity Extraction:** Recognizes and extracts entities such as names, organizations, and locations.
- **Keyword Analysis:** Extracts important keywords and phrases to highlight essential themes.
- **Insight Generation:** Generates compelling insights tailored for journalists, with options for custom user-provided insights.
- **Text-to-Speech:** Converts generated insights into an audio summary.
- **Downloadable Results:** Provides options to download insights and audio summaries for offline access.

## 📋 Requirements

- Python 3.7 or higher
- Streamlit
- OpenAI Python Client
- `pydub` library
- `dotenv` library

## 🔧 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Nhyira-J/ai_final_proj.git
   cd ai_final_proj
   ```
2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**

   - Create a `.env` file in the project root directory.
   - Add your OpenAI API key to the `.env` file:

     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
4. **Run the Application**

   ```bash
   streamlit run app.py
   ```
5. **Access the App**

   Open your web browser and go to `http://localhost:8501` to access the app.

## 🚀 Usage

1. **Upload Interview**: Provide an audio file URL or upload an audio file through the "Upload Interview" tab. The app supports files up to 50 MB.
2. **Analyze Transcript**: Once the audio is uploaded and transcribed, navigate to the "Analyze Transcript" tab. The transcript will be displayed, and you can optionally specify custom insights to be included.
3. **View Insights**: After analysis, switch to the "View Insights" tab to see the results. The analysis includes sentiment analysis, topic modeling, entity extraction, keyword analysis, and generated insights.
4. **Download Results**: Use the sidebar to download the generated insights as a text file and the audio summary.
5. **Start New Analysis**: Click the "Start New Analysis" button to reset the app and begin a new session.

## 🤖 Technologies Used

- **OpenAI GPT-4o Mini Model**: For sentiment analysis, topic modeling, entity extraction, keyword analysis, and insight generation.
- **OpenAI Whisper Model**: For speech-to-text transcription of audio files.
- **OpenAI Text-to-Speech**: For converting generated insights into an audio summary.
- **Streamlit**: For building the interactive web application.
- **Pydub**: For audio file conversion and processing.

## 📚 Detailed Functionality

### Functions Overview

- **gpt4_analysis**: Analyzes a given prompt using the GPT-4o Mini model.
- **gpt4_insights**: Generates insights based on a specified prompt.
- **analyze_sentiment**: Performs sentiment analysis on a given text.
- **perform_topic_modeling**: Identifies main topics discussed in a text.
- **extract_entities**: Extracts entities like names, organizations, and locations.
- **extract_keywords**: Extracts important keywords and phrases.
- **generate_insights**: Generates actionable insights from the analysis results, tailored for journalists.
- **summarize_insights**: Summarizes insights into a concise format.
- **transcribe_audio**: Transcribes an audio file into text.
- **text_to_speech**: Converts text into an audio file using text-to-speech.
- **download_audio**: Downloads an audio file from a URL.
- **convert_to_mp3**: Converts audio files to MP3 format using `pydub`.
- **local_css**: Applies local CSS styles to the Streamlit app.
- **reset_state**: Resets the session state for a new analysis.
- **sidebar_content**: Handles the sidebar content for resources and downloads.

### App Flow

1. **Initialize Session State**: Sets up the initial state for audio files, transcripts, results, and audio summaries.
2. **File Upload and Conversion**: Handles file uploads and URL downloads, ensuring audio files are converted to MP3 format if necessary.
3. **Transcription**: Transcribes audio files into text for analysis.
4. **Analysis and Insights**: Performs various analyses on the transcript and generates insights.
5. **Results Display and Downloads**: Displays analysis results and provides download options for insights and audio summaries.

## 💡 Future Improvements

- **Multi-language Support**: Add support for analyzing interviews in multiple languages.
- **Enhanced Visualization**: Introduce visualizations for better representation of analysis results.
- **User Feedback**: Implement a feedback mechanism to improve analysis accuracy and user experience.

## 🏆 Contributors

- **Jemima Nhyira Antwi** - Developer and AI Enthusiast
- **Jemima Konadu Antwi**- Collaborator

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 💌 Acknowledgments

- Thanks to the OpenAI team for providing powerful models and tools.
- Special appreciation to the Streamlit community for creating an excellent platform for building data apps.
