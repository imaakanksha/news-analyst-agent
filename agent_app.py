# Autonomous Multimodal News Analyst Agent

import streamlit as st
import google.generativeai as genai
import feedparser
import pyttsx3
import re
import requests
from PIL import Image
import io
import datetime # To handle timestamps if needed
import time # For potential delays/retries
from bs4 import BeautifulSoup # For parsing HTML content to find images

# --- Configuration & Initialization ---
st.set_page_config(page_title="AI News Analyst Agent", layout="wide")
st.title("📰 Autonomous Multimodal News Analyst Agent")
st.caption("Your AI agent for monitoring, analyzing, and reporting news.")

# Load API key from Streamlit secrets for secure deployment
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
    # Initialize the Gemini model (using 1.5 flash for text and vision)
    model = genai.GenerativeModel("gemini-1.5-flash")
    # st.sidebar.success("Gemini API Key Loaded Successfully!") # Keep sidebar cleaner
except KeyError:
    st.error("IMPORTANT: GOOGLE_API_KEY not found in Streamlit secrets.")
    st.error("Please add it via your Streamlit Cloud app settings.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}.")
    st.error("Please ensure your GOOGLE_API_KEY secret is correct and valid.")
    st.stop()

# Initialize Text-to-Speech engine (optional)
tts_engine = None
try:
    tts_engine = pyttsx3.init()
    voices = tts_engine.getProperty("voices")
    if not voices:
        raise RuntimeError("No TTS voices found")
except Exception as e:
    st.sidebar.warning(f"Could not initialize TTS engine: {e}. Speech output disabled.")
    tts_engine = None

# --- Helper Functions ---
def fetch_and_parse_feed(url):
    """Fetches and parses an RSS feed, returning the parsed feed object."""
    if not url:
        return None, "Please enter an RSS feed URL."
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        if feed.bozo:
            st.warning(f"Warning: Feed may be ill-formed. Bozo exception: {feed.bozo_exception}")
        if not feed.entries:
             return None, "Feed parsed, but no entries found or feed is empty."
        return feed, None
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching feed: {e}"
    except Exception as e:
        return None, f"Error parsing feed: {e}"

def filter_articles(entries, keywords_str, last_checked_id):
    """Filters entries based on keywords and identifies new entries."""
    new_articles = []
    latest_entry_id = None
    keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]

    if entries:
        latest_entry_id = entries[0].get("id", entries[0].get("link"))
        for entry in entries:
            entry_id = entry.get("id", entry.get("link"))
            if entry_id == last_checked_id:
                break
            title = entry.get("title", "").lower()
            summary = entry.get("summary", "").lower()
            content_text = title + " " + summary
            if keywords:
                if any(keyword in content_text for keyword in keywords):
                    new_articles.append(entry)
            else:
                new_articles.append(entry)
    return new_articles[::-1], latest_entry_id

def generate_with_gemini(prompt_parts, is_json_output=False):
    """Generic function to call Gemini API with multiple parts (text/image) and handle errors."""
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        generation_config = None
        if is_json_output:
             generation_config=genai.types.GenerationConfig(response_mime_type="application/json")

        response = model.generate_content(
             prompt_parts,
             safety_settings=safety_settings,
             generation_config=generation_config,
             stream=False
        )

        if not response.candidates:
             block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
             return f"Content generation blocked. Reason: {block_reason}"
        if response.candidates[0].finish_reason != 'STOP':
             return f"Content generation stopped. Reason: {response.candidates[0].finish_reason.name}"
        if not response.parts:
             return "Error: Received an empty response from the API."

        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return f"An error occurred during generation: {e}"

def extract_image_url(entry):
    """Attempts to extract the first usable image URL from a feed entry."""
    if "media_content" in entry and entry.media_content:
        for media in entry.media_content:
            if media.get("medium") == "image" and media.get("url"):
                return media["url"]
    if "media_thumbnail" in entry and entry.media_thumbnail:
        for thumbnail in entry.media_thumbnail:
            if thumbnail.get("url"):
                return thumbnail["url"]
    if "links" in entry:
        for link in entry.links:
            if link.get("type", "").startswith("image/") and link.get("href"):
                return link["href"]
    html_content = ""
    if "content" in entry and entry.content:
        html_content = entry.content[0].get("value", "")
    if not html_content and "summary" in entry:
        html_content = entry.summary
    if html_content:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            img_tag = soup.find("img")
            if img_tag and img_tag.get("src"):
                return img_tag["src"]
        except Exception:
            pass
    return None

def fetch_image(url):
    """Fetches an image from a URL and returns a PIL Image object."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        if 'image' not in content_type:
            return None, f"URL content type is not image ({content_type})"
        image = Image.open(io.BytesIO(response.content))
        return image, None
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching image: {e}"
    except Exception as e:
        return None, f"Error processing image: {e}"

def get_article_context(entry):
    """Extracts and cleans text content from an entry for analysis or Q&A."""
    title = entry.get("title", "")
    content = entry.get("content", [{}])[0].get("value", "")
    summary_text = entry.get("summary", "")
    text_content = content if len(content) > len(summary_text) + 50 else summary_text
    if len(text_content) < 100: text_content = title + ". " + summary_text
    text_content = re.sub("<[^>]*>", " ", text_content)
    text_content = re.sub("\s+", " ", text_content).strip()
    return text_content if text_content else title # Fallback to title if no other text

def analyze_article_multimodal(entry):
    """Performs text and image analysis using Gemini."""
    analysis_results = {
        "summary": "Analysis pending...",
        "sentiment": "N/A",
        "entities": [],
        "image_url": None,
        "image_analysis": None
    }
    text_to_analyze = get_article_context(entry)

    if not text_to_analyze:
        analysis_results["summary"] = "Error: No text content found for analysis."
        return analysis_results

    # --- Text Analysis --- #
    combined_prompt = f"Analyze the following news article text. Provide the output as a JSON object with three keys: \"summary\", \"sentiment\", and \"entities\".\n\n1.  **summary**: Provide a concise summary (2-3 sentences) of the key information.\n2.  **sentiment**: Classify the overall sentiment as strictly one of: \"Positive\", \"Negative\", or \"Neutral\".\n3.  **entities**: Extract key named entities (people, organizations, locations) mentioned. Return them as a list of strings.\n\nArticle Text:\n---\n{text_to_analyze}---"
    analysis_json_str = generate_with_gemini([combined_prompt], is_json_output=True)

    if analysis_json_str.startswith("An error occurred") or analysis_json_str.startswith("Content generation") or analysis_json_str.startswith("Error:"):
        analysis_results["summary"] = analysis_json_str
    else:
        try:
            cleaned_json_str = re.sub(r"^```json\n?|\n?```$", "", analysis_json_str, flags=re.MULTILINE).strip()
            parsed_json = genai.types.GenerationResponse.from_json(cleaned_json_str).candidates[0].content.parts[0].json
            analysis_results.update(parsed_json)
            if not isinstance(analysis_results.get("summary"), str): analysis_results["summary"] = "Invalid summary format."
            if analysis_results.get("sentiment") not in ["Positive", "Negative", "Neutral"]: analysis_results["sentiment"] = "Invalid sentiment."
            if not isinstance(analysis_results.get("entities"), list): analysis_results["entities"] = ["Invalid entities format."]
        except Exception as e:
            st.warning(f"Failed to parse JSON text analysis: {e}. Raw: {analysis_json_str}")
            analysis_results["summary"] = f"Error parsing text analysis: {e}"

    # --- Image Analysis --- #
    image_url = extract_image_url(entry)
    analysis_results["image_url"] = image_url
    if image_url:
        pil_image, fetch_error = fetch_image(image_url)
        if fetch_error:
            analysis_results["image_analysis"] = f"Failed to fetch image: {fetch_error}"
        elif pil_image:
            try:
                vision_prompt = [
                    "Describe this image briefly and explain how it relates to the following news article summary. If it doesn't seem related, just describe the image.",
                    pil_image,
                    "\n\nArticle Summary:\n" + analysis_results.get("summary", "Summary not available.")
                ]
                image_desc = generate_with_gemini(vision_prompt)
                analysis_results["image_analysis"] = image_desc
            except Exception as e:
                 analysis_results["image_analysis"] = f"Error during image analysis: {e}"
        else:
             analysis_results["image_analysis"] = "Image found but could not be processed."

    return analysis_results

def ask_gemini_about_article(question, article_context):
    """Asks Gemini a question based on the provided article context."""
    if not question:
        return "Please enter a question."
    if not article_context:
        return "Cannot answer without article context."

    prompt = f"Based *only* on the following article text, answer the user's question.\n\nArticle Text:\n---\n{article_context}\n---\n\nUser Question: {question}\n\nAnswer:"

    answer = generate_with_gemini([prompt]) # Pass prompt as list
    return answer

def speak_text(text):
    """Uses pyttsx3 to speak the provided text if engine is available."""
    if tts_engine and text:
        try:
            tts_engine.stop()
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            st.error(f"TTS Error: {e}")
    elif not tts_engine:
        st.warning("TTS engine not available.")

# --- Session State Initialization ---
if 'rss_url' not in st.session_state: st.session_state.rss_url = "http://rss.cnn.com/rss/cnn_topstories.rss"
if 'keywords' not in st.session_state: st.session_state.keywords = ""
if 'analyzed_articles' not in st.session_state: st.session_state.analyzed_articles = []
if 'last_checked_entry_id' not in st.session_state: st.session_state.last_checked_entry_id = None
if 'agent_status' not in st.session_state: st.session_state.agent_status = "Idle"
if 'pending_analysis' not in st.session_state: st.session_state.pending_analysis = []
if 'qa_answers' not in st.session_state: st.session_state.qa_answers = {} # Store answers {article_index: answer}

# --- Sidebar for Agent Configuration ---
st.sidebar.header("Agent Configuration")
st.session_state.rss_url = st.sidebar.text_input("Enter RSS Feed URL:", st.session_state.rss_url)
st.session_state.keywords = st.sidebar.text_input("Keywords (optional, comma-separated):", st.session_state.keywords)

run_agent_button = st.sidebar.button("Run Agent Check Now")
st.sidebar.caption(f"Status: {st.session_state.agent_status}")
if st.session_state.last_checked_entry_id:
    st.sidebar.caption(f"Last checked ID: ...{st.session_state.last_checked_entry_id[-20:]}")

# --- Main Agent Logic ---
if run_agent_button and st.session_state.agent_status == "Idle":
    st.session_state.agent_status = "Fetching feed..."
    st.rerun()

if st.session_state.agent_status == "Fetching feed...":
    feed, error = fetch_and_parse_feed(st.session_state.rss_url)
    if error:
        st.error(error)
        st.session_state.agent_status = "Error fetching/parsing feed"
    elif feed:
        st.session_state.agent_status = "Filtering articles..."
        new_articles, latest_id = filter_articles(
            feed.entries,
            st.session_state.keywords,
            st.session_state.last_checked_entry_id
        )
        if new_articles:
            st.success(f"Found {len(new_articles)} new articles matching criteria.")
            st.session_state.pending_analysis = new_articles
            st.session_state.agent_status = "Analyzing..."
        else:
            st.info("No new articles found matching criteria since last check.")
            st.session_state.agent_status = "Idle"
        if latest_id:
             st.session_state.last_checked_entry_id = latest_id
    else:
         st.session_state.agent_status = "Idle"
    st.rerun()

if st.session_state.agent_status == "Analyzing...":
    if st.session_state.pending_analysis:
        article_to_analyze = st.session_state.pending_analysis.pop(0)
        title_preview = article_to_analyze.get('title', '...')[:50]
        with st.spinner(f"Analyzing article: {title_preview}..."):
            analysis_results = analyze_article_multimodal(article_to_analyze)
            st.session_state.analyzed_articles.insert(0, {"entry": article_to_analyze, "analysis": analysis_results})
            st.session_state.analyzed_articles = st.session_state.analyzed_articles[:50]
        if st.session_state.pending_analysis:
            st.session_state.agent_status = "Analyzing..."
        else:
            st.session_state.agent_status = "Idle"
        st.rerun()
    else:
        st.session_state.agent_status = "Idle"
        st.rerun()

# --- Display Area ---
st.header("Agent Analysis Feed")

if st.session_state.analyzed_articles:
    for i, article_data in enumerate(st.session_state.analyzed_articles):
        entry = article_data["entry"]
        analysis = article_data["analysis"]
        sentiment = analysis.get('sentiment', 'N/A')
        expander_title = f"**{entry.get('title', 'No Title')}** (Sentiment: {sentiment})"

        with st.expander(expander_title):
            st.caption(f"Published: {entry.get('published', 'N/A')} | [Link]({entry.get('link', '#')})", unsafe_allow_html=True)
            st.markdown("**Summary:**")
            summary_text = analysis.get("summary", "Not available.")
            st.markdown(summary_text)

            if analysis.get("entities"): st.markdown(f"**Key Entities:** {', '.join(analysis.get('entities'))}")

            # Display Image and Analysis
            img_analysis = analysis.get("image_analysis")
            img_url = analysis.get("image_url")
            if img_url:
                st.image(img_url, width=300, caption="Detected Image")
            if img_analysis:
                 st.markdown("**Image Analysis:**")
                 st.markdown(img_analysis)
            elif img_url:
                 st.markdown("**Image Analysis:** (Not available or failed)")

            # Optional Speak Button
            if tts_engine:
                speak_content = f"Title: {entry.get('title', 'No Title')}. Sentiment: {sentiment}. Summary: {summary_text}"
                if st.button(f"🔊 Read Summary##{i}", key=f"speak_{i}"):
                    speak_text(speak_content)

            # Q&A Section
            st.markdown("--- ")
            st.markdown("**Ask a question about this article:**")
            qa_key = f"qa_{i}"
            question = st.text_input("Your question:", key=f"question_{qa_key}", label_visibility="collapsed")
            if st.button("Ask", key=f"ask_{qa_key}"):
                if question:
                    article_context = get_article_context(entry)
                    with st.spinner("Thinking..."):
                        answer = ask_gemini_about_article(question, article_context)
                        st.session_state.qa_answers[qa_key] = answer
                        st.rerun() # Rerun to display the answer
                else:
                    st.warning("Please enter a question.")

            # Display answer if exists
            if qa_key in st.session_state.qa_answers:
                st.info(f"**Answer:** {st.session_state.qa_answers[qa_key]}")

            st.markdown("--- ") # End of expander content
else:
    st.info("No articles analyzed yet. Configure the agent and run a check.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with Python, Streamlit, Gemini API, feedparser, pyttsx3, BeautifulSoup")

