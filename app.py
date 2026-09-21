import os
import sys
from datetime import date, datetime

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# ============================================================
# IMPORT CREATORLENS
# ============================================================

from predict import predict_performance, interpret_prediction
from youtube_api import get_channel_info, get_channel_size_group


# Optional AI feature. The main ML prediction does not depend on it.
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreatorLens",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "channel_info" not in st.session_state:
    st.session_state.channel_info = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = None


# ============================================================
# YOUTUBE CATEGORIES
# ============================================================

YOUTUBE_CATEGORIES = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
    30: "Movies",
    31: "Anime/Animation",
    32: "Action/Adventure",
    33: "Classics",
    34: "Comedy",
    35: "Documentary",
    36: "Drama",
    37: "Family",
    38: "Foreign",
    39: "Horror",
    40: "Sci-Fi/Fantasy",
    41: "Thriller",
    42: "Shorts",
    43: "Shows",
    44: "Trailers",
}

CATEGORY_OPTIONS = [
    f"{category_id} — {category_name}"
    for category_id, category_name in YOUTUBE_CATEGORIES.items()
]


# ============================================================
# AI HELPER
# ============================================================

def generate_ai_suggestions(video_data):
    """Generate optional content suggestions using the OpenRouter REST API."""

    if "interpretation" not in video_data:
        raise RuntimeError(
            "Prediction details are missing. Please run the performance analysis again."
        )

    if not REQUESTS_AVAILABLE:
        raise RuntimeError(
            "The requests package is not installed. Add 'requests' to requirements.txt."
        )

    if "OPENROUTER_API_KEY" not in st.secrets:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured in Streamlit Secrets."
        )

    prompt = f"""
You are a practical YouTube content editor helping a creator improve a video before publishing.

Give useful suggestions based only on the information provided below.
Do not invent facts about the video.
Do not promise more views, higher CTR, or better performance.
Do not say that an AI suggestion will increase views by a specific percentage.
Keep the language simple and practical.

VIDEO INFORMATION
Title: {video_data['title']}
Description: {video_data['description'] or 'Not provided'}
Tags: {video_data['tags'] or 'Not provided'}
Category: {video_data['category_id']} — {video_data['category_name']}
Topic: {video_data['topic']}
Format: {video_data['video_format']}
Duration: {video_data['duration_minutes']} minutes
Planned upload: {video_data['upload_datetime']}

CREATORLENS ML RESULT
Performance: {video_data['interpretation']['category']}
Performance score: {video_data['interpretation']['score']}/100
Relative performance: {video_data['relative_performance']:.4f}

Return exactly these sections:

### Title suggestions
Give 3 different title options. Keep them natural and suitable for the topic.

### Description improvement
Give a concise improved description that keeps the original meaning and does not invent information.

### Tag suggestions
Give 8 to 12 relevant tags, comma-separated.

### Why these changes
Give 3 short bullets explaining the changes in simple language.
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://creatorlens.streamlit.app",
            "X-Title": "CreatorLens",
        },
        json={
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.7,
        },
        timeout=90,
    )

    if not response.ok:
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get(
                "message",
                f"HTTP {response.status_code}"
            )
        except Exception:
            error_message = f"HTTP {response.status_code}: {response.text[:300]}"

        raise RuntimeError(f"OpenRouter request failed: {error_message}")

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            "OpenRouter returned an unexpected response format."
        )

    if not content or not str(content).strip():
        raise RuntimeError("OpenRouter returned an empty AI response.")

    return content


# ============================================================
# HEADER
# ============================================================

st.title("🎬 CreatorLens")

st.subheader("YouTube Engagement Prediction")

st.write(
    "Estimate how your video may perform compared with similar videos "
    "using your content, channel information and planned upload time."
)


# ============================================================
# 1. CHANNEL
# ============================================================

st.header("1. Your YouTube Channel")

channel_url = st.text_input(
    "YouTube Channel URL",
    placeholder="https://www.youtube.com/@yourchannel",
    help="Paste a public YouTube channel URL."
)

load_channel = st.button(
    "🔎 Load Channel",
    use_container_width=True
)

if load_channel:

    if not channel_url.strip():
        st.error("Please enter your YouTube channel URL.")

    elif "YOUTUBE_API_KEY" not in st.secrets:
        st.error(
            "YouTube API key is not configured. Add YOUTUBE_API_KEY "
            "to Streamlit Secrets."
        )

    else:
        try:
            with st.spinner("Loading YouTube channel..."):
                channel_info = get_channel_info(
                    channel_url=channel_url,
                    api_key=st.secrets["YOUTUBE_API_KEY"]
                )

            st.session_state.channel_info = channel_info
            st.session_state.prediction_result = None
            st.session_state.ai_suggestions = None
            st.success("Channel loaded successfully.")

        except Exception as error:
            st.session_state.channel_info = None
            st.error(f"Could not load channel: {error}")


channel_info = st.session_state.channel_info

if channel_info:

    st.subheader(channel_info["channel_title"])

    channel_size_group = get_channel_size_group(
        channel_info["subscriber_count"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Subscribers",
            f"{channel_info['subscriber_count']:,}"
        )

    with col2:
        st.metric(
            "Videos",
            f"{channel_info['video_count']:,}"
        )

    with col3:
        st.metric(
            "Channel Size",
            channel_size_group
        )

    if channel_info["country"]:
        st.caption(f"Country: {channel_info['country']}")

    with st.expander("Channel description"):
        st.write(
            channel_info["channel_description"]
            or "No channel description available."
        )


# ============================================================
# 2. VIDEO CONTENT
# ============================================================

st.header("2. Video Content")

st.caption(
    "Enter the content you plan to publish. These details are used by the ML model."
)

title = st.text_input(
    "Video Title",
    placeholder="Example: 10 Python Projects Every Beginner Should Build"
)

description = st.text_area(
    "Description",
    placeholder="Write the description you plan to use on YouTube...",
    height=160,
    help="Paste your planned YouTube description."
)

tags = st.text_input(
    "Tags",
    placeholder="python, programming, coding, learn python",
    help="Add relevant keywords separated by commas."
)

col1, col2 = st.columns([1, 1.5])

with col1:
    selected_category = st.selectbox(
        "YouTube Category",
        CATEGORY_OPTIONS,
        index=CATEGORY_OPTIONS.index("28 — Science & Technology"),
        help=(
            "Choose the category that best matches your video. "
            "The number is the YouTube category ID."
        )
    )

    category_id = int(selected_category.split(" — ", 1)[0])
    category_name = YOUTUBE_CATEGORIES[category_id]

with col2:
    topic_input = st.text_input(
        "Topic / Subtopic",
        placeholder="Example: Python for beginners",
        help=(
            "Enter the main subject of the video, not the full title. "
            "Examples: Python for beginners, AI tools, football training, "
            "budget travel, smartphone review."
        )
    )

    st.caption(
        "Tip: keep it short and specific. Example: Python for beginners."
    )

topic_group = (
    f"{category_name} | {topic_input.strip()}"
    if topic_input.strip()
    else f"{category_name} |"
)


# ============================================================
# 3. VIDEO DETAILS
# ============================================================

st.header("3. Video Details")

col1, col2 = st.columns(2)

with col1:
    duration_minutes = st.number_input(
        "Video Duration (minutes)",
        min_value=0.0,
        max_value=1440.0,
        value=10.0,
        step=0.5,
        help="Enter the expected length of the video."
    )

with col2:
    video_format = st.selectbox(
        "Video Format",
        ["Long-form", "Short", "Live"],
        help="Choose the type of video you plan to publish."
    )


# ============================================================
# 4. PLANNED PUBLISHING
# ============================================================

st.header("4. Planned Publishing")

st.caption(
    "Choose the date and approximate time you plan to publish the video."
)

time_col1, time_col2 = st.columns(2)

with time_col1:
    upload_date = st.date_input(
        "📅 Upload date",
        value=date.today(),
        min_value=date.today(),
        help="Select your planned publishing date."
    )

with time_col2:
    upload_time = st.time_input(
        "🕐 Upload time",
        value=datetime.now().replace(
            second=0,
            microsecond=0
        ).time(),
        help="Select your planned publishing time."
    )

upload_datetime = datetime.combine(upload_date, upload_time)

st.info(
    f"**Planned upload:** {upload_datetime.strftime('%A, %d %B %Y at %I:%M %p')}"
)

upload_year = upload_datetime.year
upload_month = upload_datetime.month
upload_day = upload_datetime.day
upload_hour = upload_datetime.hour
upload_weekday = upload_datetime.weekday()
is_weekend = int(upload_weekday >= 5)
is_missing_publish_time = 0


# ============================================================
# VIDEO SUMMARY
# ============================================================

if channel_info:
    with st.expander("🎬 Review your video details before prediction"):
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown(f"**Title**  \n{title or 'Not entered'}")
            st.markdown(f"**Category**  \n{category_id} — {category_name}")
            st.markdown(f"**Topic**  \n{topic_input or 'Not entered'}")
            st.markdown(f"**Format**  \n{video_format}")

        with summary_col2:
            st.markdown(f"**Duration**  \n{duration_minutes:g} minutes")
            st.markdown(
                f"**Planned upload**  \n{upload_datetime.strftime('%d %b %Y · %I:%M %p')}"
            )
            st.markdown(f"**Tags**  \n{tags or 'Not entered'}")


# ============================================================
# PREDICT
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Analyze Video Performance",
    type="primary",
    use_container_width=True
)

if predict_button:

    if channel_info is None:
        st.error("Please load your YouTube channel first.")
        st.stop()

    if not title.strip():
        st.error("Please enter a video title.")
        st.stop()

    if not topic_input.strip():
        st.error("Please enter the main topic or subtopic of the video.")
        st.stop()

    subscriber_count = channel_info["subscriber_count"]
    video_count = channel_info["video_count"]
    channel_description = channel_info["channel_description"]
    country = channel_info["country"]
    channel_size_group = get_channel_size_group(subscriber_count)
    duration_seconds = duration_minutes * 60

    try:
        with st.spinner("Analyzing your video..."):
            relative_performance = predict_performance(
                title=title,
                description=description,
                tags=tags,
                channel_description=channel_description,
                category_id=int(category_id),
                subscriber_count=int(subscriber_count),
                video_count=int(video_count),
                country=country,
                channel_size_group=channel_size_group,
                topic_group=topic_group,
                duration_seconds=duration_seconds,
                upload_year=upload_year,
                upload_month=upload_month,
                upload_day=upload_day,
                upload_hour=upload_hour,
                upload_weekday=upload_weekday,
                is_weekend=is_weekend,
                is_missing_publish_time=is_missing_publish_time
            )

        interpretation = interpret_prediction(relative_performance)

        st.session_state.prediction_result = {
            "relative_performance": relative_performance,
            "interpretation": interpretation,
            "title": title,
            "description": description,
            "tags": tags,
            "category_id": category_id,
            "category_name": category_name,
            "topic": topic_input.strip(),
            "video_format": video_format,
            "duration_minutes": duration_minutes,
            "upload_datetime": upload_datetime.strftime(
                "%A, %d %B %Y at %I:%M %p"
            ),
            "subscriber_count": subscriber_count,
            "video_count": video_count,
            "channel_size_group": channel_size_group,
        }

        st.session_state.ai_suggestions = None

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.prediction_result

if result:

    relative_performance = result["relative_performance"]
    interpretation = result["interpretation"]
    channel_size_group = result["channel_size_group"]
    subscriber_count = result["subscriber_count"]
    video_count = result["video_count"]
    video_format = result["video_format"]
    upload_datetime_display = result["upload_datetime"]

    st.header("📊 Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Performance",
            interpretation["category"]
        )

    with col2:
        st.metric(
            "Performance Score",
            f"{interpretation['score']}/100"
        )

    st.progress(
        max(0.0, min(1.0, interpretation["score"] / 100))
    )

    st.caption(
        "This 0–100 score is only a simple way to display the model result. "
        "It is not a probability."
    )

    if interpretation["category"] == "Above usual":
        st.success(
            "The model expects this video to perform above the usual level "
            "seen in comparable training data."
        )
    elif interpretation["category"] == "Below usual":
        st.warning(
            "The model expects this video to perform below the usual level "
            "seen in comparable training data."
        )
    else:
        st.info(
            "The model expects this video to perform around the usual level "
            "seen in comparable training data."
        )

    # --------------------------------------------------------
    # What this means
    # --------------------------------------------------------

    st.subheader("📈 What this means")

    if relative_performance > 0:
        comparison_text = (
            "The prediction is above the model's usual baseline."
        )
    elif relative_performance < 0:
        comparison_text = (
            "The prediction is below the model's usual baseline."
        )
    else:
        comparison_text = (
            "The prediction is at the model's usual baseline."
        )

    st.write(
        f"CreatorLens expects this video to perform **{interpretation['category'].lower()}** "
        "compared with similar videos represented in the training data."
    )

    st.write(comparison_text)

    st.caption(
        f"**Relative performance: {relative_performance:+.2f}** — "
        "a model value showing how the prediction compares with its usual baseline. "
        "It does **not** mean a percentage increase or decrease in views."
    )

    # --------------------------------------------------------
    # Prediction factors
    # --------------------------------------------------------

    st.subheader("🔎 What the model used")

    factor_col1, factor_col2, factor_col3 = st.columns(3)

    with factor_col1:
        st.markdown("**👤 Channel**")
        st.write(f"{channel_size_group} subscriber range")

        st.markdown("**📝 Content**")
        st.write("Title, description and tags")

    with factor_col2:
        st.markdown("**📚 Topic**")
        st.write(
            f"{result['category_name']} · {result['topic']}"
        )

        st.markdown("**🎬 Format**")
        st.write(video_format)

    with factor_col3:
        st.markdown("**⏱ Upload time**")
        st.write(upload_datetime_display)

        st.markdown("**📺 Channel data**")
        st.write(
            f"{subscriber_count:,} subscribers · {video_count:,} videos"
        )

    # --------------------------------------------------------
    # AI suggestions
    # --------------------------------------------------------

    st.subheader("✨ AI content suggestions")

    st.write(
        "Get practical suggestions for your title, description and tags. "
        "These suggestions are separate from the ML prediction."
    )

    if "OPENROUTER_API_KEY" not in st.secrets:
        st.info(
            "AI suggestions are optional. Add OPENROUTER_API_KEY to Streamlit "
            "Secrets to enable them. The ML prediction works without it."
        )
    else:
        ai_button = st.button(
            "✨ Generate AI Suggestions",
            use_container_width=True
        )

        if ai_button:
            try:
                with st.spinner("Creating content suggestions..."):
                    st.session_state.ai_suggestions = generate_ai_suggestions(
                        result
                    )
            except Exception as error:
                st.session_state.ai_suggestions = None
                st.error(f"AI suggestions failed: {error}")

        if st.session_state.ai_suggestions:
            st.markdown(st.session_state.ai_suggestions)
            st.caption(
                "AI suggestions are generated from the information you entered. "
                "Review them before publishing and do not treat them as a guarantee of performance."
            )

    # --------------------------------------------------------
    # Model context
    # --------------------------------------------------------

    with st.expander("🤖 How CreatorLens works"):

        st.markdown(
            "**Prediction type**  \n"
            "Relative performance — it compares the expected level of this video "
            "with the usual level in the model's training data."
        )

        st.markdown(
            "**Model**  \n"
            "Text + Structured Ridge regression."
        )

        st.markdown(
            "**The model looks at**  \n"
            "• Title, description and tags  \n"
            "• Channel size and public channel information  \n"
            "• Topic and category  \n"
            "• Video duration  \n"
            "• Planned upload date and time"
        )

        st.markdown(
            "**Simple meaning of relative performance**  \n"
            "It is a model number that says whether the prediction is above, "
            "near, or below the usual baseline. It is not a view count and not "
            "a percentage change in views."
        )

        st.markdown(
            "**Important**  \n"
            "The model cannot see everything that affects YouTube performance, "
            "such as future audience behavior, competition, impressions, click-through "
            "rate, watch time and recommendation traffic."
        )

    # --------------------------------------------------------
    # Technical details
    # --------------------------------------------------------

    with st.expander("Technical prediction details"):

        st.write(
            f"Relative performance: {relative_performance:.4f}"
        )

        st.write(
            f"Channel size: {channel_size_group}"
        )

        st.write(
            f"Video format: {video_format}"
        )

        st.write(
            "The performance score is a presentation score and is not a probability."
        )

    st.caption(
        "This is a model prediction, not a guarantee of future views or engagement."
    )
