import os
import sys
from datetime import datetime

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(
        0,
        SRC_PATH
    )


# ============================================================
# IMPORT CREATORLENS
# ============================================================

from predict import (
    predict_performance,
    interpret_prediction
)

from youtube_api import (
    get_channel_info,
    get_channel_size_group
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CreatorLens",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🎬 CreatorLens")

st.subheader(
    "YouTube Engagement Prediction"
)

st.write(
    "Estimate how your video may perform relative "
    "to comparable videos using your content, "
    "channel information and upload timing."
)


# ============================================================
# CHANNEL
# ============================================================

st.header("1. Your YouTube Channel")

channel_url = st.text_input(
    "YouTube Channel URL",
    placeholder=(
        "https://www.youtube.com/@yourchannel"
    )
)

load_channel = st.button(
    "🔎 Load Channel",
    use_container_width=True
)


# ------------------------------------------------------------
# Store channel information in session state
# ------------------------------------------------------------

if "channel_info" not in st.session_state:

    st.session_state.channel_info = None


# ------------------------------------------------------------
# Load channel
# ------------------------------------------------------------

if load_channel:

    if not channel_url.strip():

        st.error(
            "Please enter your YouTube channel URL."
        )

    elif "YOUTUBE_API_KEY" not in st.secrets:

        st.error(
            "YouTube API key is not configured. "
            "Add YOUTUBE_API_KEY to "
            ".streamlit/secrets.toml."
        )

    else:

        try:

            with st.spinner(
                "Loading YouTube channel..."
            ):

                channel_info = (
                    get_channel_info(

                        channel_url=(
                            channel_url
                        ),

                        api_key=(
                            st.secrets[
                                "YOUTUBE_API_KEY"
                            ]
                        )
                    )
                )

            st.session_state.channel_info = (
                channel_info
            )

            st.success(
                "Channel loaded successfully."
            )

        except Exception as error:

            st.session_state.channel_info = None

            st.error(
                f"Could not load channel: {error}"
            )


# ============================================================
# DISPLAY CHANNEL
# ============================================================

channel_info = (
    st.session_state.channel_info
)


if channel_info:

    st.subheader(
        channel_info["channel_title"]
    )

    channel_size_group = (
        get_channel_size_group(
            channel_info[
                "subscriber_count"
            ]
        )
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

        st.caption(
            f"Country: {channel_info['country']}"
        )

    with st.expander(
        "Channel description"
    ):

        st.write(
            channel_info[
                "channel_description"
            ]
            or "No channel description available."
        )


# ============================================================
# VIDEO INFORMATION
# ============================================================

st.header("2. Video Information")

title = st.text_input(
    "Video Title",
    placeholder=(
        "Example: 10 Python Projects That "
        "Will Improve Your Coding Skills"
    )
)

description = st.text_area(
    "Description",
    placeholder=(
        "Enter your YouTube video description..."
    ),
    height=180
)

tags = st.text_input(
    "Tags",
    placeholder=(
        "python, programming, coding, learn python"
    )
)

col1, col2 = st.columns(2)

with col1:

    category_id = st.number_input(
        "YouTube Category ID",
        min_value=0,
        max_value=100,
        value=28,
        step=1
    )

with col2:

    topic_group = st.text_input(
        "Topic",
        placeholder=(
            "Example: Education | python tutorial"
        )
    )


# ============================================================
# VIDEO DETAILS
# ============================================================

st.header("3. Video Details")

col1, col2 = st.columns(2)

with col1:

    duration_minutes = st.number_input(
        "Video Duration (minutes)",
        min_value=0.0,
        max_value=1440.0,
        value=10.0,
        step=0.5
    )

with col2:

    video_format = st.selectbox(
        "Video Format",
        [
            "Long-form",
            "Short",
            "Live"
        ]
    )


# ============================================================
# UPLOAD TIMING
# ============================================================

st.header("4. Planned Upload Time")

upload_datetime = st.datetime_input(
    "Planned upload date and time",
    value=datetime.now()
)

upload_year = (
    upload_datetime.year
)

upload_month = (
    upload_datetime.month
)

upload_day = (
    upload_datetime.day
)

upload_hour = (
    upload_datetime.hour
)

upload_weekday = (
    upload_datetime.weekday()
)

is_weekend = int(
    upload_weekday >= 5
)

is_missing_publish_time = 0


# ============================================================
# PREDICT
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Predict Performance",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Validate channel
    # --------------------------------------------------------

    if channel_info is None:

        st.error(
            "Please load your YouTube channel first."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate title
    # --------------------------------------------------------

    if not title.strip():

        st.error(
            "Please enter a video title."
        )

        st.stop()


    # --------------------------------------------------------
    # Channel information
    # --------------------------------------------------------

    subscriber_count = (
        channel_info[
            "subscriber_count"
        ]
    )

    video_count = (
        channel_info[
            "video_count"
        ]
    )

    channel_description = (
        channel_info[
            "channel_description"
        ]
    )

    country = (
        channel_info[
            "country"
        ]
    )

    channel_size_group = (
        get_channel_size_group(
            subscriber_count
        )
    )


    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration_seconds = (
        duration_minutes * 60
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Analyzing your video..."
        ):

            relative_performance = (
                predict_performance(

                    title=title,

                    description=description,

                    tags=tags,

                    channel_description=(
                        channel_description
                    ),

                    category_id=int(
                        category_id
                    ),

                    subscriber_count=int(
                        subscriber_count
                    ),

                    video_count=int(
                        video_count
                    ),

                    country=country,

                    channel_size_group=(
                        channel_size_group
                    ),

                    topic_group=topic_group,

                    duration_seconds=(
                        duration_seconds
                    ),

                    upload_year=(
                        upload_year
                    ),

                    upload_month=(
                        upload_month
                    ),

                    upload_day=(
                        upload_day
                    ),

                    upload_hour=(
                        upload_hour
                    ),

                    upload_weekday=(
                        upload_weekday
                    ),

                    is_weekend=is_weekend,

                    is_missing_publish_time=(
                        is_missing_publish_time
                    )
                )
            )


        # ----------------------------------------------------
        # Interpret
        # ----------------------------------------------------

        interpretation = (
            interpret_prediction(
                relative_performance
            )
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.header("📊 Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Performance",
                interpretation[
                    "category"
                ]
            )

        with col2:

            st.metric(
                "Performance Score",
                f"{interpretation['score']}/100"
            )


        # ----------------------------------------------------
        # Result explanation
        # ----------------------------------------------------

        if (
            interpretation["category"]
            == "Above usual"
        ):

            st.success(
                "The model predicts this video "
                "may perform above the usual level "
                "represented by comparable training data."
            )

        elif (
            interpretation["category"]
            == "Below usual"
        ):

            st.warning(
                "The model predicts this video "
                "may perform below the usual level "
                "represented by comparable training data."
            )

        else:

            st.info(
                "The model predicts this video "
                "may perform around the usual level "
                "represented by comparable training data."
            )


        st.caption(
            "This is a model prediction, not a guarantee "
            "of future views or engagement."
        )


        # ----------------------------------------------------
        # Technical information
        # ----------------------------------------------------

        with st.expander(
            "Technical prediction details"
        ):

            st.write(
                f"Relative performance: "
                f"{relative_performance:.4f}"
            )

            st.write(
                f"Channel size: "
                f"{channel_size_group}"
            )

            st.write(
                f"Video format entered: "
                f"{video_format}"
            )

            st.write(
                "The performance score is a presentation "
                "score and is not a probability."
            )


    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )   