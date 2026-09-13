import re

from googleapiclient.discovery import build


# ============================================================
# CHANNEL URL HELPERS
# ============================================================

def extract_channel_identifier(channel_url):
    """
    Extract a YouTube channel ID or handle from a channel URL.
    """

    if not channel_url:
        raise ValueError(
            "Please enter a YouTube channel URL."
        )

    url = channel_url.strip()

    # --------------------------------------------------------
    # Direct channel ID
    # Example:
    # https://www.youtube.com/channel/UCxxxxxxxx
    # --------------------------------------------------------

    channel_match = re.search(
        r"youtube\.com/channel/"
        r"(UC[\w-]+)",
        url,
        re.IGNORECASE
    )

    if channel_match:

        return {
            "type": "channel_id",
            "value": channel_match.group(1)
        }

    # --------------------------------------------------------
    # YouTube handle
    # Example:
    # https://www.youtube.com/@mkbhd
    # --------------------------------------------------------

    handle_match = re.search(
        r"youtube\.com/@([\w.-]+)",
        url,
        re.IGNORECASE
    )

    if handle_match:

        return {
            "type": "handle",
            "value": "@"
            + handle_match.group(1)
        }

    raise ValueError(
        "Unsupported YouTube channel URL. "
        "Use a URL such as "
        "https://www.youtube.com/@channel "
        "or https://www.youtube.com/channel/UC..."
    )


# ============================================================
# BUILD API CLIENT
# ============================================================

def build_youtube_client(api_key):

    if not api_key:
        raise ValueError(
            "YouTube API key is missing."
        )

    return build(
        "youtube",
        "v3",
        developerKey=api_key
    )


# ============================================================
# FIND CHANNEL
# ============================================================

def resolve_channel(
    youtube,
    identifier
):

    if identifier["type"] == "channel_id":

        response = (
            youtube.channels()
            .list(
                part="snippet,statistics,contentDetails",
                id=identifier["value"]
            )
            .execute()
        )

    elif identifier["type"] == "handle":

        response = (
            youtube.channels()
            .list(
                part="snippet,statistics,contentDetails",
                forHandle=identifier["value"]
            )
            .execute()
        )

    else:

        raise ValueError(
            "Unsupported channel identifier."
        )

    items = response.get(
        "items",
        []
    )

    if not items:

        raise ValueError(
            "YouTube channel could not be found."
        )

    return items[0]


# ============================================================
# GET CHANNEL INFORMATION
# ============================================================

def get_channel_info(
    channel_url,
    api_key
):

    identifier = (
        extract_channel_identifier(
            channel_url
        )
    )

    youtube = build_youtube_client(
        api_key
    )

    channel = resolve_channel(
        youtube,
        identifier
    )

    snippet = channel.get(
        "snippet",
        {}
    )

    statistics = channel.get(
        "statistics",
        {}
    )

    content_details = channel.get(
        "contentDetails",
        {}
    )

    # --------------------------------------------------------
    # Channel information
    # --------------------------------------------------------

    channel_id = channel.get(
        "id",
        ""
    )

    channel_title = snippet.get(
        "title",
        ""
    )

    channel_description = snippet.get(
        "description",
        ""
    )

    country = snippet.get(
        "country",
        ""
    )

    subscriber_count = int(
        statistics.get(
            "subscriberCount",
            0
        )
    )

    video_count = int(
        statistics.get(
            "videoCount",
            0
        )
    )

    view_count = int(
        statistics.get(
            "viewCount",
            0
        )
    )

    uploads_playlist_id = (
        content_details
        .get(
            "relatedPlaylists",
            {}
        )
        .get(
            "uploads",
            ""
        )
    )

    return {

        "channel_id":
            channel_id,

        "channel_title":
            channel_title,

        "channel_description":
            channel_description,

        "country":
            country,

        "subscriber_count":
            subscriber_count,

        "video_count":
            video_count,

        "view_count":
            view_count,

        "uploads_playlist_id":
            uploads_playlist_id
    }


# ============================================================
# CHANNEL SIZE
# ============================================================

def get_channel_size_group(
    subscriber_count
):

    if subscriber_count < 1000:
        return "0-1K"

    elif subscriber_count < 10000:
        return "1K-10K"

    elif subscriber_count < 100000:
        return "10K-100K"

    elif subscriber_count < 1000000:
        return "100K-1M"

    return "1M+"