"""
Conversation Grouping and Turn Formatting Utilities

Ports the CIE timeout-based conversation inference and turn building logic
for use in the categorization pipeline. Conversation definition mirrors the
ml-conversational-insights-engine project:
- Boundary: gap > timeout between consecutive messages by same user_id
- Conversation ID: convo_id_composite = {first_message_id}_{last_message_id}
- Default timeout: 30 minutes
"""

import re
from datetime import timedelta
from typing import Any

import pandas as pd

from categorization.settings.log import logger

DEFAULT_CONVERSATION_TIMEOUT = "30m"


def parse_timeout(timeout_str: str) -> timedelta:
    """
    Parse timeout strings like '30m', '24h', '2d' into datetime.timedelta.

    Args:
        timeout_str: A timeout string in the form "<number><unit>" where
                    unit in {'d' (days), 'h' (hours), 'm' (minutes)}.

    Returns:
        timedelta: The parsed duration

    Raises:
        ValueError: If the input does not match the expected pattern

    Examples:
        >>> parse_timeout("30m")
        datetime.timedelta(seconds=1800)
        >>> parse_timeout("24h")
        datetime.timedelta(days=1)
        >>> parse_timeout("2d")
        datetime.timedelta(days=2)
    """
    m = re.fullmatch(r"\s*(\d+)\s*([dhm])\s*", timeout_str, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid timeout: {timeout_str!r}. Use '<n><d|h|m>'.")
    qty, unit = int(m.group(1)), m.group(2).lower()
    return (
        timedelta(days=qty)
        if unit == "d"
        else timedelta(hours=qty)
        if unit == "h"
        else timedelta(minutes=qty)
    )


def infer_conversations_by_timeout(
    df: pd.DataFrame,
    timeout_str: str = DEFAULT_CONVERSATION_TIMEOUT,
    user_col: str = "user_id",
    time_col: str = "event_timestamp",
    message_id_col: str = "message_id",
    session_col: str = "convo_id_composite",
) -> pd.DataFrame:
    """
    Add a conversation-group column by splitting each user's messages when
    the inactivity gap exceeds a timeout.

    Args:
        df: Input message log. Must contain user_col, time_col, message_id_col.
        timeout_str: Timeout string (e.g. '30m', '12h', '2d').
        user_col: Column identifying the user.
        time_col: Column with message timestamps (must be datetime-like).
        message_id_col: Column with unique message identifiers.
        session_col: Name of the output column with conversation-group IDs.

    Returns:
        Copy of df sorted by user_col and time_col with added columns:
        session_col (convo_id_composite), first_message_id, last_message_id,
        conversation_start_time, conversation_end_time.
    """
    if df.empty:
        return df.copy()

    timeout = parse_timeout(timeout_str)
    out = df.sort_values([user_col, time_col]).copy()

    out["__delta"] = out.groupby(user_col)[time_col].diff().fillna(pd.Timedelta(0))
    out["__new_conv"] = (out["__delta"] > timeout).astype(int)
    out["__grp_num"] = out.groupby(user_col)["__new_conv"].cumsum()

    id_results: list[dict[str, Any]] = []
    for (_user, _grp_num), group in out.groupby([user_col, "__grp_num"]):
        sorted_group = group.sort_values(time_col)

        if (
            message_id_col in sorted_group.columns
            and not sorted_group[message_id_col].isna().all()
        ):
            first_msg_id = str(sorted_group[message_id_col].iloc[0])
            last_msg_id = str(sorted_group[message_id_col].iloc[-1])
            composite_id = f"{first_msg_id}_{last_msg_id}"
        else:
            user_id = sorted_group[user_col].iloc[0]
            grp_num = sorted_group["__grp_num"].iloc[0]
            composite_id = f"{user_id}_{grp_num}"
            first_msg_id = composite_id
            last_msg_id = composite_id

        first_msg_time = sorted_group[time_col].iloc[0]
        last_msg_time = sorted_group[time_col].iloc[-1]

        for idx in group.index:
            id_results.append(
                {
                    "index": idx,
                    "first_message_id": first_msg_id,
                    "last_message_id": last_msg_id,
                    session_col: composite_id,
                    "conversation_start_time": first_msg_time,
                    "conversation_end_time": last_msg_time,
                }
            )

    id_df = pd.DataFrame(id_results).set_index("index")
    out = out.merge(id_df, left_index=True, right_index=True, how="left")
    return out.drop(columns=["__delta", "__new_conv", "__grp_num"])


def format_conversation_text(
    df: pd.DataFrame,
    session_col: str = "convo_id_composite",
    sender_col: str = "sender_type",
    text_col: str = "message_text",
    time_col: str = "event_timestamp",
) -> pd.DataFrame:
    """
    Convert a message-level DataFrame into one row per conversation.

    Consecutive same-sender messages are merged into a single turn.
    Turn format: "User: <text>\\nAssistant: <text>\\nUser: ..."

    Args:
        df: Message-level DataFrame with conversation IDs already assigned.
            Must contain session_col, sender_col, text_col, time_col.
        session_col: Column grouping rows into conversations.
        sender_col: Column with sender label. "user" → User turn; anything
                   else → Assistant turn.
        text_col: Column with message text.
        time_col: Column with message timestamps.

    Returns:
        DataFrame with one row per conversation and columns:
        convo_id_composite, workflow_id, workflow_name, user_id,
        first_message_id, last_message_id, conversation_start_time,
        conversation_end_time, message_count, conversation_text.
    """
    static_cols = [
        "workflow_id",
        "workflow_name",
        "user_id",
        "first_message_id",
        "last_message_id",
        "conversation_start_time",
        "conversation_end_time",
    ]

    records: list[dict[str, Any]] = []
    df_sorted = df.sort_values([session_col, time_col]).copy()

    for session_id, grp in df_sorted.groupby(session_col, sort=False):
        rec: dict[str, Any] = {session_col: session_id}

        for col in static_cols:
            rec[col] = (
                grp[col].iloc[0] if col in grp.columns and not grp[col].empty else None
            )

        rec["message_count"] = len(grp)

        # Build ping-pong turn text by merging consecutive same-sender messages
        turns: list[str] = []
        current_role: str | None = None
        buf_msgs: list[str] = []

        for _, row in grp.iterrows():
            raw_sender = row[sender_col] if sender_col in grp.columns else "user"
            role = "User" if str(raw_sender).lower() == "user" else "Assistant"
            text = str(row[text_col] if text_col in grp.columns else "").strip()

            if role == current_role:
                buf_msgs.append(text)
            else:
                if buf_msgs and current_role:
                    turns.append(f"{current_role}: {' '.join(buf_msgs)}")
                current_role = role
                buf_msgs = [text]

        if buf_msgs and current_role:
            turns.append(f"{current_role}: {' '.join(buf_msgs)}")

        rec["conversation_text"] = "\n".join(turns)
        records.append(rec)

    return pd.DataFrame(records)


def format_conversations_from_sessions(
    df: pd.DataFrame,
    session_col: str = "conversation_id",
    sender_col: str = "sender_type",
    text_col: str = "message_text",
    time_col: str = "event_timestamp",
) -> pd.DataFrame:
    """
    Convert a message-level DataFrame into one row per conversation when
    conversations are already defined by a session column (e.g. from SQL).

    This is used when conversation splitting is done at the SQL level
    (via window functions), so no timeout inference is needed here.

    Args:
        df: Message-level DataFrame with conversation_id already assigned.
            Must contain session_col, sender_col, text_col, time_col.
        session_col: Column that identifies each conversation (e.g. conversation_id).
        sender_col: Column with sender label. "user" → User turn; else → Assistant.
        text_col: Column with message text.
        time_col: Column with message timestamps.

    Returns:
        DataFrame with one row per conversation and columns:
        conversation_id, workflow_id, workflow_name, user_id,
        conversation_start_time, conversation_end_time,
        message_count, conversation_text.
    """
    static_cols = ["workflow_id", "workflow_name", "user_id"]

    records: list[dict[str, Any]] = []
    df_sorted = df.sort_values([session_col, time_col]).copy()

    for session_id, grp in df_sorted.groupby(session_col, sort=False):
        rec: dict[str, Any] = {session_col: session_id}

        for col in static_cols:
            rec[col] = (
                grp[col].iloc[0] if col in grp.columns and not grp[col].empty else None
            )

        rec["conversation_start_time"] = grp[time_col].min()
        rec["conversation_end_time"] = grp[time_col].max()
        rec["message_count"] = len(grp)

        # Build ping-pong turn text by merging consecutive same-sender messages
        turns: list[str] = []
        current_role: str | None = None
        buf_msgs: list[str] = []

        for _, row in grp.iterrows():
            raw_sender = row[sender_col] if sender_col in grp.columns else "user"
            role = "User" if str(raw_sender).lower() == "user" else "Assistant"
            text = str(row[text_col] if text_col in grp.columns else "").strip()

            if role == current_role:
                buf_msgs.append(text)
            else:
                if buf_msgs and current_role:
                    turns.append(f"{current_role}: {' '.join(buf_msgs)}")
                current_role = role
                buf_msgs = [text]

        if buf_msgs and current_role:
            turns.append(f"{current_role}: {' '.join(buf_msgs)}")

        rec["conversation_text"] = "\n".join(turns)
        records.append(rec)

    return pd.DataFrame(records)


def group_messages_into_conversations(
    df: pd.DataFrame,
    timeout_str: str = DEFAULT_CONVERSATION_TIMEOUT,
) -> pd.DataFrame:
    """
    Full pipeline: raw messages DataFrame -> one row per conversation.

    Args:
        df: Raw messages from BigQuery. Must contain: user_id, event_timestamp,
            message_id, sender_type, message_text, workflow_id, workflow_name.
        timeout_str: Inactivity timeout (e.g. "30m", "1h").

    Returns:
        DataFrame with one row per conversation with columns:
        convo_id_composite, workflow_id, workflow_name, user_id,
        first_message_id, last_message_id, conversation_start_time,
        conversation_end_time, message_count, conversation_text.
    """
    required = [
        "user_id",
        "event_timestamp",
        "message_id",
        "sender_type",
        "message_text",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for conversation grouping: {missing}"
        )

    logger.info(
        f"Grouping {len(df):,} messages into conversations (timeout={timeout_str})"
    )

    df_with_convos = infer_conversations_by_timeout(df, timeout_str=timeout_str)
    n_convos = df_with_convos["convo_id_composite"].nunique()
    logger.info(f"Identified {n_convos:,} conversations from {len(df):,} messages")

    conversations_df = format_conversation_text(df_with_convos)
    logger.success(f"Built {len(conversations_df):,} conversation records")

    return conversations_df
