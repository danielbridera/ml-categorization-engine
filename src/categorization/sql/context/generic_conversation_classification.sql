-- Generic Conversation Classification Query
--
-- Implements CIE-style conversation splitting directly in SQL using window functions:
--   - Groups messages by user_id
--   - Starts a new conversation when the gap between consecutive messages > timeout
--   - Assigns convo_id_composite = first_message_id + '_' + last_message_id (same as CIE)
--   - Samples exactly n_samples conversations
--
-- Parameters (provided by the pipeline):
--   {start_date}                   - Start date for conversation_started_at filter
--   {end_date}                     - End date for conversation_started_at filter
--   {n_samples}                    - Number of conversations to sample
--   {conversation_timeout_minutes} - Inactivity timeout in minutes (e.g. 30)
--   {workflow_name_filter}         - Optional: "AND workflow_name = 'my_bot'" or empty string

WITH all_messages AS (
  -- Extract all messages (user + assistant) within the date range
  SELECT
    t.workflow_id,
    t.workflow_name,
    t.user_id,
    t2.message_id,
    t2.step_name,
    t2.sender_type,
    t2.message_text,
    t2.event_timestamp,
    -- Time of the previous message by the same user (for gap calculation)
    LAG(t2.event_timestamp) OVER (
      PARTITION BY t.user_id
      ORDER BY t2.event_timestamp
    ) AS prev_timestamp

  FROM `arched-photon-194421.DWH2.mart_conversations` AS t
  LEFT JOIN UNNEST(t.messages) AS t2

  WHERE
    t.conversation_started_at >= '{start_date}'
    AND t.conversation_started_at < '{end_date}'
    {workflow_name_filter}
    AND t2.message_text IS NOT NULL
),

with_conv_boundaries AS (
  -- Flag the first message of each new conversation
  -- (gap > timeout OR first message from this user in the dataset)
  SELECT
    *,
    CASE
      WHEN prev_timestamp IS NULL THEN 1
      WHEN TIMESTAMP_DIFF(event_timestamp, prev_timestamp, MINUTE) > {conversation_timeout_minutes} THEN 1
      ELSE 0
    END AS is_new_conv
  FROM all_messages
),

with_conv_groups AS (
  -- Cumulative sum of new-conversation flags gives a monotonically increasing
  -- group number per user — same logic as CIE's infer_conversations_by_timeout()
  SELECT
    *,
    SUM(is_new_conv) OVER (
      PARTITION BY user_id
      ORDER BY event_timestamp
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS conv_group_num
  FROM with_conv_boundaries
),

with_conv_ids AS (
  -- Build convo_id_composite = first_message_id + '_' + last_message_id (same as CIE)
  SELECT
    *,
    CONCAT(
      FIRST_VALUE(message_id) OVER (
        PARTITION BY user_id, conv_group_num
        ORDER BY event_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
      ),
      '_',
      LAST_VALUE(message_id) OVER (
        PARTITION BY user_id, conv_group_num
        ORDER BY event_timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
      )
    ) AS conversation_id
  FROM with_conv_groups
),

sampled_conversations AS (
  -- Sample exactly n_samples distinct conversations
  SELECT DISTINCT conversation_id
  FROM with_conv_ids
  ORDER BY RAND()
  LIMIT {n_samples}
)

SELECT
  w.conversation_id,
  w.workflow_id,
  w.workflow_name,
  w.user_id,
  w.message_id,
  w.step_name,
  w.sender_type,
  w.message_text,
  w.event_timestamp

FROM with_conv_ids w
INNER JOIN sampled_conversations sc ON sc.conversation_id = w.conversation_id

ORDER BY w.conversation_id, w.event_timestamp
