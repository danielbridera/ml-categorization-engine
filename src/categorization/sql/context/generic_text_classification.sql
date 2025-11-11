-- Generic Text Classification Query
--
-- This query extracts user messages from the mart_conversations table for any
-- text classification task. It's context-agnostic and used by all classifiers
-- (SENTIMENT, ESCALATION, FEEDBACK, etc.) unless a custom query is specified.
--
-- Parameters (provided by the pipeline):
--   {start_date}  - Start date for conversation_started_at filter
--   {end_date}    - End date for conversation_started_at filter
--   {n_samples}   - Maximum number of messages to return
--   {min_length}  - Minimum message length to include

WITH user_messages AS (
  SELECT
    -- Conversation-level metadata
    t.workflow_id,
    t.workflow_name,
    t.user_id,

    -- Message-level data (from UNNESTED messages array)
    t2.message_id,
    t2.step_name,
    t2.message_text,
    t2.event_timestamp,

    -- Deduplicate within workflow
    ROW_NUMBER() OVER (
      PARTITION BY t.workflow_id, LOWER(TRIM(t2.message_text))
      ORDER BY t2.event_timestamp DESC
    ) as dedup_rank

  FROM
    `arched-photon-194421.DWH2.mart_conversations` AS t
  LEFT JOIN
    UNNEST(t.messages) AS t2

  WHERE
    -- Date range filter
    t.conversation_started_at >= '{start_date}'
    AND t.conversation_started_at < '{end_date}'

    -- Only user messages
    AND t2.sender_type = 'user'

    -- Only text message type
    AND t2.message_type = 'text'

    -- Only messages with text
    AND t2.message_text IS NOT NULL

    -- Filter out very short messages (likely not meaningful)
    AND LENGTH(TRIM(t2.message_text)) > {min_length}
),

deduplicated_messages AS (
  SELECT *
  FROM user_messages
  WHERE dedup_rank = 1  -- Keep only first occurrence of each unique message per workflow
)

-- Sample messages ensuring diversity across workflows
SELECT
  workflow_id,
  workflow_name,
  message_id,
  message_text,
  event_timestamp,
  step_name
FROM deduplicated_messages

-- Stratified sampling: random within each workflow
ORDER BY workflow_id, RAND()

-- Limit to requested sample size
LIMIT {n_samples}
