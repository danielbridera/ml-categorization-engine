-- Generic Conversation Context Query
--
-- Queries pre-grouped conversations from dev-data-mlops.data_poc.v_conversations.
-- Samples up to {n_samples} rows per workflow (stratified by FARM_FINGERPRINT).
-- Agnostic of classifier — reused by all classifiers that target this data source.
--
-- Parameters injected by the pipeline:
--   {start_date}       — inclusive start date (YYYY-MM-DD)
--   {end_date}         — inclusive end date (YYYY-MM-DD)
--   {n_samples}        — max rows per workflow
--   {workflow_filter}  — optional WHERE fragment:
--                        "AND workflow_name IN ('bot-a', 'bot-b')"  → one or more workflows
--                        ""  → no filter, returns all workflows
-- Note: min_length and workflow_names_list are passed but unused here.

WITH ranked AS (
    SELECT
        conversation_id,
        workflow_id,
        workflow_name,
        conversation_start,
        message_text,
        ROW_NUMBER() OVER (
            PARTITION BY workflow_name
            ORDER BY FARM_FINGERPRINT(conversation_id)
        ) AS rn
    FROM `dev-data-mlops.data_poc.v_conversations`
    WHERE DATE(conversation_start) BETWEEN '{start_date}' AND '{end_date}'
      {workflow_filter}
      AND message_text IS NOT NULL
      AND LENGTH(TRIM(message_text)) > 0
)
SELECT
    conversation_id AS message_id,
    message_text,
    workflow_id,
    workflow_name,
    conversation_start
FROM ranked
WHERE rn <= {n_samples}
