-- Generic mart_conversations extraction.
--
-- Replaces the FEMSA-specific query. Workflow filter and has_oris filter are
-- now injectable — this query is reusable for any workflow and any eval family.
--
-- The output preserves every column the downstream LangSmith-shaped writer
-- + dbt MERGE needs (conversation_sk, flowbuilder_session_id, capability
-- flags). Downstream preprocessing groups by conversation_sk (or unnests
-- turns) as needed.
--
-- Pipeline-injected parameters:
--   {start_date}        — inclusive start (YYYY-MM-DD)
--   {end_date}          — exclusive end   (YYYY-MM-DD)
--   {workflow_filter}   — "AND t.workflow_name IN ('bot-a','bot-b')" or ""
--   {has_oris_filter}   — "AND t.has_oris = TRUE" or "" (built from the
--                          context's has_oris_default + optional
--                          --has-oris / --no-has-oris CLI override)

SELECT
  t.conversation_sk AS conversation_id,   -- pipeline's grouping key
  t.conversation_sk,                       -- preserved for DWH2_STAGE join
  t.flowbuilder_session_id,
  t.workflow_id,
  t.workflow_name,
  t.user_id,
  t.channels,
  t.has_oris,
  t.has_commerce_session,
  t.has_custom_agents,
  t.has_knowledge_genie,
  t.has_salesdesk_interaction,
  t.has_human_interaction,
  t.oris_skills_invoked,
  t.oris_tools_invoked,
  t.conversation_started_at,
  t.conversation_ended_at,
  t2.message_id,
  t2.step_name,
  t2.sender_type,
  t2.message_text,
  t2.event_timestamp

FROM `arched-photon-194421.DWH2.mart_conversations` AS t
LEFT JOIN UNNEST(t.messages) AS t2

WHERE
  t.conversation_started_at >= '{start_date}'
  AND t.conversation_started_at <  '{end_date}'
  {workflow_filter}
  {has_oris_filter}
  AND t2.message_text IS NOT NULL

ORDER BY t.conversation_sk, t2.event_timestamp
