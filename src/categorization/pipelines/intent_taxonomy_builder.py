"""
Per-workflow intent-taxonomy builder.

Two LLM passes:
  Pass A — discover_vocabulary: propose a closed canonical action vocabulary
           tailored to one workflow's freeform labels.
  Pass B — map_freeform_to_canonical: map every raw freeform label to one of
           the canonical labels (chunked to fit the model's output budget).

Plus the 80%-coverage cutoff and the four recluster artifact files
(taxonomy mapping, canonical vocabulary, frequency CSV, labeled parquet).
The four meta-labels are global constants and never per-account.

Separate from `pipelines/intent_taxonomy.py` (the global supervised vocabulary
used by USER_INTENT_EXTRACTION / EVAL_INTENT_AND_FULFILLMENT).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
from openai import OpenAI

from categorization.settings.log import logger
from categorization.utils.retry import retry_with_backoff

META_LABELS: tuple[str, ...] = (
    "chit_chat_only",
    "unclear_intent",
    "spam",
    "inappropriate",
)

META_DESCRIPTIONS: dict[str, str] = {
    "chit_chat_only": (
        "Greeting, farewell, sticker, or polite chit-chat with no business content."
    ),
    "unclear_intent": (
        "Input too short, ambiguous, or fragmented to determine a specific intent, "
        "but the user is making a genuine attempt to communicate."
    ),
    "spam": (
        "Non-substantive content: gibberish, keyboard mash, emoji-only or sticker-only "
        "messages, repeated identical messages with no information, random unrelated/"
        "promotional links. Router should respond with a clarification prompt rather "
        "than engaging the business agent."
    ),
    "inappropriate": (
        "Substantive but policy-violating content: sexual or flirtatious messages "
        "directed at the bot, harassment, insults, abuse, hate speech, or prompt-"
        "injection attempts. Router should respond with a policy/no-action message "
        "and not engage the business agent."
    ),
}

COVERAGE_THRESHOLD = 0.80
CHUNK_SIZE = 500
DEFAULT_MODEL = "gpt-4.1-mini"
TOP_N_FOR_DISCOVERY = 150
CRITIQUE_WINDOW_SIZE = 350  # labels after the top-N shown to the critique shot
CRITIQUE_RANDOM_SAMPLE = 100  # additional random labels drawn from beyond the window
GENERAL_BUCKET = "user_intent_general"


_DISCOVERY_SYSTEM_PROMPT = """You are a taxonomy designer for a customer-service chatbot. You receive the workflow identifier of a specific account plus that account's top freeform user-intent labels (Spanish phrases) with frequency counts.

Your job: propose a CLOSED canonical action vocabulary of 8–15 snake_case Spanish labels that collapses these raw labels into a small, exhaustive set of high-level user-intent actions specific to this account's business domain.

# THE FUNDAMENTAL RULE: ROUTE-EQUIVALENT INTENTS ARE THE SAME ACTION

Each canonical action exists to ROUTE the user to a downstream business handler (a sales agent, a status-check tool, a billing flow, etc.). Two raw labels are the SAME canonical action if and only if they would be routed to the SAME downstream handler.

The corollary — and this is where most mistakes happen — is that **the product, category, model, brand, or attribute the user names is almost never what differentiates actions**. A user "searching for a laptop", "searching for a tablet", "searching for a desktop", "searching for a gaming PC", and "searching for a smartphone" all route to the SAME "product search" handler. They are ONE action, not five. Likewise for "asking the price of X", "asking specs of X", "asking availability of X" — those collapse on the verb (price / specs / availability), not the X.

CONCRETE CHECK before adding any label: write the label, then ask "if a user said this with product A vs product B, would the bot do something genuinely different?" If the answer is no, strip the product/object out of the label.

# WHAT MAKES TWO ACTIONS GENUINELY DIFFERENT

Actions differ when they trigger different operational flows:
- `activar_tarjeta` ≠ `cancelar_tarjeta` ≠ `solicitar_tarjeta` — three different banking flows on the same object (card).
- `buscar_producto` ≠ `consultar_precio` ≠ `consultar_disponibilidad` — three different retail flows on the same object (product).
- `consultar_pedido_existente` ≠ `solicitar_garantia_soporte` — both touch a placed order but route to different teams.

Actions do NOT differ when the only difference is the object/product:
- `buscar_laptop` and `buscar_tablet` are the SAME action → `buscar_producto`.
- `consultar_precio_laptop` and `consultar_precio_tablet` are the SAME action → `consultar_precio` (or `consultar_precio_financiamiento` if the account bundles pricing+financing).
- `asesoria_laptop_gamer` and `asesoria_tablet_estudio` are the SAME action → `pedir_asesoria`.

# OUTPUT FIELDS

For each canonical action:
- `name`: snake_case Spanish phrase. Start with a verb (`buscar_`, `consultar_`, `solicitar_`, `cancelar_`, `activar_`, `reclamar_`, `confirmar_`, `actualizar_`, `recuperar_`, `contactar_`, etc.). NO English. NO brand or model names. NO product types or category words (no `_laptop`, `_tablet`, `_computador`, `_tarjeta_credito`, `_portatil`, `_estudio`, `_gamer`, etc.) UNLESS the object name is genuinely what differentiates two different operational flows on the SAME verb (e.g. banking actions on `tarjeta` vs `cuenta` vs `prestamo` — in retail this almost never applies).
- `description`: one sentence in English describing when this action applies, followed by 2–4 representative Spanish phrases users actually say (drawn from the input where possible). The description IS allowed to mention representative products — the constraint is on the `name`.

Also propose:
- `account_slug`: a single short lowercase identifier suitable for a filename, derived from the workflow name (e.g. workflow `wa-al2318-alkosto` → `alkosto`, `wa-op2221-openbank-main` → `openbank`).

# OTHER GUIDELINES

- Aim for 8–15 actions. If you produce more than 15, you are almost certainly slicing by product — collapse back. The hand-curated retail and banking templates we've validated land at 10 and 15 actions respectively.
- Cover the bulk of the input frequency, not just unique strings. High-frequency raw labels MUST collapse cleanly into a canonical action.
- DO NOT propose any of: chit_chat_only, unclear_intent, spam, inappropriate. These four meta-labels are appended automatically.
- DO NOT include `user_intent_general` — it is a downstream bucket, not a discovered action.
- Stay account-specific in wording where the verbs warrant it (a bank uses `activar`/`cancelar`/`reclamar` on cards; a retailer uses `buscar`/`comprar`/`consultar_precio` on products) — but never let account-specificity collapse INTO the product axis.

# OUTPUT FORMAT — respond with a JSON object only:
{
  "account_slug": "...",
  "actions": [
    {"name": "...", "description": "..."},
    ...
  ]
}
No prose, no code fences."""


_CRITIQUE_SYSTEM_PROMPT = """You are refining a canonical action vocabulary for a customer-service chatbot.

You receive:
1. An INITIAL proposed vocabulary (account_slug + actions) — produced from the top freeform labels of this account.
2. A BROADER SAMPLE of freeform user-intent labels (Spanish) with counts. These are labels that were NOT visible when the initial vocabulary was proposed: medium-frequency labels just outside the top-N plus a random sample drawn from the long tail. The same business intent may appear under many low-count phrasings — what matters is the CLUSTER, not any single string.

Your job: REVIEW the initial vocabulary against this broader sample and identify intent CLUSTERS in the sample that don't map cleanly to any existing canonical action. For each such cluster, ADD a new canonical action.

# THE FUNDAMENTAL RULE (unchanged)

Each canonical action exists to ROUTE the user to a downstream business handler. Two raw labels are the SAME canonical action if and only if they would be routed to the SAME downstream handler. **The product, category, model, brand, or attribute the user names is almost never what differentiates actions.** Strip the object out of the canonical name unless different objects would trigger different downstream flows on the same verb.

# WHAT COUNTS AS A "MISSING ACTION"

Look for verbs / business operations in the sample that the initial vocabulary cannot absorb cleanly:
- A SET of labels about asking for technical features, specs, dimensions, compatibility, what-is-included — if the initial vocab has no `consultar_especificaciones`-style action, that's a gap.
- A SET of labels about explicitly completing a purchase or saying "voy a comprar", "lo llevo" — if there's no `comprar_producto`-style action, that's a gap.
- A SET of labels about a distinct post-sale operation (warranty, returns, defective product) not covered by the initial actions.
- Any other coherent intent cluster representing more than a handful of conversations.

A single one-off raw label is NOT a missing action — only ADD when you see a cluster of semantically-similar labels with material aggregate frequency.

# CONSTRAINTS

- PRESERVE the initial vocabulary: do NOT rename, remove, or rewrite existing actions or descriptions. Only APPEND new actions to the list.
- The target is still 8–15 total actions. The initial vocab is usually 8–12; with critique additions you should typically land at 10–15. Going above 15 means you're slicing by product or being too granular — collapse instead.
- Apply the route-equivalence rule strictly to any addition. Pure-verb names; no product/category suffixes unless they identify a genuinely different operational flow.
- The 4 meta-labels (chit_chat_only, unclear_intent, spam, inappropriate) are appended automatically — do NOT propose them.
- If the initial vocabulary already covers everything in the sample, return it UNCHANGED (same account_slug, same actions list).

# OUTPUT FORMAT — respond with a JSON object only:
{
  "account_slug": "<same as initial>",
  "actions": [
    <all initial actions, verbatim>,
    <any new actions to add>
  ],
  "added_count": <integer — how many new actions were appended>
}
No prose, no code fences."""


def _build_mapping_system_prompt(workflow_name: str, vocab: dict[str, Any]) -> str:
    """Compose the Pass B system prompt from the discovered vocab + global meta-labels."""
    action_lines = "\n".join(
        f"- `{action['name']}` — {action['description']}"
        for action in vocab["actions"]
    )
    meta_lines = "\n".join(
        f"- `{label}` — pass through unchanged. {META_DESCRIPTIONS[label]}"
        for label in META_LABELS
    )
    return f"""You are a taxonomy mapper for the workflow `{workflow_name}`. You receive freeform user-intent labels (Spanish) with frequency counts. Collapse each raw label to ONE of these canonical labels ONLY.

# CANONICAL LABELS (pick exactly one per raw label)

{action_lines}
{meta_lines}

# RULES

1. Pick ONE label from the list above for every raw label. No new labels.
2. Meta-labels MUST pass through unchanged: `chit_chat_only` → `chit_chat_only`, `unclear_intent` → `unclear_intent`, `spam` → `spam`, `inappropriate` → `inappropriate`.
3. Every raw label in the input MUST appear as a key in your output. Do not skip rare labels.

# OUTPUT FORMAT

Respond ONLY with a JSON object: {{"<raw_label>": "<canonical_label>", ...}}. No prose, no code fences."""


def _propose_initial_vocabulary(
    client: OpenAI,
    workflow_name: str,
    freeform_counts: pd.Series,
    top_n: int,
    model: str,
) -> dict[str, Any]:
    """Shot 1 — propose an initial vocabulary from the top-N most frequent freeform labels."""
    top = freeform_counts.head(top_n)
    label_lines = "\n".join(f"{label}\t{count}" for label, count in top.items())
    user_msg = (
        f"Account workflow: {workflow_name}\n"
        f"Top {len(top)} freeform user-intent labels with counts (tab-separated):\n\n"
        f"{label_lines}\n\n"
        f"Propose the canonical action vocabulary. Return JSON."
    )

    def _call() -> Any:
        return client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

    resp = retry_with_backoff(_call)
    raw = json.loads(resp.choices[0].message.content)

    if "account_slug" not in raw or "actions" not in raw:
        raise ValueError(
            f"Discovery response missing required keys (got: {list(raw)})"
        )

    actions: list[dict[str, str]] = [
        {"name": a["name"], "description": a["description"]} for a in raw["actions"]
    ]
    actions = [a for a in actions if a["name"] not in META_LABELS]

    return {
        "account_slug": raw["account_slug"],
        "actions": actions,
    }


def _critique_vocabulary(
    client: OpenAI,
    workflow_name: str,
    initial_vocab: dict[str, Any],
    freeform_counts: pd.Series,
    top_n: int,
    critique_window: int,
    critique_random_sample: int,
    model: str,
    rng_seed: int = 42,
) -> tuple[dict[str, Any], int]:
    """Shot 2 — show the LLM the initial vocab plus a broader sample of unseen labels and ask for additions.

    Returns (refined_vocab, added_count). If no additional labels exist beyond
    top_n, the initial vocab is returned unchanged.
    """
    after_top = freeform_counts.iloc[top_n:]
    if after_top.empty:
        return initial_vocab, 0

    window = after_top.head(critique_window)
    remaining = after_top.iloc[critique_window:]
    if critique_random_sample > 0 and len(remaining) > 0:
        sample_size = min(critique_random_sample, len(remaining))
        random_sample = remaining.sample(
            n=sample_size, random_state=rng_seed
        ).sort_values(ascending=False)
    else:
        random_sample = remaining.iloc[:0]

    initial_payload = json.dumps(
        {
            "account_slug": initial_vocab["account_slug"],
            "actions": initial_vocab["actions"],
        },
        ensure_ascii=False,
        indent=2,
    )
    window_lines = "\n".join(f"{label}\t{count}" for label, count in window.items())
    sample_lines = "\n".join(
        f"{label}\t{count}" for label, count in random_sample.items()
    )

    user_msg = (
        f"Account workflow: {workflow_name}\n\n"
        f"# INITIAL VOCABULARY\n{initial_payload}\n\n"
        f"# BROADER SAMPLE — labels NOT seen during initial proposal\n\n"
        f"## Medium-frequency labels (next {len(window)} after the top-{top_n}):\n"
        f"{window_lines}\n\n"
        f"## Random sample from the long tail ({len(random_sample)} labels):\n"
        f"{sample_lines}\n\n"
        f"Review and return the (possibly expanded) vocabulary as JSON."
    )

    def _call() -> Any:
        return client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

    resp = retry_with_backoff(_call)
    raw = json.loads(resp.choices[0].message.content)

    if "actions" not in raw:
        raise ValueError(
            f"Critique response missing `actions` key (got: {list(raw)})"
        )

    refined_actions: list[dict[str, str]] = [
        {"name": a["name"], "description": a["description"]} for a in raw["actions"]
    ]
    refined_actions = [a for a in refined_actions if a["name"] not in META_LABELS]

    initial_names = {a["name"] for a in initial_vocab["actions"]}
    refined_names = {a["name"] for a in refined_actions}
    # Defensive: if the LLM dropped any initial action, reinstate it (constraint says preserve).
    if not initial_names.issubset(refined_names):
        missing = [
            a for a in initial_vocab["actions"] if a["name"] not in refined_names
        ]
        refined_actions = refined_actions + missing
        refined_names = refined_names | {a["name"] for a in missing}

    added_count = len(refined_names - initial_names)
    refined_vocab = {
        "account_slug": raw.get("account_slug") or initial_vocab["account_slug"],
        "actions": refined_actions,
    }
    return refined_vocab, added_count


def discover_vocabulary(
    client: OpenAI,
    workflow_name: str,
    freeform_counts: pd.Series,
    top_n: int = TOP_N_FOR_DISCOVERY,
    model: str = DEFAULT_MODEL,
    critique: bool = True,
    critique_window: int = CRITIQUE_WINDOW_SIZE,
    critique_random_sample: int = CRITIQUE_RANDOM_SAMPLE,
) -> dict[str, Any]:
    """Pass A — propose a closed canonical action vocabulary from the freeform labels.

    Runs in two shots by default:
      Shot 1 — propose initial vocab from the top-N most frequent labels.
      Shot 2 — show the proposal + a broader sample (next `critique_window`
               labels plus a random sample of `critique_random_sample` from
               the rest of the tail) and ask for additions to cover clusters
               the initial pass missed.

    Set `critique=False` to skip Shot 2 (faster, cheaper, lower recall).
    """
    initial_vocab = _propose_initial_vocabulary(
        client, workflow_name, freeform_counts, top_n, model
    )
    logger.info(
        f"[{workflow_name}] shot 1 proposed {len(initial_vocab['actions'])} actions "
        f"(slug: {initial_vocab['account_slug']})"
    )

    final_vocab = initial_vocab
    if critique:
        final_vocab, added = _critique_vocabulary(
            client,
            workflow_name,
            initial_vocab,
            freeform_counts,
            top_n=top_n,
            critique_window=critique_window,
            critique_random_sample=critique_random_sample,
            model=model,
        )
        logger.info(
            f"[{workflow_name}] shot 2 added {added} action(s) "
            f"(total: {len(final_vocab['actions'])})"
        )

    return {
        "account_slug": final_vocab["account_slug"],
        "workflow_name": workflow_name,
        "actions": final_vocab["actions"],
        "meta_labels": list(META_LABELS),
    }


def _map_chunk(
    client: OpenAI,
    workflow_name: str,
    vocab: dict[str, Any],
    chunk: pd.Series,
    model: str,
) -> dict[str, str]:
    label_lines = "\n".join(f"{label}\t{count}" for label, count in chunk.items())
    user_msg = (
        f"Account workflow: {workflow_name}\n"
        f"Raw freeform intent labels with counts (tab-separated). "
        f"Map EVERY label to one of the canonical labels:\n\n"
        f"{label_lines}\n\n"
        f"Return JSON mapping."
    )
    system_prompt = _build_mapping_system_prompt(workflow_name, vocab)

    def _call() -> Any:
        return client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )

    resp = retry_with_backoff(_call)
    return cast(dict[str, str], json.loads(resp.choices[0].message.content))


def map_freeform_to_canonical(
    client: OpenAI,
    workflow_name: str,
    vocab: dict[str, Any],
    freeform_counts: pd.Series,
    chunk_size: int = CHUNK_SIZE,
    model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Pass B — map every raw freeform label to one canonical label, chunked at `chunk_size`.

    Identity-fallback for any raw label the LLM omits; meta-labels pass through identity.
    """
    mapping: dict[str, str] = {}
    n_chunks = (len(freeform_counts) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        chunk = freeform_counts.iloc[i * chunk_size : (i + 1) * chunk_size]
        logger.info(
            f"  chunk {i + 1}/{n_chunks}: mapping {len(chunk)} labels…"
        )
        mapping.update(_map_chunk(client, workflow_name, vocab, chunk, model))

    canonical_names: set[str] = {a["name"] for a in vocab["actions"]} | set(META_LABELS)
    # Defensive fallbacks: identity-map anything the LLM dropped or hallucinated off-vocab.
    for raw in freeform_counts.index:
        if raw not in mapping or mapping[raw] not in canonical_names:
            mapping[raw] = raw
    # Meta-labels MUST pass through unchanged.
    for meta in META_LABELS:
        if meta in mapping:
            mapping[meta] = meta
    return mapping


def apply_coverage_cutoff(
    df: pd.DataFrame,
    canonical_col: str = "primary_intent_canonical",
    final_col: str = "primary_intent_final",
    workflow_col: str = "workflow_name",
    threshold: float = COVERAGE_THRESHOLD,
    general_bucket: str = GENERAL_BUCKET,
) -> pd.DataFrame:
    """Add `final_col` from `canonical_col`, collapsing tail labels beyond `threshold` cumulative coverage.

    Operates per workflow. Meta-labels are always in the head set.
    Returns the same DataFrame (mutated in place) for fluent chaining.
    """
    df[final_col] = df[canonical_col]
    for workflow, group in df.groupby(workflow_col):
        counts = group[canonical_col].value_counts()
        total = counts.sum()
        if total == 0:
            continue
        cum_pct = counts.cumsum() / total
        head_idx = cum_pct[cum_pct <= threshold].index.tolist()
        if not head_idx or cum_pct.iloc[len(head_idx) - 1] < threshold:
            head_idx = cum_pct.index[: len(head_idx) + 1].tolist()
        head_set: set[str] = set(head_idx) | set(META_LABELS)

        mask = df[workflow_col] == workflow
        tail_mask = mask & ~df[canonical_col].isin(head_set)
        df.loc[tail_mask, final_col] = general_bucket
    return df


def write_recluster_artifacts(
    experiment_dir: Path,
    df: pd.DataFrame,
    mapping_per_workflow: dict[str, dict[str, str]],
    vocab_per_workflow: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Write the four recluster artifacts to `experiment_dir`.

    Returns the paths keyed by short name.
    """
    paths: dict[str, Path] = {}

    mapping_path = experiment_dir / "taxonomy_mapping_action.json"
    mapping_path.write_text(
        json.dumps(mapping_per_workflow, ensure_ascii=False, indent=2)
    )
    paths["taxonomy_mapping_action"] = mapping_path

    for workflow, vocab in vocab_per_workflow.items():
        vocab_path = experiment_dir / f"canonical_vocabulary_{workflow}.json"
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
        paths[f"canonical_vocabulary_{workflow}"] = vocab_path

    freq_rows: list[dict[str, Any]] = []
    for workflow, group in df.groupby("workflow_name"):
        total = len(group)
        final_counts = group["primary_intent_final"].value_counts()
        for label, count in final_counts.items():
            freq_rows.append(
                {
                    "workflow_name": workflow,
                    "primary_intent_final": label,
                    "count": int(count),
                    "pct": round(count / total, 4),
                }
            )
    freq_df = pd.DataFrame(freq_rows).sort_values(
        ["workflow_name", "count"], ascending=[True, False]
    )
    freq_path = experiment_dir / "intent_frequencies_action.csv"
    freq_df.to_csv(freq_path, index=False)
    paths["intent_frequencies_action"] = freq_path

    parquet_path = experiment_dir / "labeled_with_final_action.parquet"
    df.to_parquet(parquet_path, index=False)
    paths["labeled_with_final_action"] = parquet_path

    return paths


def _classifier_method_blurb(n_raw_labels: int, chunk_size: int = CHUNK_SIZE) -> str:
    """Method blurb embedded in the per-account JSON's `classifier.method` field."""
    if n_raw_labels <= chunk_size:
        return (
            "1) freeform LLM intent extraction on first-5-user-messages context; "
            "2) LLM auto-discovery of a closed action vocabulary; "
            "3) single-pass LLM mapping to the discovered vocabulary; "
            "4) frequency aggregation."
        )
    n_chunks = (n_raw_labels + chunk_size - 1) // chunk_size
    return (
        "1) freeform LLM intent extraction on first-5-user-messages context; "
        "2) LLM auto-discovery of a closed action vocabulary; "
        f"3) chunked LLM mapping to the discovered vocabulary ({n_chunks} chunks of "
        f"up to {chunk_size} labels each); "
        "4) frequency aggregation."
    )


def build_taxonomy_json(
    experiment_dir: Path,
    workflow_name: str,
    vocab: dict[str, Any],
    df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Build and write `intent_taxonomy_<account_slug>.json` for one workflow.

    Reads `metadata.json` for experiment_id, created_at, date_range.
    Uses `primary_intent_canonical` counts so every prescribed label appears
    even when it falls below the coverage cutoff.
    """
    metadata = json.loads((experiment_dir / "metadata.json").read_text())

    wf_df = df[df["workflow_name"] == workflow_name]
    total = len(wf_df)
    counts = wf_df["primary_intent_canonical"].value_counts()

    prescribed: list[tuple[str, str, str]] = [
        (a["name"], "action", a["description"]) for a in vocab["actions"]
    ]
    prescribed += [
        (label, "meta", META_DESCRIPTIONS[label]) for label in META_LABELS
    ]
    prescribed_names = {name for name, _, _ in prescribed}

    rows = [
        (name, kind, desc, int(counts.get(name, 0)))
        for name, kind, desc in prescribed
    ]
    rows.sort(key=lambda r: r[3], reverse=True)

    unmapped_count = int(counts[~counts.index.isin(prescribed_names)].sum())

    intents: list[dict[str, Any]] = []
    cumulative = 0
    for name, kind, desc, cnt in rows:
        cumulative += cnt
        intents.append(
            {
                "intent": name,
                "type": kind,
                "description": desc,
                "count": cnt,
                "percentage": round(cnt / total * 100, 1) if total and cnt else 0.0,
                "cumulative_count": cumulative,
                "cumulative_percentage": (
                    round(cumulative / total * 100, 1) if total else 0.0
                ),
            }
        )

    params = metadata.get("parameters", {})
    output: dict[str, Any] = {
        "agent": params.get("workflow_names") or workflow_name,
        "experiment_id": metadata["experiment_id"],
        "created_at": metadata["created_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "start": params.get("start_date"),
            "end": params.get("end_date"),
        },
        "conversations_used": total,
        "classifier": {
            "freeform_intent_extractor": "USER_INTENT_FREEFORM",
            "action_only_taxonomy_collapse_model": model,
            "method": _classifier_method_blurb(int(counts.size)),
        },
        "user_intents": intents,
        "unmapped_long_tail_count": unmapped_count,
    }

    slug = vocab["account_slug"]
    out_path = experiment_dir / f"intent_taxonomy_{slug}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return out_path
