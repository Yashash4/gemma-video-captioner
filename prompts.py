"""All prompt text for the Track-2 captioning pipeline, in one place.

gemma-4-31b-it leaks chain-of-thought on ~1/3 of calls (and on the T1 spike returned
markdown reasoning with NO JSON at all for Stage A). So every output we must parse is
wrapped in a STRICT delimiter (<json> / <caption>) with a one-shot format example; the
pipeline in pipeline.py regenerates when the delimiter is absent. Text only here — the
parsing/regeneration logic lives in pipeline.py.

Style templates are verbatim from docs/51-TRACK2-PIPELINE.md §4 (incl. their example
voices); the grounding + self-check + accuracy prompts are from §5.
"""

# ----------------------------------------------------------------------------
# Stage A — grounding (ONE vision call). §5.1 schema + uncertainty flags.
# ----------------------------------------------------------------------------
GROUNDING = """You see FRAMES sampled from ONE video. Describe ONLY what is actually
visible across the frames. If you are not sure, put it in "uncertain" — never guess.
Do NOT infer brand names, exact locations, or personal identities unless a clear,
legible sign or logo proves it. Small or blurry on-screen text is unreliable (the same
sign can read differently across frames), so put any such text in "uncertain", NOT in
"on_screen_text".

Return exactly ONE JSON object with these keys, wrapped in <json></json>:
<json>{
  "subjects": [],        // people/animals/objects clearly present
  "actions": [],         // what they are doing, visible motion
  "setting": "",         // place / environment / time of day if visible
  "on_screen_text": [],  // ONLY clearly legible text/logos, else []
  "audio_summary": "no clear speech",
  "mood": "",            // overall vibe
  "uncertain": []        // things hinted at but not confirmable (blurry text, guessed brands/places/identities)
}</json>

Example of the exact shape to return (for an UNRELATED video — copy the format, not the content, and do NOT show any reasoning):
<json>{"subjects":["a cyclist"],"actions":["riding downhill"],"setting":"a mountain road at midday","on_screen_text":[],"audio_summary":"no clear speech","mood":"energetic","uncertain":["the exact location"]}</json>

Now do the same for the frames you see. Output ONLY the <json>...</json> block, no prose, no markdown, no reasoning before or after it."""

GROUNDING_RETRY = ("\n\nYour previous answer was NOT a single <json>...</json> object. "
                   "Do not explain. Output ONLY <json>{...}</json> now, with the keys above.")

# ----------------------------------------------------------------------------
# Stage A — self-check (ONE text call on the JSON, §5.2). Tightens, never adds.
# ----------------------------------------------------------------------------
SELF_CHECK = """Below is a JSON scene description generated from video frames. Make it
strictly grounded and properly hedged, WITHOUT adding anything new:
- Move any specific brand name, company, exact place/city, personal identity, or precise
  number that is NOT backed by clearly legible on-screen text into "uncertain".
- Remove anything internally contradictory or clearly speculative.
- Keep the generic, plainly-visible facts (subjects, actions, setting, mood) unchanged.
Keep the same keys. Output ONLY the cleaned object wrapped in <json></json>, nothing else.

<json><<JSON>></json>"""

# ----------------------------------------------------------------------------
# Stage B — styling. Shared preamble (§4, verbatim) + strict <caption> delimiter.
# ----------------------------------------------------------------------------
PREAMBLE = """You write ONE English caption for a video, given a verified JSON description of what
the video contains. HARD RULES:
- Use ONLY facts present in the JSON. Do not invent people, places, objects, numbers,
  brands, or events. If the JSON marks something "uncertain", do not assert it.
- Every factual claim in your caption must stay literally TRUE. The STYLE lives in
  tone, word choice, and framing — never in fabricated events. A joke that changes
  what happened is a wrong caption.
- 1-2 full sentences, ~12-40 words. No hashtags, no emojis, no preamble, no quotes.

OUTPUT: reply with ONLY the finished caption wrapped in <caption></caption> — a real,
complete sentence. No reasoning, no drafts, no notes, no placeholders, no backticks.
Copy the FORMAT of these examples (their content is UNRELATED to your video):

JSON: {"subjects":["a red kite"],"actions":["flying"],"setting":"a windy beach","uncertain":[]}
<caption>A red kite climbs and rolls high over a windy beach, tugging hard against its line.</caption>

JSON: {"subjects":["three chefs"],"actions":["plating dishes"],"setting":"a busy kitchen","uncertain":["the restaurant name"]}
<caption>Three chefs move in tight coordination, plating dishes across a busy kitchen at full speed.</caption>"""

CAPTION_RETRY = ("\n\nYour previous reply was NOT a valid caption (it was empty, a fragment, "
                 "a placeholder, or it contained reasoning/backticks). Reply with ONLY one real, "
                 "complete sentence describing the video, wrapped in <caption></caption>, and nothing else.")

# §4 style blocks — verbatim, each with its one-line example voice.
_FORMAL = """STYLE = FORMAL. Neutral, precise, documentary register. Third person. No contractions,
no slang, no jokes, no opinion. Describe what occurs plainly and objectively, as a
museum label or news caption would.
Example voice: "A brown bear wades through a shallow river, pausing to search the current for fish." """

_SARCASTIC = """STYLE = SARCASTIC. Dry, ironic, deadpan, faux-unimpressed. Understate or mock-overhype
what happens. The facts stay 100% true — the sarcasm is in the attitude, NOT in claiming
the opposite of what occurred. Do not become mean about real people.
Example voice: "A bear stands in a river hunting fish, because apparently that's the
riveting content we're all here for." """

_HUMOROUS_TECH = """STYLE = HUMOROUS (TECH). Playful and witty by COMPARING the real scene to software/
engineering concepts. CRITICAL: the tech framing must be an obvious SIMILE/comparison,
marked with words like "like", "as if", "the way a...", "with the ... of a ...", "basically" —
NEVER stated as literal fact. First say what ACTUALLY happens (true to the JSON), THEN layer
the tech comparison on top. Do NOT claim the subject literally has code, packets, servers,
uptime, latency, algorithms, or deploys — only that it is *like* them.
Draw comparisons from: APIs, buffering, caching, load-balancing, retries, bandwidth, backups.
Example voice: "A bear fishes in the river with the patience of a process waiting on a slow
API — it finally lands a single trout, no retries needed." """

_HUMOROUS_NON_TECH = """STYLE = HUMOROUS (EVERYDAY). Playful, warm, punny, relatable — like a funny friend
narrating. ABSOLUTELY NO tech/software/engineering jargon (no APIs, servers, code,
latency, bugs). Draw jokes from everyday life, food, relationships, weather. Keep facts
literally true.
Example voice: "This bear is treating the river like an all-you-can-eat sushi bar, and
honestly, respect the commitment." """

_STYLE_BLOCKS = {
    "formal": _FORMAL,
    "sarcastic": _SARCASTIC,
    "humorous_tech": _HUMOROUS_TECH,
    "humorous_non_tech": _HUMOROUS_NON_TECH,
}

# Generic block for any style string NOT in the known 4 (styles[] is read dynamically;
# the hidden set could contain another style). Same grounding rules, style from the name.
_GENERIC = """STYLE = {NAME}. Write the caption in a "{name}" style — its characteristic tone, word
choice, and framing, consistent with what "{name}" implies. Keep every factual claim
literally true and grounded in the JSON; the style is only in delivery, never in
fabricated events."""

# Distinctiveness reranker suffix (appended when regenerating a too-similar caption).
DISTINCT_SUFFIX = ("\n\nIMPORTANT: your caption MUST be clearly different from this other "
                   "caption for the same video — do not reuse its vocabulary, metaphors, or "
                   "sentence structure. Use a different vocabulary domain.\nOther caption: «<<OTHER>>»")


def style_prompt(style_name, facts_json, distinct_from=None):
    """Full Stage-B prompt: preamble + style block (known or generic) + the frozen facts."""
    block = _STYLE_BLOCKS.get(style_name)
    if block is None:
        block = _GENERIC.replace("{NAME}", style_name.upper()).replace("{name}", style_name.replace("_", " "))
    p = PREAMBLE + "\n\n" + block + "\n\nJSON: " + facts_json
    if distinct_from:
        p += DISTINCT_SUFFIX.replace("<<OTHER>>", distinct_from)
    return p


def self_check(facts_json):
    return SELF_CHECK.replace("<<JSON>>", facts_json)


# ----------------------------------------------------------------------------
# D2 — self-eval accuracy pass. ONE batched call over ALL captions (§6 / §5.2).
# ----------------------------------------------------------------------------
ACCURACY_EVAL = """You are checking captions for factual grounding against VERIFIED facts about a
video. A caption FAILS if it asserts any person, object, action, place, brand, number, or
event NOT supported by the facts (anything listed under "uncertain" counts as NOT supported).
Tone, humor, and sarcasm are fine as long as every stated fact stays true.

FACTS: <<JSON>>

CAPTIONS:
<<CAPS>>

List ONLY the labels of captions that fail. Output ONE JSON object wrapped in <json></json>:
<json>{"failed": ["label", ...]}</json>
If every caption is grounded, output <json>{"failed": []}</json>. No other text."""


def accuracy_eval(facts_json, labeled_caps):
    return ACCURACY_EVAL.replace("<<JSON>>", facts_json).replace("<<CAPS>>", labeled_caps)
