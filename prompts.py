"""All prompt text for the Track-2 captioning pipeline, in one place.

gemma-4-31b-it leaks chain-of-thought on ~1/3 of calls (and on the T1 spike returned
markdown reasoning with NO JSON at all for Stage A). So every output we must parse is
wrapped in a STRICT delimiter (<json> / <caption>) with a one-shot format example; the
pipeline in pipeline.py regenerates when the delimiter is absent. Text only here — the
parsing/regeneration logic lives in pipeline.py.

Style templates are tuned to the organizers' released reference captions (short, confident,
witty, specific; bold tech humor for humorous_tech, jargon-free everyday humor for
humorous_non_tech); the grounding + self-check prompts are from docs/51 §5.
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
the video contains. RULES:
- Ground the caption in the REAL scene: the subjects, actions, and setting in the JSON must
  stay recognizable. Do NOT invent real people, places, objects, numbers, brands, or events
  that are not in the JSON. If the JSON marks something "uncertain", do not assert it as fact.
- A clearly-figurative joke, metaphor, or comparison is EXPECTED humor, not a fabrication —
  framing the real scene through a witty lens is good. What is wrong is claiming a literal
  thing happened that didn't (a person, object, or action that isn't there).
- Be CONCISE, CONFIDENT, WITTY and SPECIFIC — anchor to a concrete detail of THIS video,
  never a generic template. Prefer ONE punchy sentence; sarcastic/humor styles may use up to
  2-3 short sentences. ~8-40 words. No hashtags, no emojis, no preamble, no quotes.

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

# Style blocks — tuned to the organizers' reference voice: short, confident, witty, specific.
_FORMAL = """STYLE = FORMAL. Clear, precise, professional — a documentary or news caption. Third
person, neutral, objective. ONE well-formed sentence naming the concrete subject, action,
and setting. No contractions, no slang, no jokes, no opinion, no metaphor.
Example voice: "A young orange tabby kitten sits among dense green foliage in an outdoor
setting, looking directly at the camera with an alert and curious expression." """

_SARCASTIC = """STYLE = SARCASTIC. One dry, deadpan, witty line that pokes fun at the scene while every
stated fact stays TRUE. Understate or mock-overhype what actually happens — the irony is in
the attitude, NEVER in claiming the opposite of reality. Confident and specific; do not become
mean about real people. Vary your wording — do not lean on a stock hype word like "thrilling".
Example voice: "A person at a computer, apparently working, which is exactly what someone
would do if they were not working." """

_HUMOROUS_TECH = """STYLE = HUMOROUS (TECH). One or two PUNCHY lines that reframe the REAL scene through bold
software/engineering humor, stated CONFIDENTLY as the joke — NOT hedged with "like" or "as if".
Lean into: deployments, rollbacks, breaking changes, bugs, the stack trace, dev vs prod, merge
conflicts, refactoring, uptime, shipping to production. The metaphor IS the humor; keep it
anchored to a CONCRETE detail from the JSON — name the actual subject/object/action (the kitten,
the zucchini, the waves), never a generic word like "landscape", "scene", or "environment".
Do not invent a new literal object or person that isn't there.
Example voice: "Nature's annual deployment: all leaf nodes updated to yellow simultaneously,
no breaking changes reported." """

_HUMOROUS_NON_TECH = """STYLE = HUMOROUS (EVERYDAY). One or two warm, funny, relatable lines — like a witty friend
narrating. ABSOLUTELY NO tech, software, engineering, or internet words (no code, apps, servers,
deploys, bugs, latency, algorithms, uploads/downloads, wifi, online). Draw the joke ONLY from
everyday life: food, chores, weather, relationships, moods, effort. Confident and specific to
THIS scene.
Example voice: "A woman at a computer, visibly handling something extremely important that will
be completely forgotten by Thursday." """

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
ACCURACY_EVAL = """You are checking video captions for MAJOR hallucinations against VERIFIED facts about
a video. A caption FAILS only if it asserts a real, LITERAL thing that is NOT in the facts —
an invented person, object, place, action, brand, or specific number that a viewer would take
as a factual claim about the scene (anything listed under "uncertain" counts as NOT supported).

NOT a failure (do NOT flag these): jokes, sarcasm, hyperbole, and clearly-figurative
software/engineering or everyday-life METAPHORS. A caption may confidently frame the real scene
through a witty lens (e.g. calling autumn leaves a "deployment", or a kitten an "autonomous
agent") — that is intended STYLE, not a hallucination, as long as the real subject, action, and
setting stay recognizable.

FACTS: <<JSON>>

CAPTIONS:
<<CAPS>>

List ONLY the labels of captions that fail (invent a literal, real detail not in the facts).
Output ONE JSON object wrapped in <json></json>:
<json>{"failed": ["label", ...]}</json>
If every caption is acceptable, output <json>{"failed": []}</json>. No other text."""


def accuracy_eval(facts_json, labeled_caps):
    return ACCURACY_EVAL.replace("<<JSON>>", facts_json).replace("<<CAPS>>", labeled_caps)
