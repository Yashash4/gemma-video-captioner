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
# SYSTEM — persistent rules for EVERY call (Gemma-4 native system role). The
# per-call user turns carry the frames + the specific ask + the exact delimiter.
# Kept delimiter-agnostic (<json> OR <caption>) so one system fits grounding,
# self-check, styling and eval; the parenthetical protects figurative humor so
# the styling calls are not pushed into over-literal captions.
# ----------------------------------------------------------------------------
SYSTEM = """You are a precise video assistant. These rules ALWAYS apply:
- Base everything on what is actually VISIBLE in the frames. If you are not sure of a detail, hedge it — never guess.
- Do NOT infer brand names, exact locations, companies, or personal identities unless a clear, legible on-screen sign or logo proves it. (Figurative jokes, metaphors, and comparisons are fine — they are not factual claims.)
- Reply with ONLY the single delimited block you are asked for — either a <json>...</json> object OR a <caption>...</caption> line — with no prose, markdown, notes, or chain-of-thought before or after it."""

# ----------------------------------------------------------------------------
# Stage A — grounding (ONE vision call). §5.1 schema + uncertainty flags.
# The visible-only / no-infer / output-only discipline now lives in SYSTEM;
# this user turn keeps the schema, the format example, and the on-screen-text
# nuance, plus the exact <json> delimiter contract the parser depends on.
# ----------------------------------------------------------------------------
# v12 (Fireworks full-res vision): RICHER grounding. Ollama capped images at ~280 tokens so
# "be specific" made it invent (v1/v9 regressions); Fireworks sees full detail (reads real signs,
# car colors), so richness = ACCURATE detail. Discipline kept: tiny/distant/blurry -> generic +
# "uncertain", so we don't over-claim past the judge's 768px/6-frame verification.
GROUNDING = """You see FRAMES sampled from ONE video. Describe what is visible in RICH, CONCRETE
detail — colors, counts, distinctive features, spatial layout, and any clearly legible text —
but ONLY what you can actually SEE. If a detail is blurry, distant, tiny, ambiguous, or guessed,
describe it GENERICALLY and put the specific guess in "uncertain" (never in the main fields).

Be specific in each field:
- subjects: name each with its COLOR, COUNT, and any distinctive feature (e.g. "three red sedans",
  "a woman in a beige jacket with a high bun").
- actions: the specific motion AND direction (e.g. "walking left-to-right toward the camera").
- setting: specific place, time of day, and light/weather if visible.
- on_screen_text: read any CLEARLY LEGIBLE sign, logo, or caption VERBATIM. A sign that is present
  but not clearly readable is NOT text — put "an unreadable sign" in "uncertain".
- spatial: where the main subjects sit (foreground/background, left/right/center).

Return exactly ONE JSON object with these keys, wrapped in <json></json>:
<json>{
  "subjects": [],
  "actions": [],
  "setting": "",
  "on_screen_text": [],
  "spatial": "",
  "audio_summary": "no clear speech",
  "mood": "",
  "uncertain": []
}</json>

Example of the exact shape (UNRELATED video — copy the format, not the content, no reasoning):
<json>{"subjects":["two red kites","a small child in a yellow coat"],"actions":["the kites climbing left-to-right"],"setting":"a windy beach in the afternoon","on_screen_text":[],"spatial":"the kites are high in the background, the child is in the foreground center","audio_summary":"no clear speech","mood":"energetic","uncertain":["the exact beach location"]}</json>

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
  never a generic template. The JSON may be RICH — do NOT list every detail; pick the 1-2 MOST
  vivid and build ONE punchy caption around them. Prefer ONE sentence; sarcastic/humor styles may
  use up to 2-3 short sentences. Hard limit ~40 words. No hashtags, no emojis, no preamble, no quotes.

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
stated fact stays TRUE. State the REAL subject and action first, then layer the dry aside on top —
do NOT invent a detail (a screen, a number, a label, an extra object) just to have something to mock.
Understate or mock-overhype what actually happens — the irony is in the attitude, NEVER in claiming
the opposite of reality. Confident and specific; do not become mean about real people. Vary your
wording — do not lean on a stock hype word like "thrilling".
Example voice: "A person sits at a computer typing with great purpose, which is exactly what someone
would do if they were not actually working." """

_HUMOROUS_TECH = """STYLE = HUMOROUS (TECH). Build EVERY caption in two layers. LAYER 1 (must come
FIRST): state the LITERAL scene in plain words — name the real subject, the real action they are
doing, and the setting straight from the JSON (the kitten walking, the sprinter running, the waves
rolling), never a generic word like "landscape", "scene", or "environment". LAYER 2: on TOP of that
already-true sentence, layer ONE software/engineering metaphor that MAPS to the actual ACTION or
CHANGE in THIS scene, tied to a real visible detail. The metaphor must FIT what is happening — it is
never generic decoration bolted onto any clip.
Pick the FITTING metaphor from this palette (do not reuse the same one across clips): deployment,
rollback, hotfix, refactor, compile, caching, merge conflict, debugging, indexing, garbage
collection, race condition, load balancing, CI pipeline, lazy loading, buffering, rendering.
Map it to the scene:
- Motion / travel / an arrival -> deploy, ship it, rollback (something ran away -> a rollback).
- Transformation (chopping, cooking, building, assembling) -> refactor, compile, merge.
- Something repeated, cleared, tidied, or sorted -> caching, indexing, garbage collection.
- A static or scenic shot -> a FITTING idle metaphor: a cached view, a frozen render, an idle
  process, a read-only replica, a long buffer, a graceful shutdown (a sunset). NEVER "uptime".
FORBIDDEN filler — do NOT use ANY of these words, they map to nothing and read as lazy: "uptime",
"downtime", "100% uptime", "99.9%", "high-availability", "zero downtime".
ONE metaphor per caption, mapped to THIS scene's real subject and action. The joke ADDS to a
literally-true sentence; it must NEVER replace the real verb. Litmus test: delete the tech joke and
the sentence must STILL correctly say who is doing what, and where. Do not invent a literal object or
person that isn't there.
BAD (real action hidden by the metaphor): "Shipping a hotfix straight to production at 2 AM before
the whole stack crashes." — the real typist and laptop vanished.
BAD (lazy filler mapped to nothing): "A calm coastline glows at sunset, running at 100% uptime." —
uptime describes NOTHING in the shot.
GOOD (metaphor MAPS to the action): "A chef dices the onions fast on a wooden board, refactoring one
big vegetable into a dozen clean little components." — chopping IS refactoring.
GOOD (fitting idle metaphor for a still shot): "The empty green valley sits motionless at dawn, a
read-only replica nobody has pushed a change to yet." — stillness IS read-only.
GOOD (motion mapped): "A brown dog bolts across the yard away from its owner, executing an
unscheduled rollback." — running off IS a rollback. """

_HUMOROUS_NON_TECH = """STYLE = HUMOROUS (EVERYDAY). Build EVERY caption in two layers. LAYER 1 (must come
FIRST): state the LITERAL scene in plain words — name the real subject, the real action, and the
setting straight from the JSON (the man sprinting, the woman typing, the sun setting). LAYER 2: on
TOP of that already-true sentence, layer ONE warm, relatable everyday aside — food, chores, weather,
relationships, moods, running late, effort. The joke ADDS to a literally-true sentence; it must NEVER
replace the real verb with a figurative one. Litmus test: delete the joke and the sentence must STILL
correctly say who is doing what, and where.
ABSOLUTELY NO tech, software, engineering, or internet words (no code, apps, servers, deploys, bugs,
latency, algorithms, uploads/downloads, wifi, online).
BAD (no real subject or action, just vague whimsy): "Just a little bit of everyday magic in motion."
GOOD (real action named, joke layered on top): "A man sprints down the track, running like he is late
for the last slice of pizza." — the real sprint stays; the everyday joke just rides along. """

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
