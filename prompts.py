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
GROUNDING = """You see FRAMES sampled from ONE video. Report ONLY what is actually visible,
but be SPECIFIC and concrete — vague descriptions lose. If you are genuinely unsure, put it
in "uncertain" — never guess.

READ the frames closely and capture:
- subjects: people/animals/objects, WITH distinguishing detail (color, count, clothing,
  type/model, breed, what kind).
- actions: what each subject is doing; visible motion; how the scene changes across frames.
- setting: place, environment, time of day, weather — as specific as the frames prove.
- on_screen_text: TRANSCRIBE any legible text you can read — signs, captions, scoreboards,
  jersey/car numbers, timers, prices, logos, watermarks — exactly as shown. This is high-value;
  read it. Put text you truly cannot make out in "uncertain" instead, never a guess.
- notable_details: distinctive colors, counts, salient objects, anything that makes THIS clip
  specific rather than generic.
- mood: the overall vibe.
- uncertain: things hinted at but not confirmable (illegible text, guessed brands/places/identities).

Do NOT infer brand names, exact locations, or personal identities unless a clearly legible
sign or logo proves it.

Return exactly ONE JSON object with these keys, wrapped in <json></json>:
<json>{
  "subjects": [],
  "actions": [],
  "setting": "",
  "on_screen_text": [],
  "notable_details": [],
  "mood": "",
  "uncertain": []
}</json>

Example of the exact shape to return (for an UNRELATED video — copy the FORMAT, not the content, and do NOT show any reasoning):
<json>{"subjects":["a red rally car, number 7"],"actions":["sliding through a gravel corner","kicking up dust"],"setting":"a forest rally stage on an overcast day","on_screen_text":["STAGE 4","07"],"notable_details":["red and white livery","spectators lining the track"],"mood":"tense, fast","uncertain":["the sponsor names on the doors"]}</json>

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
- Keep the plainly-visible facts (subjects, actions, setting, mood, notable_details) and any
  clearly-legible on_screen_text unchanged — do NOT strip specific detail you can see.
Keep the same keys. Output ONLY the cleaned object wrapped in <json></json>, nothing else.

<json><<JSON>></json>"""

# ----------------------------------------------------------------------------
# Stage B — styling. Shared preamble (§4, verbatim) + strict <caption> delimiter.
# ----------------------------------------------------------------------------
PREAMBLE = """You write ONE English caption for a video, given a verified JSON description of what
the video contains.

YOUR TARGET: one clear sentence that names the CONCRETE subject + its action + the setting,
plus ONE salient real detail from the JSON (a color, a count, a legible sign/number, a
distinctive object). SPECIFIC-and-true beats vague-but-safe, and specific beats long — a
short precise caption scores as high as a long one. Do NOT pad. ~12-32 words, 1-2 sentences.

GROUNDING DISCIPLINE (applies to EVERY style — a hallucination is the heaviest penalty):
- Use only what is in the JSON. Do NOT invent actions, reactions, named people/places/brands,
  numbers, intent, or events that are not there. If the JSON marks something "uncertain", do
  not assert it as fact.
- If on-screen text is present in the JSON, you MAY name it; if text was unreadable, stay
  generic about it — never make up what a sign or number says.
- A clearly-figurative joke or metaphor is welcome, but it must map onto a REAL visible detail
  from the JSON. What is forbidden is asserting a literal thing that isn't there (a person,
  object, place, or action not in the JSON).

OUTPUT: reply with ONLY the finished caption wrapped in <caption></caption> — a real,
complete sentence. No reasoning, no drafts, no notes, no placeholders, no backticks, no
hashtags, no emojis, no quotes. Copy the FORMAT of these examples (content UNRELATED to your video):

JSON: {"subjects":["a red kite"],"actions":["flying"],"setting":"a windy beach","notable_details":["a long orange tail"],"uncertain":[]}
<caption>A red kite with a long orange tail climbs and rolls high over a windy beach, tugging against its line.</caption>

JSON: {"subjects":["three chefs"],"actions":["plating dishes"],"setting":"a busy kitchen","on_screen_text":[],"uncertain":["the restaurant name"]}
<caption>Three chefs work in tight coordination, plating dishes across a busy kitchen at full speed.</caption>"""

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

_HUMOROUS_TECH = """STYLE = HUMOROUS (TECH). ONE punchy line that reframes the REAL scene through a SINGLE
software/engineering metaphor, stated CONFIDENTLY as the joke — NOT hedged with "like" or "as if".
Pick one lens (deployment, rollback, breaking change, bug, stack trace, dev vs prod, merge
conflict, refactor, uptime, shipping to prod) and map it onto the ACTUAL subject/action from the
JSON — name the real thing (the rally car, the sunset, the kitten), never a generic word like
"landscape", "scene", or "environment", and never pile on a second unrelated metaphor. Do not
invent a new literal object or person that isn't there.
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

def style_prompt(style_name, facts_json):
    """Full Stage-B prompt: preamble + style block (known or generic) + the frozen facts."""
    block = _STYLE_BLOCKS.get(style_name)
    if block is None:
        block = _GENERIC.replace("{NAME}", style_name.upper()).replace("{name}", style_name.replace("_", " "))
    return PREAMBLE + "\n\n" + block + "\n\nJSON: " + facts_json


def self_check(facts_json):
    return SELF_CHECK.replace("<<JSON>>", facts_json)
