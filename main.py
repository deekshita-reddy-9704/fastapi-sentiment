from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import re

app = FastAPI(title="Batch Sentiment API")

class SentencesIn(BaseModel):
    sentences: List[str]

POSITIVE = {
    "love","loved","loving","like","liked","enjoy","enjoyed","enjoying",
    "great","good","wonderful","amazing","excellent","awesome","best",
    "fantastic","pleased","delighted","positive","happy","joy","joyful",
    "cute","brilliant","favorite","favourite","yay","yay!"
}
NEGATIVE = {
    "hate","hated","hating","dislike","disliked","angry","angry","mad",
    "terrible","awful","horrible","worst","sad","sadness","disappointed",
    "disappointing","poor","sucks","sucked","annoyed","upset","unhappy",
    "tragic","ruined","problem","problems"
}
# simple emoticon sets
EMOJI_HAPPY = {"🙂","😊","😃","😄","😀","😁","😍","😺","👍"}
EMOJI_SAD = {"☹️","😢","😞","😟","😿","👎"}

# punctuation that can boost sentiment
EXCLAMATION_BOOST = 1.2

_negation_tokens = {"not","no","never","n't","hardly","rarely","neither"}

def tokenize_keep_positions(text: str):
    """Return list of lower token strings; simple word tokenizer"""
    # keep 'don't' as dont etc.
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    return tokens

def contains_emoji(text: str):
    for ch in text:
        if ch in EMOJI_HAPPY: 
            return "happy"
        if ch in EMOJI_SAD:
            return "sad"
    return None

def score_sentence(text: str) -> str:
    if not text or not text.strip():
        return "neutral"

    # emoji quick path
    emoji_sent = contains_emoji(text)
    if emoji_sent:
        return emoji_sent

    tokens = tokenize_keep_positions(text)
    pos_count = 0.0
    neg_count = 0.0

    # For negation handling: we consider a negation affecting next 3 tokens
    negation_positions = set(i for i,t in enumerate(tokens) if t in _negation_tokens)

    for i, token in enumerate(tokens):
        # simple exact match in sets
        if token in POSITIVE:
            # check if negation appears within 3 tokens before this token
            negated = any((i - 3) <= p < i for p in negation_positions)
            if negated:
                neg_count += 1.0
            else:
                pos_count += 1.0
        if token in NEGATIVE:
            negated = any((i - 3) <= p < i for p in negation_positions)
            if negated:
                pos_count += 1.0
            else:
                neg_count += 1.0

    # punctuation boost: exclamations tend to indicate stronger emotion
    if "!" in text:
        # if net positive, boost pos_count; if net negative, boost neg_count
        if pos_count > neg_count:
            pos_count *= EXCLAMATION_BOOST
        elif neg_count > pos_count:
            neg_count *= EXCLAMATION_BOOST
        # if neither, exclamation alone slightly leans positive
        else:
            pos_count += 0.2

    # very short sentences: use heuristics on words like "yes"/"no"/"yess"/"nah"
    short_map = {"yes":"happy","yeah":"happy","yup":"happy","yess":"happy","yesss":"happy",
                 "no":"sad","nah":"sad","nope":"sad"}
    if len(tokens) <= 3 and tokens:
        for t in tokens:
            if t in short_map:
                return short_map[t]

    # final decision
    if pos_count - neg_count > 0.3:
        return "happy"
    if neg_count - pos_count > 0.3:
        return "sad"
    return "neutral"

@app.post("/sentiment")
def batch_sentiment(payload: SentencesIn) -> Dict[str, Any]:
    # validate
    if payload.sentences is None:
        raise HTTPException(status_code=400, detail="Missing 'sentences' field")
    if not isinstance(payload.sentences, list):
        raise HTTPException(status_code=400, detail="'sentences' must be a list")

    results = []
    for s in payload.sentences:
        # ensure we preserve original exactly as requested in output
        if not isinstance(s, str):
            # convert non-strings to str to keep ordering and outputs predictable
            s_text = str(s)
        else:
            s_text = s
        sentiment = score_sentence(s_text)
        # enforce only allowed labels
        if sentiment not in {"happy","sad","neutral"}:
            sentiment = "neutral"
        results.append({"sentence": s_text, "sentiment": sentiment})

    return {"results": results}
