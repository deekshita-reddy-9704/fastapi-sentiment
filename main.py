from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/sentiment")
async def sentiment_analysis(request: Request):
    data = await request.json()
    sentences = data.get("sentences", [])
    results = []

    for sentence in sentences:
        text = sentence.lower()
        if any(word in text for word in ["love", "great", "good", "amazing", "happy", "wonderful", "excellent"]):
            sentiment = "happy"
        elif any(word in text for word in ["sad", "bad", "terrible", "awful", "hate", "angry", "upset"]):
            sentiment = "sad"
        else:
            sentiment = "neutral"
        results.append({"sentence": sentence, "sentiment": sentiment})

    return {"results": results}

@app.get("/")
def root():
    return {"message": "FastAPI Sentiment Analysis API is running!"}
