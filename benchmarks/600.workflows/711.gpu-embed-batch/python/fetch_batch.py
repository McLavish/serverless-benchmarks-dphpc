def handler(event):
    texts = event.get(
        "texts",
        [
            "serverless workflows on gpu",
            "short test sentence",
            "embedding benchmark sample",
            "fan out and summarize",
        ],
    )
    return {"texts": texts}
