import uuid


def handler(event):
    docs = event["segments"]
    input_bucket = event["input_bucket"]
    output_bucket = event["output_bucket"]
    benchmark_bucket = event["benchmark_bucket"]
    max_tokens = event["max_tokens"]
    question = event["question"]

    return {
        "segments": [
            {
                "prefix": str(uuid.uuid4().int & ((1 << 64) - 1))[:8],
                "doc": doc,
                "input_bucket": input_bucket,
                "output_bucket": output_bucket,
                "benchmark_bucket": benchmark_bucket,
                "max_tokens": max_tokens,
                "question": question,
            }
            for doc in docs
        ]
    }
