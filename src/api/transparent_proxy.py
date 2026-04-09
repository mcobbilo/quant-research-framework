import os
import json
import uuid
import httpx
from datetime import datetime
from typing import Request, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# Create standalone proxy or it can be mounted somewhere
proxy_app = FastAPI(
    title="MetaClaw Transparent Proxy",
    description="Intercepts and logs completions requests to train the RL engine."
)

LOG_FILE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "proxy_logs.jsonl"
)

# Ensure the data directory exists
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


async def forward_request(method: str, url: str, headers: dict, body: bytes) -> httpx.Response:
    # Filter out headers that might cause issues like Host
    filtered_headers = {
        k: v for k, v in headers.items() 
        if k.lower() not in ["host", "content-length", "connection"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=url,
            headers=filtered_headers,
            content=body,
            timeout=120.0
        )
    return response


@proxy_app.post("/v1/chat/completions")
async def proxy_openai_chat(request: Request):
    """
    Transparently intercepts OpenAI /v1/chat/completions requests.
    """
    body = await request.body()
    try:
        body_json = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        body_json = {"raw_body_decode_error": True}

    target_url = "https://api.openai.com/v1/chat/completions"
    
    # Forward the request to the actual OpenAI API
    try:
        response = await forward_request(
            method=request.method,
            url=target_url,
            headers=dict(request.headers),
            body=body
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Proxy forwarding failed: {str(e)}")
    
    # Attempt to decode the response for logging
    response_content = response.content
    try:
        response_json = json.loads(response_content.decode("utf-8"))
        is_success = response.status_code == 200
    except json.JSONDecodeError:
        response_json = {"raw_response_decode_error": True}
        is_success = False

    # Log the interaction silently
    interaction_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "trace_id": str(uuid.uuid4()),
        "target": "openai",
        "request_payload": body_json,
        "response_payload": response_json,
        "status_code": response.status_code,
        "is_success": is_success
    }

    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(interaction_log) + "\n")

    return JSONResponse(
        content=response_json,
        status_code=response.status_code,
        headers={k: v for k, v in response.headers.items() if k.lower() not in ["content-length", "content-encoding"]}
    )


@proxy_app.post("/v1/messages")
async def proxy_anthropic_messages(request: Request):
    """
    Transparently intercepts Anthropic /v1/messages requests.
    """
    body = await request.body()
    try:
        body_json = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        body_json = {"raw_body_decode_error": True}

    target_url = "https://api.anthropic.com/v1/messages"
    
    # Forward the request to the actual Anthropic API
    try:
        response = await forward_request(
            method=request.method,
            url=target_url,
            headers=dict(request.headers),
            body=body
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Proxy forwarding failed: {str(e)}")
    
    # Attempt to decode the response for logging
    response_content = response.content
    try:
        response_json = json.loads(response_content.decode("utf-8"))
        is_success = response.status_code == 200
    except json.JSONDecodeError:
        response_json = {"raw_response_decode_error": True}
        is_success = False

    # Log the interaction silently
    interaction_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "trace_id": str(uuid.uuid4()),
        "target": "anthropic",
        "request_payload": body_json,
        "response_payload": response_json,
        "status_code": response.status_code,
        "is_success": is_success
    }

    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(interaction_log) + "\n")

    return JSONResponse(
        content=response_json,
        status_code=response.status_code,
        headers={k: v for k, v in response.headers.items() if k.lower() not in ["content-length", "content-encoding"]}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("transparent_proxy:proxy_app", host="0.0.0.0", port=8000, reload=True)
