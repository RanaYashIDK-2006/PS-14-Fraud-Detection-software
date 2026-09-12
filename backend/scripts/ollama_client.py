#!/usr/bin/env python3
"""Connect to local Ollama gemma4:26b model.

Usage:
    python scripts/ollama_client.py "your prompt here"
    python scripts/ollama_client.py --interactive
    python scripts/ollama_client.py --api  # starts a local HTTP API on port 11435
"""
import sys
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:26b"


def generate(prompt: str, stream: bool = False) -> str:
    """Send a prompt to Ollama and get a response."""
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    if stream:
        resp = urllib.request.urlopen(req, timeout=300)
        full_response = []
        for line in resp:
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response.append(token)
                if chunk.get("done"):
                    break
        print()
        return "".join(full_response)
    else:
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        return result.get("response", "")


def chat(messages: list[dict]) -> str:
    """Multi-turn chat with Ollama."""
    data = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    return result.get("message", {}).get("content", "")


def interactive():
    """Interactive chat loop."""
    print(f"Connected to {MODEL} via Ollama")
    print(f"URL: {OLLAMA_URL}")
    print(f"Type 'quit' to exit\n")

    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        response = chat(messages)
        messages.append({"role": "assistant", "content": response})
        print(f"\n{MODEL}: {response}\n")


def start_api_server(port: int = 11435):
    """Start a simple HTTP API that proxies to Ollama."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            prompt = data.get("prompt", "")
            response = generate(prompt, stream=False)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "model": MODEL,
                "response": response,
                "done": True,
            }).encode())

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, format, *args):
            print(f"[{self.log_date_time_string()}] {format % args}")

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Ollama proxy API running on http://127.0.0.1:{port}")
    print(f"Forwarding to {MODEL} at {OLLAMA_URL}")
    print(f"POST / with {{\"prompt\": \"...\"}} to query\n")
    server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            interactive()
        elif sys.argv[1] == "--api":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 11435
            start_api_server(port)
        else:
            prompt = " ".join(sys.argv[1:])
            response = generate(prompt, stream=True)
    else:
        interactive()
