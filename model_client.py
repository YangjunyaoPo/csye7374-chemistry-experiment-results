"""JSON action transport for a local or explicitly configured chat model."""

import json
import os
from urllib import error, parse, request


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-14b"
MAX_REPLY_BYTES = 128 * 1024


class ModelError(RuntimeError):
    """A model endpoint is unavailable or supplied an unusable response."""


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward the question, observations or credentials to another URL.
        return None


class ModelClient:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=120):
        url = base_url if base_url is not None else os.getenv("CHEM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model if model is not None else os.getenv("CHEM_MODEL", DEFAULT_MODEL)
        self._api_key = api_key if api_key is not None else os.getenv("CHEM_API_KEY")
        self.timeout = timeout
        try:
            parts = parse.urlsplit(url)
            host = parts.hostname
            _ = parts.port  # Validate the optional port before making the request.
        except ValueError as exc:
            raise ModelError("Invalid model endpoint URL.") from exc
        if not host or parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ModelError("Use a model base URL without credentials, query or fragment.")
        loopback = host in {"localhost", "127.0.0.1", "::1"}
        if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
            raise ModelError("Use HTTPS for a remote model or HTTP on loopback.")
        if not self.model or not self.model.strip():
            raise ModelError("Set CHEM_MODEL to the identifier of a chat model.")
        self.endpoint = url.rstrip("/") + "/chat/completions"
        handlers = [_NoRedirect()]
        if loopback:
            handlers.append(request.ProxyHandler({}))
        self._opener = request.build_opener(*handlers)

    def decide(self, messages):
        """Send the complete observation history and return one proposed action."""
        history = list(messages)
        if "qwen3" in self.model.lower() and history and history[0].get("role") == "system":
            history[0] = {**history[0], "content": history[0]["content"] + "\n/no_think"}
        body = json.dumps({
            "model": self.model, "messages": history, "temperature": 0,
            "max_tokens": 1000, "stream": False,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        query = request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with self._opener.open(query, timeout=self.timeout) as reply:
                raw = reply.read(MAX_REPLY_BYTES + 1)
        except error.HTTPError as exc:
            exc.close()
            raise ModelError(f"Model endpoint returned HTTP {exc.code}. Check the model ID, endpoint and credentials.") from None
        except (OSError, ValueError):
            raise ModelError("Cannot contact the model endpoint. Check the URL, server and timeout.") from None
        if len(raw) > MAX_REPLY_BYTES:
            raise ModelError("Model response exceeded the size limit.")
        try:
            response = json.loads(raw)
            choice = response["choices"][0]
            content = choice["message"]["content"].strip()
            if choice.get("finish_reason") == "length":
                raise ModelError("Model response reached its output limit.")
            # Some Qwen backends include an empty or textual thinking block in content.
            if content.startswith("<think>") and "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content[8:-4].strip()
            action = json.loads(content)
        except (ValueError, TypeError, IndexError, KeyError, AttributeError):
            raise ModelError("Model response did not contain a JSON action.") from None
        if not isinstance(action, dict):
            raise ModelError("Expected a JSON action object.")
        return action
