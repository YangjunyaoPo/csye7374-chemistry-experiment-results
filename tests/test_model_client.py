import io
import json
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from model_client import MAX_REPLY_BYTES, ModelClient, ModelError


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        self.client = ModelClient("http://127.0.0.1:1234/v1", "qwen3-test", api_key="test-key")
        self.client._opener = Mock()
        self.history = [{"role": "system", "content": "Choose a JSON action."},
                        {"role": "user", "content": "List the experiment records."}]

    def response(self, content, finish="stop"):
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": finish,
                          "message": {"content": content}}]}).encode())

    def test_history_and_authorization_are_sent_and_input_is_immutable(self):
        action = {"action": "list_records", "arguments": {}, "reason": "Read the experiment IDs."}
        self.client._opener.open.return_value = self.response(json.dumps(action))
        self.assertEqual(self.client.decide(self.history), action)
        sent = self.client._opener.open.call_args.args[0]
        payload = json.loads(sent.data)
        self.assertEqual(sent.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(payload["model"], "qwen3-test")
        self.assertEqual(payload["messages"][-1], self.history[-1])
        self.assertEqual(self.history[0]["content"], "Choose a JSON action.")

    def test_non_json_truncation_and_empty_choices(self):
        for reply in (self.response("not a JSON action"), self.response("{}", "length"),
                      io.BytesIO(b'{"choices":[]}'), self.response("[]")):
            with self.subTest(reply=reply):
                self.client._opener.open.return_value = reply
                with self.assertRaises(ModelError):
                    self.client.decide(self.history)

    def test_size_limit(self):
        self.client._opener.open.return_value = io.BytesIO(b"x" * (MAX_REPLY_BYTES + 1))
        with self.assertRaisesRegex(ModelError, "size limit"):
            self.client.decide(self.history)

    def test_transport_errors_omit_credentials_and_private_response(self):
        private_text = "test-key; private provider detail"
        for failure in (TimeoutError(private_text), HTTPError("https://example.org/v1", 401,
                        private_text, {}, io.BytesIO(private_text.encode()))):
            self.client._opener.open.side_effect = failure
            with self.assertRaises(ModelError) as caught:
                self.client.decide(self.history)
            self.assertNotIn("test-key", str(caught.exception))
            self.assertNotIn("private provider", str(caught.exception))

    def test_insecure_remote_and_credential_urls_rejected(self):
        for url in ("http://example.org/v1", "https://u:p@example.org", "https://host/v1?k=secret",
                    "file:///tmp/data", "https://host:invalid/v1"):
            with self.subTest(url=url), self.assertRaises(ModelError):
                ModelClient(url)

    def test_fenced_json_after_thinking_block(self):
        self.client._opener.open.return_value = self.response(
            '<think>irrelevant</think>\n```json\n{"action":"list_records"}\n```')
        self.assertEqual(self.client.decide(self.history), {"action": "list_records"})
