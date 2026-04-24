import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDelta:
    def __init__(self, content=None):
        self.content = content


class FakeChoice:
    def __init__(self, message=None, delta=None, text=None):
        self.message = message
        self.delta = delta
        self.text = text


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(message=FakeMessage(content))]


class FakeChunk:
    def __init__(self, content=None, empty=False):
        self.choices = [] if empty else [FakeChoice(delta=FakeDelta(content))]


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChat:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)


class FakeClient:
    def __init__(self, outcomes):
        self.chat = FakeChat(outcomes)


class UnsupportedExtraBody(Exception):
    status_code = 400


class InferenceCompatTests(unittest.TestCase):
    def test_non_stream_requests_disable_thinking_by_default(self):
        module = load_module("java/inference.py", "java_inference_for_test")
        client = FakeClient([FakeResponse("[CODE]\nreturn this;")])

        model = module.OpenAIModel(
            model_id="any-reasoning-model",
            max_new_tokens=32,
            client=client,
        )

        self.assertEqual(model.infer("prompt"), "[CODE]\nreturn this;")
        call = client.chat.completions.calls[0]
        self.assertFalse(call["stream"])
        self.assertEqual(call["extra_body"]["enable_thinking"], False)

    def test_unsupported_extra_body_retries_without_it(self):
        module = load_module("java/inference.py", "java_inference_retry_test")
        client = FakeClient([
            UnsupportedExtraBody("unsupported parameter: enable_thinking"),
            FakeResponse("ok"),
        ])

        model = module.OpenAIModel(model_id="plain-model", client=client)

        self.assertEqual(model.infer("prompt"), "ok")
        self.assertIn("extra_body", client.chat.completions.calls[0])
        self.assertNotIn("extra_body", client.chat.completions.calls[1])

    def test_streaming_skips_empty_chunks(self):
        module = load_module("rust/inference.py", "rust_inference_for_test")
        client = FakeClient([[FakeChunk(empty=True), FakeChunk("fn ok() {}")]])

        model = module.OpenAIModel(model_id="stream-model", client=client, stream=True)

        self.assertEqual(model.infer("prompt"), "fn ok() {}")

    def test_reasoning_markup_is_removed_from_final_content(self):
        module = load_module("rust/inference.py", "rust_inference_markup_test")

        self.assertEqual(
            module.remove_reasoning_markup("<think>hidden</think>\nfn ok() {}"),
            "fn ok() {}",
        )
        self.assertEqual(module.remove_reasoning_markup("think>\n\nok"), "ok")


if __name__ == "__main__":
    unittest.main()
