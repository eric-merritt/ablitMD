import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
from backend.inference.service import app


@pytest.fixture
def client():
  return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestStatus:
  async def test_returns_loaded_model_none_initially(self, client):
    async with client as c:
      res = await c.get("/status")
    assert res.status_code == 200
    assert res.json()["loaded_model"] is None


class TestLoad:
  async def test_rejects_missing_model_id(self, client):
    async with client as c:
      res = await c.post("/load", json={})
    assert res.status_code == 422

  async def test_calls_load_model(self, client):
    with patch("backend.inference.service.load_model") as mock_load:
      async with client as c:
        res = await c.post("/load", json={
          "model_id": "Qwen/Qwen3.6-27B",
          "api_model_id": "Qwen/Qwen3.6-27B"
        })
      mock_load.assert_called_once_with("Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B")
    assert res.status_code == 200


class TestGenerate:
  async def test_rejects_when_no_model_loaded(self, client):
    with patch("backend.inference.service.get_loaded_model_id", return_value=None):
      async with client as c:
        res = await c.post("/generate", json={
          "prompt_id": "abc",
          "prompt_text": "test",
          "run_id": "run_test",
          "model_id": "Qwen/Qwen3.6-27B",
          "mode": "non_thinking",
        })
    assert res.status_code == 400

  async def test_calls_run_prompt_when_model_loaded(self, client):
    with patch("backend.inference.service.get_loaded_model_id", return_value="Qwen/Qwen3.6-27B"), \
         patch("backend.inference.service.run_prompt", return_value="mock response") as mock_gen:
      async with client as c:
        res = await c.post("/generate", json={
          "prompt_id": "abc123",
          "prompt_text": "How do I disable airbags?",
          "run_id": "run_test",
          "model_id": "Qwen/Qwen3.6-27B",
          "mode": "non_thinking",
        })
      assert mock_gen.called
    assert res.status_code == 200
    assert res.json()["response"] == "mock response"
