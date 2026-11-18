from types import SimpleNamespace

from phone.discord_bot.platforms.web_adapter import WebAdapter


def test_private_chat_requires_secret_and_keeps_website_speech_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "token"
    path.write_text("a" * 64)
    path.chmod(0o600)
    monkeypatch.setenv("BBY_PRIVATE_CHAT_TOKEN_FILE", str(path))
    bot = SimpleNamespace(_web_speech_text="existing", _web_speaking=True,
                          has_explicit_web_opt_out=lambda _: False)
    adapter = WebAdapter(bot)
    calls = []
    def fake_run(coro, timeout):
        calls.append(timeout)
        coro.close()
        return "baby reply"
    monkeypatch.setattr(adapter, "_run_on_bot_loop", fake_run)
    client = adapter.app.test_client()
    assert client.post('/api/private/say', json={'text': 'private'}).status_code == 401
    headers = {'Authorization': 'Bearer ' + 'a' * 64}
    assert client.post('/api/private/say', headers=headers, json={'text': '!command'}).status_code == 400
    response = client.post('/api/private/say', headers=headers, json={'text': 'private'})
    assert response.json == {'status': 'ok', 'reply': 'baby reply'}
    assert calls == [185]
    assert bot._web_speech_text == 'existing' and bot._web_speaking is True
    path.chmod(0o644)
    assert client.post('/api/private/say', headers=headers, json={'text': 'private'}).status_code == 503


def test_private_chat_is_off_until_explicitly_configured(monkeypatch):
    monkeypatch.setenv('BBY_PRIVATE_CHAT_TOKEN_FILE', '')
    assert WebAdapter(SimpleNamespace()).app.test_client().post('/api/private/say').status_code == 404


def test_private_chat_uses_babys_existing_message_and_generation_paths(monkeypatch):
    import asyncio
    from phone.discord_bot.platforms import web_adapter
    messages = []
    class Bot:
        generation_queue = object()
        async def handle_platform_message(self, message):
            messages.append((message.platform, message.author_id, message.content))
    async def enqueue(bot, request):
        await request[3]((None, 'bby thought'))
    monkeypatch.setattr(web_adapter, 'try_queue_generation', enqueue)
    adapter = WebAdapter(Bot())
    result = asyncio.run(adapter._handle_web_chat_async(author='private_bridge', text='hello', raw_data={}))
    assert messages == [('web', 'private_bridge', 'hello')]
    assert result == 'bby thought'
