"""Tests for GitHub metadata extraction behavior."""

from hackathon_analysis.data_extraction import github_extractor as ge


def test_fetch_repo_metadata_stops_after_rate_limit(monkeypatch):
    """If rate limit is hit, remaining repos should be marked without extra API calls."""

    class FakeClient:
        def __init__(self, token=None):
            self.calls = 0

        def graphql(self, query, variables):
            self.calls += 1
            raise RuntimeError("403 Client Error: rate limit exceeded for url: https://api.github.com/graphql")

    fake_client = FakeClient()
    monkeypatch.setattr(ge, "GitHubClient", lambda token=None: fake_client)

    urls = [
        "https://github.com/example/one",
        "https://github.com/example/two",
        "https://github.com/example/three",
    ]
    out = ge.fetch_repo_metadata(urls, token=None)

    assert set(out.keys()) == set(urls)
    assert fake_client.calls == 1
    assert "rate limit" in out[urls[0]]["error"].lower()
    assert "stopped early" in out[urls[1]]["error"].lower()
    assert "stopped early" in out[urls[2]]["error"].lower()
