"""upload_report_to_r2 must not invent a public address.

It used to fall back to "https://pub-default.r2.dev" when R2_PUBLIC_URL_BASE was
unset, and return a URL to a host that does not resolve. Telegram renders that as a
link and it fails on SSL. Suppressing it was then every caller's job, and callers
are added over time: three of five stripped the sentinel, and the two that did not
are weekly_forecast's own — every weekly card from at least 2 Aug to 6 Sep 2026 went
out with https://pub-default.r2.dev/reports/weekly_report_*.html, stored in
weekly_reports.r2_url as well.

So the contract is the fix: a returned URL is one a reader can open, and None means
there is no such address — whether because the upload failed or because the bucket
has no public base.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.pipeline.weekly_forecast as wf


@pytest.fixture
def _r2_creds(monkeypatch):
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("R2_PUBLIC_URL_BASE", raising=False)
    monkeypatch.delenv("R2_BUCKET_NAME", raising=False)


class TestUploadReportToR2:
    def test_no_public_base_returns_none(self, _r2_creds):
        with patch.object(wf.boto3, "client", return_value=MagicMock()):
            assert wf.upload_report_to_r2("a.html", b"x", "text/html") is None

    def test_the_file_is_still_uploaded(self, _r2_creds):
        # The archive object is worth having even when nobody can link to it.
        s3 = MagicMock()
        with patch.object(wf.boto3, "client", return_value=s3):
            wf.upload_report_to_r2("a.html", b"x", "text/html")
        assert s3.put_object.called
        assert s3.put_object.call_args.kwargs["Bucket"] == "sim-archive"

    def test_a_configured_base_produces_a_real_url(self, _r2_creds, monkeypatch):
        monkeypatch.setenv("R2_PUBLIC_URL_BASE", "https://files.example.org/")
        with patch.object(wf.boto3, "client", return_value=MagicMock()):
            url = wf.upload_report_to_r2("reports/a.html", b"x", "text/html")
        assert url == "https://files.example.org/reports/a.html"

    def test_it_never_returns_the_old_placeholder_host(self, _r2_creds):
        with patch.object(wf.boto3, "client", return_value=MagicMock()):
            url = wf.upload_report_to_r2("a.html", b"x", "text/html")
        assert url is None or "pub-default.r2.dev" not in url

    def test_the_private_archive_notice_is_said_once_not_per_upload(self, _r2_creds,
                                                                    monkeypatch, caplog):
        """A SITREP run uploads six files. Six identical lines an hour about a
        settled decision is how a log stops being read."""
        monkeypatch.setattr(wf, "_ANNOUNCED_PRIVATE_ARCHIVE", False)
        with caplog.at_level("INFO"), patch.object(wf.boto3, "client",
                                                   return_value=MagicMock()):
            for i in range(6):
                wf.upload_report_to_r2(f"a{i}.html", b"x", "text/html")
        said = [r for r in caplog.records if "private archive" in r.message]
        assert len(said) == 1

    def test_missing_credentials_still_return_none(self, monkeypatch):
        monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
        assert wf.upload_report_to_r2("a.html", b"x", "text/html") is None

    def test_an_upload_failure_returns_none(self, _r2_creds, monkeypatch):
        monkeypatch.setenv("R2_PUBLIC_URL_BASE", "https://files.example.org")
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("boom")
        with patch.object(wf.boto3, "client", return_value=s3):
            assert wf.upload_report_to_r2("a.html", b"x", "text/html") is None


class TestNoCallerStripsTheSentinelAnyMore:
    def test_the_sentinel_check_is_gone_from_every_consumer(self):
        """A guard each caller must remember is a defect waiting for the next
        caller — which is precisely how the weekly report and the first Iran
        bulletin both shipped the dead link."""
        import pathlib

        for path in ("src/pipeline/daily_sitrep.py",
                     "src/pipeline/iran_bulletin_run.py"):
            text = pathlib.Path(path).read_text()
            code = [ln for ln in text.splitlines()
                    if "pub-default.r2.dev" in ln and not ln.strip().startswith("#")]
            assert code == [], f"{path} still branches on the sentinel: {code}"
