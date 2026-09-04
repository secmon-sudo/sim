"""
validate_sitrep tolerance — a verification label the LLM editorialized (extra
words inside/after the parentheses) is normalized to the nearest canonical label
instead of failing the whole country report. Run #19 (29 Jul) lost IR's entire
SITREP to a single stray label: "Onaylandı (Çoklu kaynak, ancak detaylar
doğrulanmamış)". Normalization never raises the claimed confidence tier.
"""

import pytest

from src.core.sitrep_verify import LABEL_MULTI, LABEL_OFFICIAL, LABEL_SINGLE
from src.services.sitrep_generator import cites_a_listed_url, validate_sitrep

_HDR = "YÖNETİCİ ÖZETİ\nGünün özeti.\n"


def _label_line(label: str) -> str:
    return _HDR + f"— Bir olay. Doğruluk Durumu: {label}"


def _tail(out: str) -> str:
    return out.splitlines()[-1].split("Doğruluk Durumu:", 1)[1].strip()


class TestVerificationLabelNormalization:
    def test_run19_regression_multi_editorialized(self):
        out = validate_sitrep(
            _label_line("Onaylandı (Çoklu kaynak, ancak detaylar doğrulanmamış)"), [])
        assert _tail(out) == LABEL_MULTI

    def test_official_editorialized(self):
        out = validate_sitrep(_label_line("Onaylandı (Resmî, teyit sürüyor)"), [])
        assert _tail(out) == LABEL_OFFICIAL

    def test_single_editorialized(self):
        out = validate_sitrep(_label_line("Doğrulanmamış (Tek kaynak, şüpheli)"), [])
        assert _tail(out) == LABEL_SINGLE

    def test_canonical_unchanged(self):
        for lbl in (LABEL_OFFICIAL, LABEL_MULTI, LABEL_SINGLE):
            out = validate_sitrep(_label_line(lbl), [])
            assert _tail(out) == lbl

    def test_uninferrable_degrades_to_most_conservative(self):
        # Never invent verification: an unrecognisable label → single-source.
        out = validate_sitrep(_label_line("Teyit edilemedi bir şekilde"), [])
        assert _tail(out) == LABEL_SINGLE

    def test_confirmed_without_tier_stays_conservative(self):
        # "Onaylandı" with no Resmî/Çoklu keyword must not become OFFICIAL.
        out = validate_sitrep(_label_line("Onaylandı"), [])
        assert _tail(out) == LABEL_MULTI

    def test_source_remainder_preserved(self):
        line = (_HDR + "— Olay. Doğruluk Durumu: Onaylandı (Çoklu kaynak, ekstra)"
                " — Kaynak: reuters (http://r)")
        out = validate_sitrep(line, ["http://r"])
        assert LABEL_MULTI in out
        assert "— Kaynak: reuters" in out
        assert "ekstra" not in out

    def test_never_raises_on_bad_label(self):
        # The whole point: a stray label no longer fails the report.
        validate_sitrep(_label_line("garbage xyz"), [])

    def test_missing_header_still_raises(self):
        with pytest.raises(ValueError):
            validate_sitrep("no header at all", [])

    def test_unknown_url_still_masked(self):
        out = validate_sitrep(_HDR + "Bkz http://evil.example/x", [])
        assert "evil.example" not in out
        assert "[kaynak listede]" in out


class TestCitationPunctuation:
    """The URL allowlist pass used to strip the punctuation that followed a URL,
    which deleted the closing paren of every "Kaynak: Ad (url)" citation. The
    HTML renderer keys on that paren, so it silently dropped the source chips
    from every bullet of every SITREP for a week before anyone saw it."""

    def test_closing_paren_survives_the_allowlist_pass(self):
        line = (_HDR + "— Olay. Kaynak: Reuters (https://r.example/a),"
                " AP (https://ap.example/b).")
        out = validate_sitrep(line, ["https://r.example/a", "https://ap.example/b"])
        assert "Reuters (https://r.example/a)" in out
        assert "AP (https://ap.example/b)." in out

    def test_masked_url_keeps_its_punctuation_too(self):
        out = validate_sitrep(_HDR + "— Olay. Kaynak: X (https://nope.example/a).", [])
        assert "X ([kaynak listede])." in out

    def test_sources_reach_the_rendered_html(self):
        from src.services.sitrep_html import _render_bullet, _strip_md
        line = _HDR + (
            "• **2026-08-01** Olay — Doğruluk Durumu: Onaylandı (Çoklu kaynak)"
            " — Kaynak: Reuters (https://r.example/a), AP (https://ap.example/b)")
        out = validate_sitrep(line, ["https://r.example/a", "https://ap.example/b"])
        html = _render_bullet(_strip_md(out.splitlines()[-1]))
        assert "https://r.example/a" in html and "https://ap.example/b" in html
        assert "2026-08-01" in html

    def test_markdown_link_scaffolding_still_yields_sources(self):
        """Some models wrap the URL as "IRNA ([url](https://…))"; the renderer
        unwraps it rather than losing the attribution."""
        from src.services.sitrep_html import _render_bullet, _strip_md
        line = "• Olay — Kaynak: IRNA English ([url](https://irna.example/z))"
        html = _render_bullet(_strip_md(line))
        assert "https://irna.example/z" in html
        assert "IRNA English" in html

    def test_italicised_attribution_keeps_its_url(self):
        """Run #23 (UA, PL): the narrator italicised the whole attribution, so the
        closing "*" stayed glued to the URL, the allowlist check failed on a URL
        that was in the list, and the citation was blanked — closing paren and
        all ("The Moscow Times ([kaynak listede]")."""
        line = _HDR + "— Olay. *Kaynak: Moscow Times (https://mt.example/a)*"
        out = validate_sitrep(line, ["https://mt.example/a"])
        assert "Moscow Times (https://mt.example/a)*" in out

    def test_cosmetic_url_variants_are_repaired_not_blanked(self):
        out = validate_sitrep(_HDR + "— Olay. Kaynak: X (https://ex.example/a/).",
                              ["https://www.ex.example/a"])
        assert "X (https://www.ex.example/a)." in out

    def test_invented_url_is_still_blanked(self):
        out = validate_sitrep(_HDR + "— Olay. Kaynak: X (https://other.example/a).",
                              ["https://ex.example/a"])
        assert "[kaynak listede]" in out


class TestUncitedBullets:
    """2026-08-21 (GB): a bullet closed with "Kaynak: Yukarıda belirtilen
    kaynaklar." — the shape of a citation with none of the substance. The prompt
    forbids it, the model obeys almost always, and every guardrail here passed it
    through because the line contained no URL to check.
    """

    HEADER = "YÖNETİCİ ÖZETİ\nÖzet paragrafı.\n\n**OLAYLAR**\n"

    def test_sourceless_bullet_is_marked(self):
        text = self.HEADER + (
            "• **2026-08-20** Bir olay oldu. — Doğruluk Durumu: Onaylandı "
            "(Çoklu kaynak) — Kaynak: Yukarıda belirtilen kaynaklar.\n")
        out = validate_sitrep(text, ["https://reuters.com/a"])
        assert "Yukarıda belirtilen kaynaklar" not in out
        assert "Kaynak: belirtilmedi (bkz. rapor sonundaki künye)" in out
        assert "Onaylandı (Çoklu kaynak)" in out

    def test_cited_bullet_is_untouched(self):
        line = ("• **2026-08-20** Bir olay. — Doğruluk Durumu: Doğrulanmamış "
                "(Tek kaynak) — Kaynak: Reuters (https://reuters.com/a)\n")
        out = validate_sitrep(self.HEADER + line, ["https://reuters.com/a"])
        assert "Kaynak: Reuters (https://reuters.com/a)" in out
        assert "belirtilmedi" not in out

    def test_blanked_url_keeps_its_own_marker(self):
        """A citation the allowlist blanked already says so, and keeps the
        publisher name — which tells the reader more than this notice would."""
        line = ("• **2026-08-20** Bir olay. — Kaynak: Reuters "
                "(https://uydurma.example/x)\n")
        out = validate_sitrep(self.HEADER + line, ["https://reuters.com/a"])
        assert "[kaynak listede]" in out
        assert "belirtilmedi" not in out

    def test_prose_mentioning_sources_is_untouched(self):
        line = "Bu paragrafta kaynak kelimesi geçiyor ama bir madde değil.\n"
        out = validate_sitrep(self.HEADER + line, [])
        assert out.endswith(line.rstrip("\n"))


class TestCitationContract:
    """`cites_a_listed_url` is the check every narrator model is now held to.

    Measured 2026-09-04: minimax-m2.7 wrote every source as "Middle East Eye
    (https://middleeasteye.net)" — the publisher's domain instead of the article
    link it was handed — so all 108 citations across five SITREPs were blanked and
    the reports shipped with no working link. Six other models over the preceding
    21 days averaged 0.3 blanked per report, so the bar is set where it separates
    those two populations and nowhere tighter.
    """

    ALLOWED = ["https://middleeasteye.net/news/iran-strike-12345",
               "https://apnews.com/article/hormuz-9f2"]

    def test_a_bare_domain_is_not_a_citation(self):
        assert not cites_a_listed_url(
            "Kaynak: Middle East Eye (https://middleeasteye.net)", self.ALLOWED)

    def test_the_article_link_passes(self):
        assert cites_a_listed_url(
            "Kaynak: MEE (https://middleeasteye.net/news/iran-strike-12345)",
            self.ALLOWED)

    def test_one_survivor_is_enough(self):
        """Deliberately weak: this catches a model that does not do citations,
        not one that lost a link to a stray markdown character."""
        text = ("Kaynak: X (https://uydurma.example/a), "
                "Y (https://apnews.com/article/hormuz-9f2)")
        assert cites_a_listed_url(text, self.ALLOWED)

    def test_trailing_punctuation_does_not_fail_a_good_link(self):
        assert cites_a_listed_url(
            "*Kaynak: AP (https://apnews.com/article/hormuz-9f2).*", self.ALLOWED)

    def test_a_narrative_with_no_urls_at_all_fails(self):
        assert not cites_a_listed_url("Hiçbir kaynak verilmedi.", self.ALLOWED)

    def test_an_empty_allowlist_cannot_be_violated(self):
        """Nothing was fetched, so nothing can be cited — that is not the
        narrator's failure and must not cost the country its report."""
        assert cites_a_listed_url("Hiçbir kaynak verilmedi.", [])
