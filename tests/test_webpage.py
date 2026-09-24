from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_landing_page_has_prediction_flow_and_design_tokens():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "styles.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "共享单车需求估计器" in html
    assert 'id="prediction-form"' in html
    assert 'id="prediction-result"' in html
    assert "--color-blurple: #5865f2" in css
    assert "fetch(\"/predict\"" in js


def test_invalid_input_has_accessible_error_target():
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="form-error"' in html
    assert 'role="alert"' in html
    assert 'aria-live="polite"' in html


def test_motion_hooks_and_reduced_motion_support_exist():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "styles.css").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'data-reveal' in html
    assert 'class="chart-line"' in html
    assert "@keyframes chart-draw" in css
    assert "@keyframes float-gentle" in css
    assert "IntersectionObserver" in js
    assert "animateNumber" in js
    assert "prefers-reduced-motion: reduce" in css
