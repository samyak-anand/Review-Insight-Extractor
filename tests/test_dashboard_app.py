from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_navigation_renders_every_view_without_exceptions() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception

    for page in [
        "Explore findings",
        "Aspect matrix & priorities",
        "Evaluation & errors",
        "Data status",
        "Overview",
    ]:
        app.radio[0].set_value(page)
        app.run()
        assert not app.exception, f"Dashboard view {page!r} raised an exception"