"""PolyRoute dashboard (Flask).

Run (from src/):
    python -m dashboard.app                          # laptop: http://127.0.0.1:8501
    POLYROUTE_HOST=0.0.0.0 python -m dashboard.app   # inside the LinuxONE container
"""

import os

from flask import Flask, render_template, request

from . import integrations, pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit


def page(view=None, error=None, form=None):
    return render_template(
        "index.html", view=view, error=error, form=form or {},
        samples=integrations.list_sample_files(), defaults=pipeline.DEFAULT_LIMITS,
        machine=pipeline.machine(),
    )


@app.get("/")
def index():
    return page()


@app.post("/analyse")
def analyse():
    try:
        view = pipeline.run(request.form, request.files)
        return page(view=view, form=request.form)
    except (ValueError, NotImplementedError) as e:
        return page(error=str(e), form=request.form), 400


if __name__ == "__main__":
    app.run(host=os.environ.get("POLYROUTE_HOST", "127.0.0.1"),
            port=int(os.environ.get("POLYROUTE_PORT", "8501")), debug=False)
