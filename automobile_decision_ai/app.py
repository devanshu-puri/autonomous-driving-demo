from __future__ import annotations

import base64
import sys
from pathlib import Path

from flask import Flask, render_template_string, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automobile_decision_ai.dataset import ProjectPaths, discover_records
from automobile_decision_ai.pipeline import analyze_sample, ensure_artifacts
from automobile_decision_ai.visualization import array_to_png_bytes, overlay_mask

app = Flask(__name__)


def _to_data_uri(image_array) -> str:
    encoded = base64.b64encode(array_to_png_bytes(image_array)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    ensure_artifacts(ROOT)
    paths = ProjectPaths.from_root(ROOT)
    records = discover_records(paths)
    test_records = [record for record in records if record.split == "test"]
    sample_id = request.args.get("sample", test_records[0].sample_id)
    result = analyze_sample(ROOT, sample_id)

    original_uri = _to_data_uri(result["image"])
    gt_uri = _to_data_uri(overlay_mask(result["image"], result["gt_mask_rgb"]))
    pred_uri = _to_data_uri(overlay_mask(result["image"], result["pred_mask_rgb"]))

    template = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Automobile Driving Decision Demo</title>
      <style>
        body { font-family: Segoe UI, sans-serif; margin: 0; background: #f5f7fa; color: #1c2430; }
        .wrap { max-width: 1240px; margin: 0 auto; padding: 24px; }
        .hero { background: linear-gradient(135deg, #103c5d, #2f6f52); color: white; padding: 24px; border-radius: 18px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 20px; }
        .card { background: white; border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
        img { width: 100%; border-radius: 12px; display: block; }
        .decision { font-size: 28px; font-weight: 700; margin: 8px 0; }
        .muted { color: #5d6875; }
        select, button { font-size: 16px; padding: 10px 12px; border-radius: 10px; border: 1px solid #cbd5df; }
        ul { margin-top: 8px; }
        .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
        .pill { background: #eef3f8; padding: 10px 12px; border-radius: 12px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="hero">
          <h1>Automobile Driving Decision Demo</h1>
          <p>This demo uses an open autonomous-driving dataset, predicts scene understanding, and explains what the AI thinks the car should do.</p>
          <form method="get">
            <label for="sample">Choose a sample:</label>
            <select name="sample" id="sample">
              {% for record in records %}
              <option value="{{ record.sample_id }}" {% if record.sample_id == selected %}selected{% endif %}>
                {{ record.sample_id }}
              </option>
              {% endfor %}
            </select>
            <button type="submit">Analyze</button>
          </form>
        </div>

        <div class="card" style="margin-top: 20px;">
          <div class="muted">Sample: {{ result.sample_id }} | Split: {{ result.split }}</div>
          <div class="decision">{{ result.action }}</div>
          <div class="muted">Confidence: {{ "%.1f"|format(result.confidence * 100) }}%</div>
          <ul>
            {% for line in result.explanation %}
            <li>{{ line }}</li>
            {% endfor %}
          </ul>
          <div class="metrics">
            {% for key, value in result.metrics.items() %}
            <div class="pill">{{ key }}: {{ "%.3f"|format(value) }}</div>
            {% endfor %}
          </div>
        </div>

        <div class="grid">
          <div class="card">
            <h3>Original Image</h3>
            <img src="{{ original_uri }}" alt="Original image">
          </div>
          <div class="card">
            <h3>Ground Truth Scene</h3>
            <img src="{{ gt_uri }}" alt="Ground truth scene">
          </div>
          <div class="card">
            <h3>AI Predicted Scene</h3>
            <img src="{{ pred_uri }}" alt="Predicted scene">
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    return render_template_string(
        template,
        records=test_records,
        selected=sample_id,
        result=result,
        original_uri=original_uri,
        gt_uri=gt_uri,
        pred_uri=pred_uri,
    )


if __name__ == "__main__":
    ensure_artifacts(ROOT)
    app.run(debug=False, host="127.0.0.1", port=5000)
