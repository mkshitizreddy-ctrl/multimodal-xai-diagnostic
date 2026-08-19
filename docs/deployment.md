# Deploying the Dashboard (Free)

This gets you a public, shareable link to the dashboard — worth doing for a
placement portfolio since a live demo is far more convincing than a
code-only repo.

**Note:** Hugging Face Spaces now requires a paid PRO plan to create Gradio
or Docker Spaces (which Streamlit apps run on) — only Static Spaces are
free, and those can't run Python. The path below is **fully free**: the app
runs on [Streamlit Community Cloud](https://streamlit.io/cloud), and the
trained checkpoint (too large for a normal git repo) is hosted on a free
Hugging Face Hub **model repo** — Hub storage is unaffected by the Spaces
compute pricing change.

## 1. Upload your checkpoint to a Hugging Face Hub model repo

This is separate from a Space — it's just free file storage.

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co) if needed.
2. Go to **Settings -> Access Tokens -> Create new token**, with **Write** permission.
3. Create a new **model** repo (not a Space) at
   [huggingface.co/new](https://huggingface.co/new) - e.g.
   `<your-username>/multimodal-xai-diagnostic-weights`. Public visibility.
4. Upload the checkpoint using the `huggingface_hub` Python client (already
   in `requirements.txt`):

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='checkpoints/vision_baseline/best_model.pth',
    path_in_repo='vision_baseline_best_model.pth',
    repo_id='<your-username>/multimodal-xai-diagnostic-weights',
    repo_type='model',
    token='<your-hf-write-token>',
)
"
```

## 2. Push your code to GitHub (already done)

Streamlit Community Cloud deploys directly from a GitHub repo - no
separate git remote or push needed, unlike the old Spaces workflow. Just
make sure your latest code is pushed to `main`.

## 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. **Repository:** `<your-username>/multimodal-xai-diagnostic`. **Branch:** `main`. **Main file path:** `dashboard/app.py`.
4. Before deploying, click **Advanced settings** and add a secret so the app knows where to download your checkpoint from:
   ```toml
   HF_MODEL_REPO_ID = "<your-username>/multimodal-xai-diagnostic-weights"
   ```
5. Click **Deploy**.

The app will install `requirements.txt`, then on first run `dashboard/app.py` automatically downloads the checkpoint from your Hugging Face model repo (no local file needed) and switches out of demo mode - no code changes required, this is already wired up in `load_vision_model()`.

## 4. Link it back

Once live, add the demo link to the top of this repo's `README.md`:

```markdown
[Live Demo](https://<your-username>-multimodal-xai-diagnostic.streamlit.app)
```

(Streamlit Community Cloud shows your app's exact URL once deployed - copy it from there.)

This is the single highest-impact addition you can make to the README for
a placement audience - a clickable, working demo beats another paragraph
of description every time.

## Notes

- Free tier CPU is enough for single-image inference - no GPU needed for the demo.
- The app sleeps after inactivity on the free tier and takes a few seconds to wake up on the next visit - normal, not a bug.
- If you also want the fusion model in the live demo, upload `checkpoints/fusion/best_model.pth` to the same Hub repo and extend `dashboard/app.py`'s `load_vision_model()` pattern for a fusion-model loader (not wired up by default, since the current UI only demos the vision-only model).
