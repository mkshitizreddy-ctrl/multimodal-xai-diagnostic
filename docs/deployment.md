# Deploying the Dashboard to Hugging Face Spaces

This gets you a public, shareable link to the dashboard — worth doing for a
placement portfolio since a live demo is far more convincing than a
code-only repo.

## 1. Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pick a name (e.g. `multimodal-xai-diagnostic`).
3. **SDK:** choose **Streamlit**.
4. **Hardware:** the free CPU tier is fine for the demo-mode fallback (no
   trained checkpoint); if you want a real trained model to power the demo,
   you'll need to either upload a checkpoint (see step 3) or use paid GPU
   hardware to train inside the Space, which is usually not worth it —
   train locally/on Colab and upload the checkpoint instead.
5. **Visibility:** Public (so it's linkable from your resume/README).

## 2. Push this repo to the Space

Hugging Face Spaces are git repos. Add it as a second remote alongside GitHub:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/multimodal-xai-diagnostic
git push space main
```

You'll be prompted for credentials — use a Hugging Face access token
(Settings → Access Tokens) as the password.

## 3. (Optional but recommended) Upload a trained checkpoint

The dashboard works out of the box in demo mode without one, but a real
checkpoint makes the live demo far more impressive.

- Checkpoints are usually too large for a normal git push — use
  [Git LFS](https://git-lfs.com/) or the Hugging Face `huggingface_hub`
  Python client to upload `checkpoints/vision_baseline/best_model.pth`
  directly to the Space's file storage.
- Once uploaded, the dashboard will automatically detect it at
  `checkpoints/vision_baseline/best_model.pth` and switch out of demo mode
  — no code changes needed (see `dashboard/app.py`'s `load_vision_model()`).

## 4. Space configuration file

Hugging Face Spaces need a small YAML header at the top of the Space's own
`README.md` (this is separate from this repo's main README — Spaces reads
it from the file at the Space's root):

```yaml
---
title: Multimodal XAI Diagnostic
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: dashboard/app.py
pinned: false
---
```

Add this block to the very top of the version of `README.md` that lives in
the Space (you can keep this repo's own README as-is and only add the
header when you push, or maintain a small `README_space.md` and copy it in
at deploy time).

## 5. Link it back

Once live, add the demo link to the top of this repo's `README.md`:

```markdown
🔗 **[Live Demo](https://huggingface.co/spaces/<your-username>/multimodal-xai-diagnostic)**
```

This is the single highest-impact addition you can make to the README for
a placement audience — a clickable, working demo beats another paragraph
of description every time.
