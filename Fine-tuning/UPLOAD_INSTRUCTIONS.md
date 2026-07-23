# Add this package to the existing HighSS repository

1. Unzip this archive.
2. Copy the `training/` directory into the root of the existing HighSS clone.
3. Append the contents of `ADD_TO_ROOT_README.md` to the repository root
   `README.md`.
4. Do not upload checkpoints, `ccd.pkl`, processed structures, MSAs, logs or
   validation-ID files to GitHub.
5. Review the staged files before committing:

```bash
git status
grep -RniE '(api[_-]?key|secret|password|/data/|/mnt/|127\.0\.0\.1|CUDA_VISIBLE_DEVICES)' training
python -m compileall -q training
```

6. Commit and push:

```bash
git add training README.md
git commit -m "Add HighSS fine-tuning scripts and reference configurations"
git push
```
