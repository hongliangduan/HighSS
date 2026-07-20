# HighSS checkpoint files

The repository intentionally excludes model binaries.

Create this directory and place the following external files here before
running the bundled examples:

- `com-FT.ckpt`
- `ccd.pkl`

Expected layout:

```text
HighSS-main/
└── checkpoints/
    ├── README.md
    ├── com-FT.ckpt
    └── ccd.pkl
```



The model and CCD files are excluded by the root `.gitignore` and should not
be committed to GitHub.
