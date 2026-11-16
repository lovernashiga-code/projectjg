# Building the final deck

Every number on the slides is read from `results/` (metrics.json, targets.json, final_numbers.json, open_set.json,
fei/, video/), so after re-running the experiment, rebuild the deck instead of editing numbers by hand:

The group-member slides (section 08) also need, from `src/`, after step 1-5 with `--dataset lfw+custom`:

```bash
python video.py ../VDO/test/<clip> --model ../models/lfw+custom_M4.joblib --out ../results/video/members_M4   # each test clip (M2 too)
python members_eval.py          # scores the clips against VDO/test/truth.json -> results/lfw+custom/*.json, figures/demo/
```

Then:

```bash
cd slides/build
npm install                      # pptxgenjs
node build.js ../.. ../Partial_Face_Recognition_Final.pptx   # run twice: the 2nd run fills the page numbers on slide 2
node build.js ../.. ../Partial_Face_Recognition_Final.pptx
python fix_fonts.py ../Partial_Face_Recognition_Final.pptx   # Thai text -> Leelawadee UI (not Angsana/Cordia)
```

`build.js` ends by calling the pptx skill's `apply_theme.js`; outside Claude, delete those three lines at the end
(the theme colours are only cosmetic).

## Presentation deck (5-minute talk)

`build_short.js` writes the deck that is presented: 12 main slides (with the speaker and a script in each slide's notes)
plus 6 backup slides for questions. It uses the font TH SarabunPSK, so install that font on the presenting computer:

```bash
node build_short.js ../.. ../Partial_Face_Recognition_Final.pptx
python fix_fonts.py ../Partial_Face_Recognition_Final.pptx "TH SarabunPSK"
```

`build.js` writes the long 43-slide version (`Partial_Face_Recognition_Final_full.pptx`, Leelawadee UI) kept for reference.
