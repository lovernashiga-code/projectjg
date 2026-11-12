# Building the final deck

Every number on the slides is read from `results/` (metrics.json, targets.json, final_numbers.json, open_set.json,
fei/, video/), so after re-running the experiment, rebuild the deck instead of editing numbers by hand:

```bash
cd slides/build
npm install                      # pptxgenjs
node build.js ../.. ../Partial_Face_Recognition_Final.pptx   # run twice: the 2nd run fills the page numbers on slide 2
node build.js ../.. ../Partial_Face_Recognition_Final.pptx
python fix_fonts.py ../Partial_Face_Recognition_Final.pptx   # Thai text -> Leelawadee UI (not Angsana/Cordia)
```

`build.js` ends by calling the pptx skill's `apply_theme.js`; outside Claude, delete those three lines at the end
(the theme colours are only cosmetic).
