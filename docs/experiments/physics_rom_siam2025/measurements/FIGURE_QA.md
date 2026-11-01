# Figure contract and QA

Claim: waveform accuracy, inversion-gradient accuracy, and online cost must be assessed jointly.

Accuracy-cost: quantitative grid; every completed rank is retained; background and held-out 2% perturbation are shown. Truth-state gradients are zero, so relative gradient errors are undefined; truth rows remain in source_data.csv. No random seeds are averaged. Timing marks are medians, with min–max ranges of three warmed, alternating-order technical repeats. No significance test. 1% waveform and 5% gradient guides are declared engineering thresholds. S1, nt=1000 only; this is not the 49-shot baseline training protocol.

Wavefield plate: largest completed joint rank (or primal fallback), background model, four equally spaced recorded frames. Whole physical grid; PML excluded for both. Common symmetric linear pressure scale for both methods and all times; separate symmetric difference scale. No smoothing, normalization, gamma, stitching, or local editing. Raw arrays and exact frame indices are saved.

Python matplotlib; 183 mm width; editable PDF/SVG and 600-dpi PNG. Automated source and PDF glyph audits plus panel inspection are recorded alongside the report.

Final inspection: all three accuracy/cost panels and twelve wavefield panels were reviewed. Shared pressure and separate difference scales are correct; long row labels were shortened to avoid overlap. PDF glyph audits pass at 5.6 pt and 7 pt. Static preflight: 19 PASS, 0 FAIL; PNG-only raster warning accepted because these are report previews with editable PDF/SVG, not TIFF submission files. No omitted completed rank or state in source_data.csv.
