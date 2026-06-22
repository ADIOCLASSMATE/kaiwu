# Uzi Poster Asset Manifest

This folder contains reusable high-resolution assets for manually composing the poster.

## AI Components

- `ai_components_png/*_original.png`: original image-generation components.
- `ai_components_png/*_highres.png`: upscaled high-resolution versions for PPT placement.

## Labeled Figures

- `labeled_figures_png/*_600dpi.png`: high-resolution PNG exports of the paper figures.
- `vector_pdf/*.pdf`: PDF figure sources. Data-only figures are vector; mixed figures include embedded AI raster components plus vector labels.
- `vector_svg/*.svg`: SVG exports where conversion succeeded.

## Icons

- `icons_png_transparent/*.png`: 1024px transparent standalone icons.
- `icons_png_transparent/icon_sheet_4096_transparent.png`: transparent icon sheet.

Recommended poster workflow: use the AI components as visual backgrounds, add the labeled PNG/PDF figures when you want the exact academic diagram, and use the transparent icons for extra callouts.
