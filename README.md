# imged

A small, layer-based image editor for Linux, written in Rust. It aims to sit
between MS Paint and GIMP: quick to understand, but with real layers,
per-layer color depth (8-bit, 16-bit, 32-bit float), transparency
everywhere and HDR support.

## Build and run

Requires Rust 1.95 or newer.

```sh
cargo run --release -p imged -- [image] [more images as layers…]
```

* Rendering goes through wgpu (Vulkan, or OpenGL as a fallback). To force
  OpenGL, set `WGPU_BACKEND=gl`. The canvas is composited on the GPU (see
  *GPU rendering* below); `IMGED_RENDERER=cpu` or **View → GPU rendering**
  switches to the CPU renderer.
* File dialogs use the XDG desktop portal. If no portal is running, use
  **File → Open from path…**, drag files onto the window, or pass them on the
  command line. The Export dialog has its own path field.
* Every format the `image` crate supports can be opened and saved (PNG, JPEG,
  GIF, WebP, TIFF, BMP, ICO, TGA, QOI, PNM/PAM, farbfeld, OpenEXR, Radiance
  HDR, AVIF), plus PFM, HEIC/HEIF and JPEG XL. Opening AVIF needs the system
  `dav1d` library: build with `--features avif-decode`. Saving AVIF works
  without it.
* **HEIC/HEIF** goes through the system **libheif** (default `heif` feature):
  building needs `libheif-dev` (Debian/Ubuntu), `libheif-devel` (Fedora) or
  `libheif` (Arch). Opening needs libheif's HEVC decoder (libde265 plugin) and
  saving its encoder (x265 plugin), which distributions ship with libheif or
  as `libheif-plugin-*` packages. Build with `--no-default-features
  --features jxl` to leave it out.
* **JPEG XL** (default `jxl` feature) is pure Rust: jxl-oxide decodes and
  zune-jpegxl encodes. Saving is lossless, at 16 bits for opaque images and 8
  bits when there is transparency (the encoder can't store 16-bit alpha
  correctly).
* Projects are saved as layered PSD files (see *Projects* below).
* **File → Import raw…** opens headerless pixel data: you choose the
  channel layout (gray, RGB, BGR, ARGB, HSL, HSV…, with or without alpha),
  sample type (8/16/32-bit integer, 32/64-bit float, either endianness),
  size, header bytes, row padding, planar storage and row order. A live
  preview and the sizes that fit the file exactly are shown.
* **Stitching photos** (the Stitch tool) needs the system **OpenCV 4.x**
  development files and **libclang**: build with `--features stitch`, e.g.
  `cargo run --release -p imged --features stitch`. Packages:
  `libopencv-dev clang libclang-dev` (Debian/Ubuntu), `opencv-devel clang-devel`
  (Fedora), `opencv clang` (Arch). Without it the tool is shown but can't run.
* Run the core tests with `cargo test -p imged-core` (add `--features stitch`
  to include the stitching tests).

## The window

| Area | Contents |
|---|---|
| Left | Tools, options for the current tool (with snapping, paint mode, anti-aliasing and the layer options where they apply), the **Transform & filters** section (**Apply to** Layer / Image / Selection, flips, 90° turns and the color filters), and Open / Open as layer / Save project / Export / Paste image / Copy as PNG buttons in the bottom corner |
| Center | Canvas. The mouse wheel zooms; middle-drag or Space+drag pans. A checkerboard shows transparency. |
| Right | Color panel (see *Colors*), layers and layer properties (including **Canvas size**), then Undo/Redo with the names of the previous and next operations, and zoom (−, a percentage you can type, +, Fit, 1:1) |
| Bottom | Messages, cursor position, the pixel's R G B A values at the image's depth (with the maximum, e.g. `/ 255`, `/ 65535`; previews included), the size of the shape or box being drawn, HDR exposure and sRGB view toggle, zoom |

### Tools

| Key | Tool | Notes |
|---|---|---|
| V | Move layer | Moves the active layer or group. With **Snap to edges** on, its edges snap to other layers' visible edges and to edges visible in the image (Alt moves freely). Arrow keys nudge (Shift = 10 px); a run of nudges is one undo step. |
| B | Brush | Left click paints the primary color, right click the secondary. Shift+click draws a line from the last stroke. `[` and `]` change the size. |
| N | Pen | Square, hard, non-antialiased tip that sets pixels to the exact color, alpha included. |
| I | Color picker | Hovering shows the color under the cursor in the color panel's controls; left click sets the primary, right click the secondary (and makes it the edited color). Can average a square (3×3 up to 101×101). Samples all layers or only the active one. |
| U | Shapes | Line, rectangle, square, ellipse, circle, arrow (picked by icon). Outline, filled, or outline with a secondary-color fill. Options: rounded corners/ends (sharp by default), pixel-aligned (on by default), draw from center. Shift constrains. Snaps to edges. Each shape is its own **shape layer** and stays editable (see *Shape and text layers*). |
| G | Paint bucket | Tolerance, contiguous, diagonal, sample all layers. Hovering previews the area that would be filled, tinted with the paint color. A fill that starts a new layer (the new-layer options) detects the area from the visible image. |
| T | Text | Choose a system font, size, bold, italic, rotation and line spacing. Click to place the text, drag it to move, press Enter or Apply to commit. Snaps to edges. Each text is its own **text layer** and stays editable. |
| M | Box select | Selects a box. Snaps to edges. See *Selections*. |
| W | Magic wand | Selects the clicked color region. Hovering previews it. |
| O | Outline select | Finds the solid-color outline (or flat background) around the click and selects everything inside it. **Content on a solid background** instead selects the smallest rectangle with only the solid color around the click outside it (e.g. the bounding box of text on a plain background); the outline tolerance decides which colors count as that solid color. Hovering previews the result. |
| E | Resize | Drag a corner to resize (the opposite corner stays put; obeys **Keep ratio**), drag inside to move, or type a size in pixels or percent. Untick **Move all corners** to move each corner on its own (free warp). Live preview; Enter or Apply commits, Esc resets. |
| R | Rotate | Drag on the canvas or set the angle. With snapping on, dragging sticks to multiples of 45° (Alt turns freely); Shift snaps to 15°. The **anchor** drawn on the canvas is the point it turns around: drag it anywhere (with snapping on it snaps to edges, to the middle of the canvas/selection and to the corners, edge middles and center of what is rotated); **Center anchor** puts it back. Live preview; the result is trimmed. Enter or Apply commits. |
| P | Tilt | Drag an edge handle (or pick the edge with an arrow) to tilt in 3D. It hinges on the opposite edge. Drag inward to tilt away, outward to tilt toward you. |
| H | Shear | Drag an edge handle along the edge to turn the area into a parallelogram. |
| J | Stitch photos | Click a letter on one photo, then the same letter on another: the second is aligned onto the first and both are merged into a new layer. See *Stitching photos*. |

Resize, Rotate, Tilt and Shear, the flip and 90° buttons, and the color
filters act on the **Apply to** choice: the active
layer (or group), the whole image (every layer; resizing and rotating the
image also changes the canvas), or the **Selection** (only the selected
pixels of the active layer, which are transformed and laid over the rest;
the selection follows them). With nothing selected, **Selection** acts like
**Layer**.

Number fields (sizes, zoom, color values…) can be dragged left and right to
change them. While dragging, the mouse pointer is hidden and held inside the
window, so you can drag as far as you like; it reappears where you started
when you let go.

Other shortcuts: X swaps colors, D resets to black and white, S toggles
snapping, Ctrl+Z / Ctrl+Shift+Z undo and redo, Ctrl+S saves the project
(Ctrl+Shift+S: save as), Ctrl+E exports, Ctrl+V pastes an image as a layer,
Ctrl+Shift+C copies the canvas to the clipboard as PNG, Ctrl+K opens
the convolution dialog, Ctrl+I inverts, Ctrl+Shift+L runs auto levels,
Ctrl+G groups and Ctrl+Shift+G ungroups, Ctrl+A selects all, Ctrl+Shift+A
deselects and Ctrl+Shift+I inverts the selection. F1 lists every shortcut.

## Concepts

**Transparency everywhere.** The default paint mode is **Replace**: painted
pixels take the paint color including its alpha, so painting with a
transparent color erases. Switch to **Blend over** for normal alpha
compositing. **Anti-aliasing** turns smooth edges on or off; turn it off for
pixel-exact, hard-edged painting.

**Per-layer color depth.** Each layer stores `u8`, `u16` or `f32` pixels.
Images open at their native depth, and PFM, EXR and Radiance HDR open as
`f32` with HDR values kept. You can convert a layer from the layer panel.
Merging uses the deepest depth involved. **Image → Depth for new layers**
sets the default. Compositing and filters always run in `f32`.

**HDR viewing.** The EV control in the status bar scales the display, and the
sRGB toggle encodes linear data for viewing. Both are view-only and turn on
automatically for linear formats. The status bar shows float pixel values.

**Colors.** The primary and secondary swatches are at the top left (click
one to edit it; the arrow icon between them swaps them, X). Next to them
are the **H**, **S** and **V** bars, below them the **R**, **G**, **B** and
**A** bars, each with its number (RGB can go above 1 for HDR), and the hue
/ saturation wheel. **Hex** and **Oklab** (perceptual L, a, b) codes can be
typed. **Recent** and **Palette** show one row; the arrow shows them all.
The palette's **+** adds the edited color, Ctrl+click replaces a swatch,
Shift+click removes it, and the palette is remembered. While the color
picker hovers the canvas, every control shows the color under the cursor.

**Shape and text layers.** Shapes and text are not turned into pixels: each
one is a layer that remembers what it is (the layer panel says "shape ·
editable" or "text · editable"). With the shape tool, the active shape layer
shows handles: drag them (or the shape itself) to change it, and its options
(kind, line width, style, rounded corners, line and fill colors) change it
directly. Ctrl+drag, or dragging away from the shape, draws a new one. With
the text tool, the active text layer is edited as you type or change the
font, size, rotation or color, and dragging the text moves it. A run of
edits to the same layer is one undo step, and moving the layer with the
Move tool keeps it editable. Pixel operations on it (filters, transforms,
cutting a selection, erasing) turn it into plain pixels, and undo brings
the shape or text back. The brush, pen and bucket paint on a new layer
above a shape or text layer instead. **Rasterize** (layer panel, Layer
menu, or the tool options) turns it into pixels on purpose. Projects keep
them editable: other programs see their rendered pixels.

**New layers while drawing.** The brush, pen and bucket tools have
**New layer when the color changes** and **New layer when the tool
changes**: when on, drawing with a different color (or tool) than the last
drawing starts a new layer first, and the first drawing always does.

**Remembered settings.** Checkbox states (in the tool options, the
selection options, the convolution dialog…), the palette and the
ImageMagick command are saved in `~/.local/share/imged/settings.json`
(`$XDG_DATA_HOME/imged/` when set) and loaded at start.

**Selections.** The selection tools share two checkboxes:
* **Move to new layer** (on by default): the selected pixels move to a new
  layer above the source, and the editor switches to the Move tool
  (optional). If the active layer has nothing at the click or box, the
  topmost visible layer that does is used.
* With it off, the area stays **selected**, shown with marching ants, for
  filters and transforms with **Apply to: Selection**. The options then also
  offer Deselect, Invert, Cut / Copy to layer and Crop (to the selection);
  the Edit and Image menus have the same commands. Clicking without
  dragging with box select deselects.
* **Add to previous selection**: new areas are added to the selection; when
  moving to a layer, the new pixels join the layer the previous selection
  made, so several regions can be collected into one layer.

Selection changes are part of the undo history.

**Filter layers.** The funnel icon under a layer's eye (or **Layer → Toggle
filter layer**) turns the layer into a filter layer: its pixels are no
longer drawn, and the color filters applied to it change everything below
it instead, without touching those pixels. Its painted area (alpha) is the
mask of where the filters apply; an empty layer applies them everywhere, so
paint or erase on it to limit them. While a filter layer is active, the
color filters section shows and edits its settings live (Invert, Auto
levels and Switch become toggles; one undo step per change; convolution
isn't available). Its opacity sets how strongly it applies. The layer list
shows what it does ("filter: invert, exposure +1.0"). **Merge down** bakes
the filter into the layer below, and clicking the funnel again shows the
pixels as a normal layer.

**Growing layers.** Painting outside a layer's area grows the layer. The
brush, pen, shape, bucket and text tools have a **Grow layer when painting
outside it** checkbox (one shared setting); turn it off to clip painting to
the layer. Rotating, warping, tilting and shearing trim the transparent
edges of the result automatically.

**Opening several images.** Select several files in Open (or pass/drop
several): each becomes a layer above a transparent background. **Tile** in the
layer panel arranges the layers without overlaps, in rows forming a roughly
square image, and fits the canvas to them. A bottom layer of one uniform
color is treated as the background and left out; any other bottom layer is
tiled too. With layers ticked, only those (and the active one) are tiled,
starting from their top-left corner, and the canvas only grows if needed.

**Stitching photos.** For photos of a whiteboard taken in parts, from
different angles (camera turned up to about 30°). Open the photos as layers
and **Tile** them (or hide the top one) so you can see both, pick the Stitch
tool (J), click a letter or symbol on the photo that should stay in place,
then the same letter on the other photo (right click takes a click back).
Starting from that pair, imged searches the rotation and scale between the
two clicks, then matches the letters around them, ring by ring outward, so
repeated letters and empty board don't confuse it. The matched points are
shown (green on the first photo, blue on the second) with the median error;
**Stitch** (Enter) merges. Repeat with the result and the next photo to
stitch more than two.

* The alignment is a perspective transform plus a smooth correction for lens
  distortion and boards that aren't perfectly flat, refined densely where the
  photos overlap so strokes don't double.
* Lighting is evened out across the overlap, so there is no seam.
* Where the photos **disagree** (something was written or erased between the
  shots), each region comes from the photo that has **ink** there, or, with
  **Prefer upper layer where they differ**, from the photo higher in the
  layer list.
* **Projection** mode is for a document projected onto the board and scrolled
  between the photos: click inside the projected (brightest) area. Only that
  area is aligned and kept, so the board and wall, which did not move with
  the document, are left out. It is outlined in yellow after aligning.
* The result is a new layer, **Stitched**, above the photos. With **Keep the
  photos as hidden layers** (default) the photos stay, hidden, with the
  second one warped into place for touch-ups; otherwise they are removed. The
  canvas is fitted to the visible layers. It is one undo step.
* The same code runs from the command line:
  `cargo run --release -p imged-core --features stitch --example stitch --
  fixed.heic moving.heic FX FY MX MY out.png [--prefer-upper] [--projection]`.

**ImageMagick.** **Layer → Edit with ImageMagick…** (or **Image → …** for
the whole image) saves the active layer and the ticked ones (or the
flattened image) in a temporary folder in `/tmp` and opens a window with the
command line, pre-filled as `magick <in> <out>`: add the operations in
between (`-blur 0x3`, `-resize 50%`, …), choose Layer or Image, and press
Run. Several layers are written as `in-0.png`, `in-1.png`… from the top
down (the window lists which is which), so commands like `-append`,
`-compose … -composite` or `-ift` can combine them. Operations that make
several images (`-fft`, `-separate`, `-crop 2x2@`…) give `out-0.png`,
`out-1.png`…, and each becomes a layer. With as many outputs as inputs, each
output replaces its layer (keeping its position); otherwise the outputs
replace the input layers where the topmost one was (for the image, they
become its layers). It is one undo step, and the temporary files are
deleted. The
**Program** field (remembered) can be `magick`, `convert` (ImageMagick 6),
`gm convert` (GraphicsMagick) or a full path; the default is whichever is
installed. 8/16-bit layers go through PNG, float ones through TIFF.

**Clipboard.** **Paste image** (Ctrl+V) reads whatever image type is on the
clipboard (PNG, JPEG, BMP, WebP, TIFF…) and adds it as a layer, or opens it as
a new image if the window still holds an untouched new canvas. Copied files
or file paths are opened as layers.

**Layers.** You can toggle visibility, drag a layer's thumbnail or name to
reorder, merge down, merge ticked layers, duplicate, rename (double-click),
set opacity,
**Fit to canvas** (resize the layer's area to the canvas), **Fit the canvas**
(resize the canvas around all layers), and trim transparent edges. Tick
layers with their checkboxes and press **Group**: they then show as a
single entry with a blue name and move, filter, transform, show and hide as
one layer, but keep their own pixels. **Ungroup** splits them again.
**Split channels** turns the active layer into four grayscale layers (R, G,
B, A). With 3 or 4 layers ticked (the active one counts) the button becomes
**Merge as RGB** / **Merge as RGBA** and combines them again; layer names
ending in R/G/B/A are matched to channels, otherwise the order is top to
bottom.

**History.** There is one global undo history. Pixel edits are stored as
compact per-layer patches and structural changes (layers added, deleted,
reordered or merged, canvas-wide transforms) as snapshots, but they share
one timeline: Ctrl+Z always undoes the newest change wherever it happened.
The Undo and Redo buttons at the bottom right show what they will undo or
redo.

**Projects.** **Save project** (Ctrl+S) writes a layered Photoshop `.psd`
file that Photoshop, GIMP, Krita, Photopea and ImageMagick can open. It keeps
layer names, positions, visibility and opacity, and groups become PSD
groups. Filter layers are stored as hidden pixel layers (their mask) with
their settings in the file's XMP metadata, so imged restores them exactly
while other programs show the image without them. Photoshop adjustment
layers in files from other programs (brightness/contrast, hue/saturation,
levels, exposure, invert) open as filter layers, approximately. Opening a
`.psd` restores everything (nested groups are
flattened into one level; masks, blend modes and effects in files from other
programs are not applied). PSD has one bit depth per file, so a project is
saved at its deepest layer's depth (8-bit, 16-bit or 32-bit float), and on
reopening every layer has that depth. Open projects are autosaved every
minute when they have changed, to a folder with a random name that is
different for every running copy of imged, such as
`/tmp/imged-autosave-3f9c1a7e/` (File menu shows the exact one). The title bar shows
`*` while there are unsaved changes.

**Snapping.** **Snap to edges** (S) is in the options of the tools that use
it. Box selection, shapes, text placement and the rotation anchor snap
their x and y independently to the nearest color boundary within about 10
screen pixels, and to the middle of the canvas's and of the selection's
width and height. Snapping to a middle draws a dashed cyan guide and the
status bar says which ("snapped to 1/2 canvas width", "canvas center",
"selection center"). Moving a layer snaps its edges to other layers' edges and
to edges in the visible image (never to edges of layers that are covered). Boundaries lie between pixels, so a snapped box encloses an
object exactly. **Edge strength** sets the minimum color difference that
counts as an edge.

**Color filters.** Invert, auto levels (rescale so the darkest pixel becomes
0% and the brightest 100%), brightness, contrast, gamma, hue and saturation
(with live preview), exposure (in EV stops), a channel switch (e.g. R ↔ G,
or any channel with alpha), and convolution. The convolution kernel is a text box of
comma-separated rows that you can paste into; fractions like `1/16` work and
the numbers are re-aligned in columns when you leave the box. Its presets
follow the selected size (3×3 to 9×9): box blur, Gaussian blur, motion blur,
**low pass** (a windowed sinc cutting off at half the Nyquist frequency) and
**high pass** (the image minus its low pass, around mid-gray, so flat areas
turn gray and details stand out), plus sharpen, edge detect, emboss and
Sobel. **Alpha** goes
from -100% to +100% and is added to every pixel's opacity: +100% removes
transparency (everything becomes opaque), -100% makes everything
transparent. Filters follow the Transform
section's **Apply to** choice: the active layer (or group), every layer of
the image, or only the selected pixels (blended by the selection's
coverage).

**Fonts.** Large font collections such as Noto install dozens of
per-script families ("Noto Sans Thai", "Noto Kufi Arabic"…). The font menu
hides families that look like such variants, judged by their names, and
shows a checkbox to list them all. Searching always looks through every
family.

**Number inputs.** Typed values are applied when you press Enter or leave the
field, so typing "12" in a field whose minimum is 3 works.

## GPU rendering

The canvas is drawn by `GpuRenderer` (`renderer_gpu.rs`, `gpu.wgsl`), on the
same wgpu device as the window:

* Every visible layer (or its live preview) is kept in a GPU texture, so an
  edit uploads only the changed rectangle. 8-bit layers are `Rgba8Unorm`;
  16-bit and float layers are `Rgba32Float`, so HDR values are kept.
* A shader draws the layers bottom to top into a premultiplied 16-bit float
  canvas with "over" blending. A filter layer reads what is below and writes
  the filtered result into a second canvas texture, so filter layers (and
  their masks and opacity) work exactly as on the CPU.
* A display pass applies the view exposure and sRGB encoding into an 8-bit
  texture with mipmaps: smooth when zoomed out, crisp pixels when zoomed in.

The CPU compositor stays the reference: a test renders a document with
8/16-bit and float layers, offsets, opacity and filter layers on both and
checks they match within 1–3 levels out of 255. **View → GPU rendering**
(remembered) switches back to the CPU renderer, whose 512 px tiles are
re-composited where the image changed.

## GPU filters

Convolution and resizing (the Resize tool, Image → Resize, and the scaling
done by transforms) run as compute shaders on the GPU (`gpu_filters.rs`,
`gpu_filters.wgsl`). These are the operations that do enough work per pixel
to be worth sending the pixels to the graphics card and back: a 9×9 kernel
is 81 multiply-adds per channel per pixel. Pixels are sent in their own
depth (8-bit images as 4 bytes per pixel), and the results match the CPU's
within one 8-bit level (a test checks every preset, edge mode and resize
filter at all three depths).

The core library stays GPU-free: `imged_core::accel` is a small hook that
the GUI registers the GPU implementation with. Every operation keeps its CPU
code (`convolve_cpu`, `resize_cpu`), used when there is no GPU, when the
image is larger than the device's buffer limits, and by a future CLI.
**View → GPU filters** (remembered) switches it off to compare.

Per-pixel color filters, flips, rotations and warps stay on the CPU. They do
so little work per pixel that copying the image to the GPU and back would
take longer than running them on all CPU cores.

## Architecture

```
crates/
  imged-core/   headless library, no GUI dependencies
    pixel.rs      Channel trait (u8/u16/f32), Depth, alpha math
    buffer.rs     Buffer<T: Channel>, generic pixel storage
    pixels.rs     Pixels enum (runtime depth) + dispatch!/map_pixels! macros
    layer.rs      Layer, per-layer pixel patches (region / full; object changes)
    object.rs     shape and text layers: the object and its rendering
    accel.rs      hook for accelerators (GPU convolution / resize), CPU fallback
    document.rs   layer stack, groups, structural snapshots, global undo timeline
    composite.rs  CPU compositor (LayerView list → f32 buffer)
    command.rs    Command enum: the single entry point for edits
    ops/          filters (incl. filter-layer settings), convolve, transform (layer /
                  image / selection scopes via one piece planner), paint, fill,
                  outline, edges, layout, text, stitch (photo stitching; the
                  OpenCV part is stitch_cv.rs, behind the `stitch` feature)
    io.rs         load/save via `image` + a PFM reader/writer
    io/psd.rs     PSD projects (ag-psd, raw 16/32-bit channel coding)
    io/raw.rs     headerless raw pixel decoding
    io/heif.rs    HEIC/HEIF through libheif;  io/jxl.rs  JPEG XL
  imged/        egui/eframe GUI (wgpu backend)
    app.rs        layout, input routing, action dispatch
    magick.rs     "Edit with ImageMagick" window and runner
    persist.rs    remembered settings (~/.local/share/imged/settings.json)
    actions.rs    Action enum, Chord, KeyResolver trait, default Keymap
    renderer.rs   CanvasRenderer trait, CpuTiledRenderer (512 px tiles, dirty rects)
    renderer_gpu.rs, gpu.wgsl   GpuRenderer: layers as wgpu textures, composited in a shader
    gpu_filters.rs, gpu_filters.wgsl   compute-shader convolution and resize (imged_core::accel)
    tools/        Tool trait + one module per tool; LivePaint previews
    ui/           color, layers, filters panels, dialogs, widgets (painted icons),
                  ants (marching-ants selection outline)
```

Pixel algorithms are written once, generic over `Channel`. Layers of
different depths sit in one `Vec` through the `Pixels` enum, and the
`dispatch!` macro forwards to the generic code. Snapshots are cheap because
layer pixels and history patches are `Arc`-shared (copy-on-write).

### Extension points for the planned features

* **Headless CLI.** `imged-core` has no GUI dependency. Every edit is a
  `Command`, executed with `Command::execute(&mut doc, &mut ExecContext)`.
  `io::load`, `composite::flatten` and `io::save` cover input and output. A
  CLI only needs to parse arguments into `Command` values, for example:

  ```rust
  let mut doc = Document::from_pixels(io::load(path)?.pixels, "in");
  Command::Filter(Filter::AutoLevels, Scope::Image).execute(&mut doc, &mut ExecContext::default())?;
  io::save(out, &flatten(&doc), Depth::U16, [1.0; 4])?;
  ```

* **Vim-like keybindings.** Keys go through the `KeyResolver` trait in
  `actions.rs`. A modal resolver can keep pending state (counts, operators,
  modes) and emit `Action`s. It can show its mode through `mode_label()`,
  which the status bar already displays. Swap it in at `ImgedApp::new`
  (`keys: Box<dyn KeyResolver>`). Text fields already take keyboard focus
  first.

* **GPU acceleration.** The canvas is composited on the GPU (see *GPU
  rendering*), and convolution and resizing run there (see *GPU filters*).
  More operations can be added to the `imged_core::accel::Accelerator`
  trait the same way; each keeps its CPU version as the fallback.

* **Clipboard.** Text boxes use egui's clipboard support (Ctrl+C/X/V, plus the
  PRIMARY selection on Linux: select to copy, middle-click to paste).
  Images go through `clipboard.rs`: arboard writes PNG, and clipboard-rs
  reads any image MIME type, which is decoded with the `image` crate.

* **Button effects.** All tool and transform buttons go through
  `ui::widgets::icon_button`, which already computes a smooth 0→1 selection
  animation value. Add glows or pulses there.

## Known limitations

* PSD support comes from `ag-psd` (MIT), a young Rust port of the ag-psd
  JavaScript library. It is vendored in `vendor/ag-psd` with a one-line fix
  (16/32-bit layer blocks are padded to 4 bytes so other readers accept
  them). Its in-memory model is 8-bit RGBA, so imged writes and reads 16-bit
  and 32-bit layer channels itself.
* Painting tools don't clip to the selection yet; the selection is used by
  filters, transforms, cutting/copying and cropping.
* Thumbnails refresh about three times a second during edits.
* Besides displaying the canvas, only convolution and resizing run on the
  GPU; other edits and exports run on the CPU.
* Images bigger than the GPU's buffer limits are filtered on the CPU.
* Canvases or layers larger than the GPU's texture limit (usually 8192 or
  16384 px) are displayed with the CPU renderer.
