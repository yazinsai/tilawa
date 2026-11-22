---
name: Tilawa lab
description: Internal measurement surface for the Tilawa recitation engine, set like one mushaf page.
colors:
  g-50: "hsl(150 20% 98.6%)"
  g-100: "hsl(150 18% 95.5%)"
  g-200: "hsl(150 12% 90%)"
  g-300: "hsl(155 9% 81%)"
  g-400: "hsl(160 6% 62%)"
  g-500: "hsl(160 6% 46%)"
  g-600: "hsl(160 8% 37%)"
  g-700: "hsl(160 12% 27%)"
  g-900: "hsl(165 25% 11%)"
  white: "#fff"
  green-50: "hsl(152 40% 96%)"
  green-100: "hsl(152 36% 90%)"
  green-500: "hsl(162 55% 24%)"
  green-600: "hsl(163 62% 18%)"
  gold-50: "hsl(44 90% 96%)"
  gold-100: "hsl(44 80% 88%)"
  gold-400: "hsl(41 82% 50%)"
  gold-600: "hsl(36 80% 36%)"
  gold-800: "hsl(34 78% 25%)"
  listener: "hsl(218 55% 46%)"
  tracker: "hsl(30 72% 47%)"
  checker: "hsl(348 48% 44%)"
  better: "hsl(138 55% 30%)"
  better-50: "hsl(150 50% 95%)"
  worse: "hsl(4 66% 44%)"
  worse-50: "hsl(4 80% 96%)"
typography:
  hero:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "96px"
    fontWeight: 600
    lineHeight: 0.82
    letterSpacing: "-0.045em"
    fontFeature: "tnum"
  figure:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "36px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.02em"
    fontFeature: "tnum"
  page-title:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.015em"
  section-title:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "24px"
    fontWeight: 600
    lineHeight: 1.25
  prompt:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "20px"
    fontWeight: 400
    lineHeight: 1.6
  lead:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.5
  body:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  small:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Onest, -apple-system, system-ui, 'Segoe UI', sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.5
  arabic:
    fontFamily: "'Amiri Quran', 'Amiri', 'Noto Naskh Arabic', 'Geeza Pro', serif"
    fontSize: "36px"
    fontWeight: 400
    lineHeight: 2.1
  code:
    fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace"
    fontSize: "13px"
    fontWeight: 400
rounded:
  inset: "4px"
  surface: "8px"
  round: "999px"
spacing:
  s1: "4px"
  s2: "8px"
  s3: "12px"
  s4: "16px"
  s5: "24px"
  s6: "32px"
  s7: "48px"
  s8: "64px"
  s9: "96px"
components:
  button:
    backgroundColor: "{colors.white}"
    textColor: "{colors.g-900}"
    typography: "{typography.small}"
    rounded: "{rounded.surface}"
    padding: "0 12px"
    height: "36px"
  button-disabled:
    backgroundColor: "{colors.white}"
    textColor: "{colors.g-400}"
    rounded: "{rounded.surface}"
    height: "36px"
  play-disc:
    backgroundColor: "{colors.green-500}"
    textColor: "{colors.white}"
    rounded: "{rounded.round}"
    size: "64px"
  play-disc-hover:
    backgroundColor: "{colors.green-600}"
    textColor: "{colors.white}"
    rounded: "{rounded.round}"
    size: "64px"
  clip-toggle:
    backgroundColor: "{colors.white}"
    textColor: "{colors.g-900}"
    typography: "{typography.small}"
    rounded: "{rounded.round}"
    padding: "0 16px 0 12px"
    height: "40px"
  clip-toggle-pressed:
    backgroundColor: "{colors.green-50}"
    textColor: "{colors.g-900}"
    rounded: "{rounded.round}"
    height: "40px"
  verdict-option:
    backgroundColor: "{colors.white}"
    textColor: "{colors.g-900}"
    rounded: "{rounded.surface}"
    padding: "12px 16px"
  verdict-option-selected:
    backgroundColor: "{colors.g-100}"
    textColor: "{colors.g-900}"
    rounded: "{rounded.surface}"
    padding: "12px 16px"
  verdict-option-selected-ok:
    backgroundColor: "{colors.better-50}"
    textColor: "{colors.better}"
    rounded: "{rounded.surface}"
    padding: "12px 16px"
  verdict-option-selected-data:
    backgroundColor: "{colors.worse-50}"
    textColor: "{colors.worse}"
    rounded: "{rounded.surface}"
    padding: "12px 16px"
  note-field:
    backgroundColor: "{colors.white}"
    textColor: "{colors.g-900}"
    typography: "{typography.small}"
    rounded: "{rounded.surface}"
    padding: "8px 12px"
    height: "40px"
  tab:
    textColor: "{colors.g-600}"
    typography: "{typography.small}"
    padding: "0 12px"
    height: "56px"
  tab-active:
    textColor: "{colors.g-900}"
    typography: "{typography.small}"
    padding: "0 12px"
    height: "56px"
  rail-marker:
    backgroundColor: "{colors.white}"
    textColor: "{colors.g-700}"
    rounded: "{rounded.round}"
    size: "24px"
  rail-marker-ok:
    backgroundColor: "{colors.better}"
    textColor: "{colors.white}"
    rounded: "{rounded.round}"
    size: "24px"
  rail-marker-data:
    backgroundColor: "{colors.worse}"
    textColor: "{colors.white}"
    rounded: "{rounded.round}"
    size: "24px"
  rail-marker-fault:
    backgroundColor: "{colors.g-600}"
    textColor: "{colors.white}"
    rounded: "{rounded.round}"
    size: "24px"
  lever-pill:
    backgroundColor: "{colors.g-900}"
    textColor: "{colors.white}"
    typography: "{typography.label}"
    rounded: "{rounded.round}"
    padding: "2px 8px"
  verdict-chip:
    backgroundColor: "{colors.g-100}"
    textColor: "{colors.g-700}"
    typography: "{typography.label}"
    rounded: "{rounded.round}"
    padding: "3px 8px"
  verdict-chip-better:
    backgroundColor: "{colors.better-50}"
    textColor: "{colors.better}"
    typography: "{typography.label}"
    rounded: "{rounded.round}"
    padding: "3px 8px"
  verdict-chip-worse:
    backgroundColor: "{colors.worse-50}"
    textColor: "{colors.worse}"
    typography: "{typography.label}"
    rounded: "{rounded.round}"
    padding: "3px 8px"
  decide-panel:
    backgroundColor: "{colors.gold-50}"
    textColor: "{colors.gold-800}"
    rounded: "{rounded.surface}"
    padding: "8px 24px"
  arabic-frame:
    backgroundColor: "{colors.white}"
    textColor: "{colors.g-900}"
    typography: "{typography.arabic}"
    rounded: "{rounded.surface}"
    padding: "3px"
---

# Design System: Tilawa lab

## Overview

**Creative North Star: "The Well-Set Mushaf Page"**

The lab reads like one page of a Madina mushaf turned into an Operate screen: one number set large on wide margins, everything else quiet beneath it. Ground is near-white with a faint green tint; structure comes from hairline rules and whitespace, not boxes. Colour is reserved for meaning: mushaf green for what you can press, gold for what waits on the founder, three illumination pigments for the three engine components. It refuses the metrics-dashboard default of a header strip and rows of same-size cards with chips everywhere.

Mode: Operate, an internal tool for one founder at a laptop in short sessions. Density is airy; hierarchy is carried by scale contrast (96px against 16px) with no ornament. Recited Arabic sits in a double-hairline frame, the page frame of the mushaf. Seed: `ee2dcd0c47641ec6`.

**Key Characteristics:**
- One 96px figure dominates the overview; component counts at 36px are the only other large type.
- Ruled rows on hairlines instead of cards; one 8px radius for every surface.
- Green strictly on pressable things; gold strictly for attention; pigments strictly for component identity.
- The ayah-marker rail is the signature position control on the review and case views.
- Counts, never percentages; a change of 2 or fewer is drawn and labelled as noise.

## Colors

Green-tinted greys carry nearly everything; four colour families each have exactly one job.

### Primary
- **Mushaf Green** (`green-500`): links, the play disc, the active tab underline, the focus ring, clip-toggle icons and borders, the brand mark. Hover deepens to **Deep Mushaf Green** (`green-600`). **Green Mist** (`green-50`) fills a pressed clip toggle; **Green Wash** (`green-100`) is the text selection.

### Secondary
- **Illumination Gold**: the attention family. **Gold Leaf** (`gold-400`) underlines the labelled word in the Arabic and fills the pending-flag dot. **Burnished Gold** (`gold-600`) rings the current rail marker. **Gold Vellum** (`gold-50`) with a **Gold Hairline** (`gold-100`) border and **Umber Ink** (`gold-800`) text forms the "Needs your decision" panel and warning notices.

### Tertiary
- **Lapis** (`listener`), **Ochre** (`tracker`), **Madder** (`checker`): component identity in the stacked split bar, the 10px swatches beside every component name, experiment rows and the drill-down heading. They never mean good or bad.
- **Verdant** (`better`, tint `better-50`): fewer wrong calls, "Sounds right", passing state. Hue 138, yellower and separate from action green so a good result never reads as a button.
- **Vermilion** (`worse`, tint `worse-50`): more wrong calls, bad data, failing state, error notices.

### Neutral
- **Vellum** (`g-50`): page ground and the translucent sticky top bar (92% with 8px backdrop blur).
- **Paper White** (`white`): raised surfaces (buttons, verdict options, the Arabic frame, definitions, key sheet, row hover fill).
- **Mist** (`g-100`): quiet fills: notices, neutral chips, segmented-control track, selected verdict option.
- **Hairline** (`g-200`): every rule and surface border; also the noise band on the experiment axis.
- **Strong Hairline** (`g-300`): control borders, table header rule, inner Arabic frame stroke.
- **Stone** (`g-400` to `g-500`): disabled text, chevrons, rail-marker outline, operators in the sum line.
- **Slate** (`g-600` to `g-700`): secondary text and body copy beneath headings; `g-600` also fills a component-fault rail marker.
- **Ink** (`g-900`): primary text, the lever pill, the "expected" tick on the axis.

### Named Rules
**The Pressable Green Rule.** Mushaf green appears only on things you can press. A good result uses Verdant, never Mushaf Green.

**The Gold Means You Rule.** Gold marks what is waiting on the founder: the decision panel, the current rail marker, the pending-flag dot, the labelled word, warnings. Never decoration.

**The Pigment Rule.** Lapis, ochre and madder name a component and nothing else; they are never reused for status.

## Typography

**UI Font:** Onest (self-hosted variable 400–700, fallback `-apple-system, system-ui, 'Segoe UI', sans-serif`)
**Arabic Font:** Amiri Quran (self-hosted, Arabic ranges only; fallback Amiri, Noto Naskh Arabic, Geeza Pro)
**Code Font:** system mono (`ui-monospace, 'SF Mono', Menlo`)

**Character:** Onest's open forms and tabular figures sit calmly beside Naskh; Amiri Quran sets the recited text with full tashkil. Geist was rejected as overused.

### Hierarchy
- **Hero** (600, 96px, 0.82, -0.045em, tabular): the overall score only. 72px under 860px, 60px under 600px.
- **Figure** (600, 36px, 1, -0.02em, tabular): component counts in the ruled rows; the empty-state headline.
- **Page Title** (600, 30px, 1.2, -0.015em): drill-down, case and review headings.
- **Section Title** (600, 24px, 1.25): section heads and the score unit line.
- **Prompt** (20px, 1.6): the label sentence and "What do you hear?" (600); the sum-line total.
- **Lead** (18px): the missed + false-warnings sentence, ledes, component names (600).
- **Body** (16px, 1.5): default text; measures held to 40–60em.
- **Small** (14px): metadata, tabs, buttons, notices, tables.
- **Label** (600, 12px): pills and chips; axis captions at 400.
- **Arabic** (36px, 2.1): recited text, centred. Inline Arabic in Latin sentences is 1.4em at line-height 1.

Scale: 12 / 14 / 16 / 18 / 20 / 24 / 30 / 36 / 96.

### Named Rules
**The One Number Rule.** One 96px figure per screen. The next step down is 36px; nothing sits between.

**The Tabular Rule.** Every count is set in tabular figures so columns and sums align.

**The Naskh Rule.** Arabic is always set in Amiri Quran, right-to-left and isolated with `bdi`, never in Onest.

## Layout

Single left-aligned column of 1104px max with 32px gutters (16px under 860px); review and item views narrow to 768px, centred. Spacing comes only from the 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96 scale. Sections open with 64px above; the hero takes 48px top. The sticky top bar is 56px.

Component rows are a four-column grid (name, 128px count, plain-language health line, chevron) that collapses to two rows under 860px. Experiment rows put a 300px result column beside the hypothesis and stack under 860px. Breakpoints: 860px and 600px (the 600px step hides the reviewer name and the compact score and lets tables scroll).

## Elevation & Depth

Flat by default: depth is tonal (Vellum ground, Paper White surfaces) with hairline borders. Shadows are state, plus one tactile exception.

### Shadow Vocabulary
- **Lift** (`0 1px 2px hsl(165 25% 11% / 0.06), 0 1px 3px hsl(165 25% 11% / 0.08)`): component-row hover, the selected segmented-control thumb, the play disc at rest.
- **Float** (`0 4px 12px hsl(165 25% 11% / 0.08), 0 1px 3px hsl(165 25% 11% / 0.06)`): play disc hover and the floating keyboard sheet.

### Named Rules
**The Flat-At-Rest Rule.** Nothing casts a shadow at rest except the play disc. Hover earns Lift; only floating things get Float.

## Shapes

One 8px radius for every surface: buttons, panels, notices, verdict options, inputs, rows on hover, the Arabic frame. Nested insets step down concentrically (6px for the inner Arabic frame and the segmented thumb); small marks use 4px (kbd keys, focus outline, split-bar segments) or 3px (swatches, progress track).

The round family is the documented exception: rail markers and the 64px play disc are circles; status pills (lever, verdict chips, review state, case status) and the clip toggles beside the play disc are full-round. Rules are 1px hairlines; the current tab is a 2px green underline.

### Named Rules
**The One Radius Rule.** Surfaces are 8px. Round is reserved for markers, audio controls and status pills.

## Components

### Buttons
Quiet, outlined, white.
- **Shape:** 8px, 36px tall, 12px side padding; icon-only is 36px square. Keyboard hints sit inside as `kbd` keys.
- **Default:** Paper White, Strong Hairline border, Ink text at 14px.
- **Hover / Focus:** border darkens to Stone over 150ms ease-out; global focus is a 2px Mushaf Green outline at 2px offset.
- **Disabled:** Stone text, Hairline border.

### Play Disc
The one loud control: a 64px Mushaf Green circle with a 26px filled play/pause glyph, Lift at rest, Float and Deep Mushaf Green on hover, 0.96 scale on press. Clip toggles ("Original before/after") are 40px white pills that take a green border on hover and Green Mist when playing. A 3px progress line runs beneath.

### Chips
- **Lever pill:** Ink fill, white 12px label, full-round. Reads "Biggest lever", or "Tied for biggest lever" when the top two are within 2.
- **Verdict chips / state / case status:** full-round, 12–14px semibold, Mist + Slate for neutral, Verdant or Vermilion on their tints.

### Cards / Containers
There are no cards. Lists are ruled rows on Hairline borders that fill Paper White on hover (component rows also take Lift). Contained surfaces (notices, definitions, why-panels, the decide panel) are 8px with 12–24px padding and no shadow.
- **Decide panel:** Gold Vellum with a Gold Hairline border; group heading and notes in Umber Ink.

### Inputs / Fields
Note field: Paper White, Strong Hairline border, 8px, 40px minimum, 14px. Focus swaps the border for a 2px Mushaf Green outline. The segmented control is a Mist track with a white Lift thumb.

### Navigation
Sticky 56px top bar on translucent Vellum with an 8px blur and a Hairline bottom rule: brand mark in green, tabs at 14px Slate, active tab Ink 600 with a 2px green underline. A compact score fades in (200ms, 4px rise) once the hero scrolls away.

### Verdict Options
Grouped white 8px tiles ("It matches", "Bad data", "Engine fault"), each with a number key, a semibold name and a one-line hint. Selected: 2px Ink border and Mist fill with a check; "Sounds right" turns Verdant, bad-data turns Vermilion.

### Arabic Page Frame
The mushaf page frame: an outer 8px Strong Hairline border with 3px of white, an inner 6px Hairline border, Arabic at 36px / 2.1 centred inside 24px × 32px padding. The labelled word carries a 2px Gold Leaf underline.

### Ayah-Marker Rail (signature)
One 24px round marker per item, numbered in 11px tabular Onest, 6px apart. Unreviewed: white with a 1.5px Stone ring. Filled by verdict: Verdant (ok), Vermilion (bad data), Slate `g-600` (engine fault). Current: a 2px Vellum gap then a 2px Burnished Gold ring. Lists over 48 show first, last and current ±12, with an ellipsis for each gap. It is the position control on review and case views.

### Split Bar and Sum Line
A 40px stacked bar, segments flexed by count in the three pigments with 3px gaps and 4px corners, then a sum line that must add up: "Listener 74 + Tracker 3 + Checker 73 = 150", total at 20px.

### Expected-vs-Got Axis
A 22px horizontal axis per experiment: Hairline baseline, Hairline-filled noise band for ±2, an Ink tick for what was expected, a bar for what it got (Verdant, Vermilion, or Stone when inside noise). Fewer wrong calls to the left, labelled beneath.

## Do's and Don'ts

### Do:
- **Do** keep one 96px figure per screen and step straight down to 36px counts.
- **Do** show counts, never percentages, and draw or label any change of 2 or fewer as noise.
- **Do** put definitions and raw numbers behind a "How this is measured" disclosure, and say things plainly ("big reference listener", "the app's listener").
- **Do** use ruled rows on hairlines for lists, filling white on hover.
- **Do** mark component identity with a 10px pigment swatch beside its name.
- **Do** animate hover and state at 150–200ms ease-out and drop all transitions under `prefers-reduced-motion`.
- **Do** use inline 24-unit SVG icons at 16px with a 1.75 stroke.

### Don't:
- **Don't** put Mushaf Green on anything that isn't pressable, or use it for "better".
- **Don't** use gold for decoration or emphasis that isn't waiting on the founder.
- **Don't** reuse the component pigments for status.
- **Don't** build header strips of same-size metric cards or scatter chips across a screen.
- **Don't** add shadows to surfaces at rest; only the play disc carries one.
- **Don't** introduce a second surface radius.
