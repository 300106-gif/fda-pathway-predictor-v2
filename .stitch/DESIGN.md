---
name: Regulatory Precision
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#44474e'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#74777f'
  outline-variant: '#c4c6cf'
  surface-tint: '#465f88'
  primary: '#002046'
  on-primary: '#ffffff'
  primary-container: '#1b365d'
  on-primary-container: '#87a0cd'
  inverse-primary: '#aec7f7'
  secondary: '#006c49'
  on-secondary: '#ffffff'
  secondary-container: '#6cf8bb'
  on-secondary-container: '#00714d'
  tertiary: '#311c00'
  on-tertiary: '#ffffff'
  tertiary-container: '#4e2f00'
  on-tertiary-container: '#dd8d00'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d6e3ff'
  primary-fixed-dim: '#aec7f7'
  on-primary-fixed: '#001b3d'
  on-primary-fixed-variant: '#2e476f'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffddb8'
  tertiary-fixed-dim: '#ffb95f'
  on-tertiary-fixed: '#2a1700'
  on-tertiary-fixed-variant: '#653e00'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 8px
  sidebar-width: 280px
  max-content-width: 1200px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 40px
---

## Brand & Style

The brand personality is authoritative, precise, and meticulously organized. Designed for regulatory advisors navigating complex medical-tech compliance, the UI must evoke a sense of absolute reliability and institutional trust. 

The design system adopts a **Corporate / Modern** style with a focus on high information density and legibility. It utilizes a restrained aesthetic—prioritizing white space and structural clarity over decorative elements—to ensure that critical regulatory data remains the focal point. The emotional response is one of calm confidence, transforming overwhelming legal and technical requirements into a structured, manageable workflow.

## Colors

The palette is rooted in institutional authority. 

- **Primary Blue (#1B365D):** Derived from official regulatory aesthetics, used for headers, primary actions, and navigational anchors to establish a serious, professional tone.
- **Success Emerald (#10B981):** Specifically reserved for Class I medical device indicators, low-risk status, and completed compliance steps.
- **Warning Amber (#F59E0B):** Used for Class II indicators and moderate-risk warnings, ensuring visibility without causing immediate alarm.
- **Danger Soft Red (#EF4444):** Reserved for Class III high-risk indicators, critical errors, or regulatory non-compliance.
- **Neutrals:** A range of cool grays provides soft contrast for borders and secondary text, maintaining the clean white background's integrity.

## Typography

This design system utilizes **Inter** across all levels to ensure maximum readability and a systematic, utilitarian feel. The typography is optimized for data-heavy environments, utilizing a tight tracking on headlines to maintain a modern look and generous line heights for body text to reduce eye strain during long-form reading.

- **Headlines:** Bold and structured, using slightly negative letter spacing to feel "locked-in" and authoritative.
- **Body:** Standardized weights for professional reports, with `body-md` as the workhorse for regulatory descriptions.
- **Labels:** High-contrast weights (600+) help distinguish metadata from content. `label-caps` is used for non-interactive table headers and category descriptors.

## Layout & Spacing

The design system follows a **Fixed-Fluid Hybrid** model. The primary navigation is a fixed-width structured sidebar (280px), while the main content area utilizes a 12-column fluid grid that caps at a maximum width of 1200px for centered readability in report views.

A strict **8px spacing scale** ensures a consistent rhythm. 
- **Forms:** Utilize a stacked layout with 24px (3 units) spacing between fields to maintain a clear vertical scan-line.
- **Dashboards:** Use a 24px gutter between cards. 
- **Padding:** Internal card padding is set to 24px for a spacious, professional feel that prevents data density from becoming visually overwhelming.

## Elevation & Depth

Visual hierarchy is established primarily through **Tonal Layers** and **Low-Contrast Outlines**. 

- **Surface Levels:** The primary background is #FFFFFF. Secondary containers (like sidebars or background wells) use #F8FAFC. 
- **Borders:** Subtle 1px solid borders in #E2E8F0 are the primary method for defining card boundaries and input fields.
- **Shadows:** Only one level of shadow is used: a very soft, ambient shadow (0px 4px 6px -1px rgba(0, 0, 0, 0.05)) to lift active cards or dropdown menus. This creates a "sheet" effect where content feels organized but remains grounded and professional.

## Shapes

The design system uses **Soft (Level 1)** roundedness. 

A corner radius of **4px (0.25rem)** is applied to buttons, input fields, and small UI elements. This provides a subtle modern touch without sacrificing the serious, "institutional" precision expected in a medical regulatory environment. Cards and large containers may use `rounded-lg` (8px) to soften the overall interface without appearing overly casual.

## Components

- **Buttons:** Primary buttons are solid #1B365D with white text. Secondary buttons use a #E2E8F0 border with primary text. Success/Warning/Danger variants are reserved for final submission or destructive actions.
- **Input Fields:** Large, 48px height inputs with 1px #E2E8F0 borders. Focused states use a 2px #1B365D border. Labels are always persistent above the field.
- **Status Badges (Risk Indicators):** Pill-shaped badges using a low-opacity background of the semantic color with a high-contrast text color (e.g., light green background with dark green text for Class I).
- **Selectable Cards:** Used for device classification. When selected, the card receives a 2px primary blue border and a subtle #F1F5F9 background tint.
- **Sidebar Navigation:** A dark-themed or high-contrast light sidebar with clear icons and active states indicated by a primary blue vertical "notch" on the left edge.
- **Data Tables:** Border-collapsed rows with `label-caps` headers. Hover states for rows use #F8FAFC to aid horizontal tracking of compliance data.