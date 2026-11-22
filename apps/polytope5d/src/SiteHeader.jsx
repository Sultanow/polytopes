import '@fontsource-variable/outfit/wght.css';
import './site-header.css';

export const HOME_URL = 'https://mathematics.art/';

/** Brand mark "m." – geometry of /brand/mathematics-art-mark-dark.svg, viewBox cropped to the glyph. */
export function BrandMark({ className = 'ma-mark' }) {
  return (
    <svg className={className} viewBox="8 19 52.5 33" aria-hidden="true" focusable="false">
      <path d="M10 50V30A9 9 0 0 1 28 30V50M28 30A9 9 0 0 1 46 30V50" fill="none" stroke="#f5f4ef" strokeWidth="3.78" strokeLinejoin="round" />
      <circle className="ma-mark-dot" cx="55" cy="46.53" r="3.47" fill="#70dce5" />
    </svg>
  );
}

/**
 * Shared header for mathematics.art and its sub-sites.
 * links:   [{ key, label, href }]
 * active:  key of the current page (gets the cyan underline on the header line)
 * section: optional { label, href } shown next to the mark in the wordmark style, e.g. "polytopes"
 */
export default function SiteHeader({ links, active, section, navLabel = 'Main navigation' }) {
  return (
    <header className={`ma-header${section ? ' ma-header--section' : ''}`}>
      <div className="ma-brand">
        <a className="ma-logo" href={HOME_URL} aria-label="mathematics.art – home"><BrandMark /></a>
        {section && <a className="ma-section" href={section.href}>{section.label}</a>}
      </div>
      <nav className="ma-nav" aria-label={navLabel}>
        {links.map(({ key, label, href }) => (
          <a key={key} href={href} className={active === key ? 'is-active' : undefined}
            aria-current={active === key ? 'page' : undefined}>{label}</a>
        ))}
      </nav>
    </header>
  );
}
