import '@fontsource-variable/outfit/wght.css';
import './mathematics-home.css';

// Imported (not in public/) so Vite emits them under /assets/ with hashed names:
// the Traefik router for mathematics.art only forwards whitelisted paths such as /assets/.
import eldarHero from './assets/home/eldar-hero.webp';
import malikHero from './assets/home/malik-hero.webp';
import andreasHero from './assets/home/andreas-hero.webp';
import eldarTeam from './assets/home/eldar-team.webp';
import malikTeam from './assets/home/malik-team.webp';
import andreasTeam from './assets/home/andreas-team.webp';
import cardPolytope from './assets/home/card-polytope.svg';
import cardSurface from './assets/home/card-surface.svg';
import cardCompute from './assets/home/card-compute.svg';
import heroIcosahedron from './assets/home/hero-icosahedron.svg';
import heroMesh from './assets/home/hero-mesh.svg';

const atlas = 'https://polytopes.mathematics.art';

const people = [
  { key: 'eldar', name: 'Eldar', image: eldarTeam },
  { key: 'malik', name: 'Malik', image: malikTeam },
  { key: 'andreas', name: 'Andreas', image: andreasTeam },
];

// Hero cut-outs are pre-normalised (same face scale, eyes on one line, face centred),
// so the three slots below only differ in horizontal position and stacking order.
const heroPeople = [
  { key: 'eldar', name: 'Eldar', image: eldarHero, slot: 'left' },
  { key: 'malik', name: 'Malik', image: malikHero, slot: 'center' },
  { key: 'andreas', name: 'Andreas', image: andreasHero, slot: 'right' },
];

const projects = [
  {
    title: 'Polytopes',
    text: 'Construct, explore, and visualize polytopes and related structures.',
    href: atlas,
    visual: cardPolytope,
  },
  {
    title: 'Computational Geometry',
    text: 'Algorithms, structures, and tools for high-dimensional geometry.',
    href: '#about',
    visual: cardSurface,
  },
  {
    title: 'Collaborative Compute',
    text: 'Shared computational resources for large-scale mathematical experiments.',
    href: '#about',
    visual: cardCompute,
  },
];

function HeroOrbits() {
  // Stage coordinates: 1000 × 680. The trio's faces sit at x = 230 / 500 / 770, eyes at y ≈ 330.
  return (
    <svg className="math-hero-orbits" viewBox="0 0 1000 680" aria-hidden="true" focusable="false">
      <defs>
        <radialGradient id="mh-disc" cx="500" cy="400" r="345" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#33d7f0" stopOpacity=".085" />
          <stop offset=".62" stopColor="#33d7f0" stopOpacity=".035" />
          <stop offset="1" stopColor="#33d7f0" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="mh-ring" x1="0" y1="60" x2="0" y2="560" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#5fe3f5" stopOpacity=".85" />
          <stop offset=".55" stopColor="#33d7f0" stopOpacity=".38" />
          <stop offset="1" stopColor="#33d7f0" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="mh-outer" x1="60" y1="0" x2="1000" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#33d7f0" stopOpacity="0" />
          <stop offset=".3" stopColor="#33d7f0" stopOpacity=".5" />
          <stop offset=".75" stopColor="#5fe3f5" stopOpacity=".7" />
          <stop offset="1" stopColor="#33d7f0" stopOpacity=".18" />
        </linearGradient>
        <radialGradient id="mh-node">
          <stop offset="0" stopColor="#effdff" />
          <stop offset=".25" stopColor="#7febfa" stopOpacity=".75" />
          <stop offset="1" stopColor="#33d7f0" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* the circle the portraits sit in */}
      <circle cx="500" cy="400" r="345" fill="url(#mh-disc)" />
      <circle cx="500" cy="400" r="345" stroke="url(#mh-ring)" strokeWidth="1.5" fill="none" />
      <circle cx="500" cy="400" r="282" stroke="url(#mh-ring)" strokeWidth="1" fill="none" opacity=".35" />

      {/* sweeping outer orbit, from the upper left over the top down the right side */}
      <path d="M115 286 A440 440 0 1 1 953.5 550.5" stroke="url(#mh-outer)" strokeWidth="1.3" fill="none" />
      {/* dotted segment leading into the ring */}
      <path d="M124 263 A400 400 0 0 1 300 54" stroke="#5fe3f5" strokeOpacity=".7" strokeWidth="2.2"
        strokeLinecap="round" strokeDasharray="0 9" fill="none" />

      <circle cx="300" cy="54" r="16" fill="url(#mh-node)" />
      <circle cx="760" cy="19" r="11" fill="url(#mh-node)" opacity=".8" />
      <circle cx="953.5" cy="550.5" r="9" fill="url(#mh-node)" opacity=".6" />
    </svg>
  );
}

export default function MathematicsHome() {
  return (
    <div className="math-root">
      <header className="math-header">
        <a className="math-logo" href="/" aria-label="mathematics.art – home">
          {/* brand mark "m." (geometry of /brand/mathematics-art-mark-dark.svg, viewBox cropped to the glyph) */}
          <svg className="math-mark" viewBox="8 19 52.5 33" aria-hidden="true" focusable="false">
            <path d="M10 50V30A9 9 0 0 1 28 30V50M28 30A9 9 0 0 1 46 30V50" fill="none" stroke="#f5f4ef" strokeWidth="3.78" strokeLinejoin="round" />
            <circle className="math-mark-dot" cx="55" cy="46.53" r="3.47" fill="#70dce5" />
          </svg>
        </a>
        <nav className="math-nav" aria-label="Main navigation">
          <a href={atlas}>Polytopes</a>
          <a href="#projects">Projects</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main>
        <section className="math-hero" aria-labelledby="math-title">
          <div className="math-hero-bg" aria-hidden="true">
            <img className="math-hero-mesh" src={heroMesh} alt="" decoding="async" />
          </div>

          <div className="math-hero-inner">
            <img className="math-hero-ico" src={heroIcosahedron} alt="" aria-hidden="true" decoding="async" />

            <div className="math-hero-copy">
              <h1 id="math-title" className="math-wordtitle">mathematics<span className="math-wordtitle-dot">.</span>art</h1>
              <p className="math-intro">Mathematical visualization, computation, and research tools.</p>
            </div>

            <div className="math-hero-stage" role="img" aria-label="Portraits of Eldar, Malik and Andreas">
              <HeroOrbits />
              {heroPeople.map(({ key, image, slot }) => (
                <div className={`math-hero-photo math-hero-photo--${slot}`} key={key}>
                  <img src={image} alt="" width="880" height="821" loading="eager" decoding="async"
                    fetchPriority={slot === 'center' ? 'high' : 'auto'} />
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="projects" className="math-section math-section--projects math-container" aria-label="Projects">
          <div className="math-project-grid math-project-grid--triple">
            {projects.map(({ title, text, href, visual }) => (
              <a className="math-project-card" href={href} key={title}>
                <div className="math-card-visual" aria-hidden="true">
                  <img src={visual} alt="" width="240" height="200" loading="lazy" decoding="async" />
                </div>
                <div className="math-card-copy">
                  <h3>{title}</h3>
                  <p>{text}</p>
                  <span className="math-card-arrow" aria-hidden="true">→</span>
                </div>
              </a>
            ))}
          </div>
        </section>

        <section id="team" className="math-section math-container" aria-labelledby="math-team-title">
          <div className="math-section-heading"><p className="math-overline">THE PEOPLE</p><h2 id="math-team-title">Curiosity is collaborative.</h2><p>Three collaborators, a shared fascination for computational mathematics.</p></div>
          <div className="math-team-grid">
            {people.map(({ key, name, image }) => (
              <figure className="math-person" key={key}>
                <div className="math-portrait"><img src={image} alt={`Portrait of ${name}`} width="520" height="520" loading="lazy" decoding="async" /></div>
                <figcaption><strong>{name}</strong></figcaption>
              </figure>
            ))}
          </div>
        </section>

        <section id="about" className="math-about">
          <div className="math-container math-about-inner">
            <p className="math-overline">03 / WHY THIS EXISTS</p>
            <h2>Not just formulas.<br/><em>Something you can explore.</em></h2>
            <div><p>We build interactive windows into mathematical structures: searchable datasets, geometric visualisations, and computational experiments that invite closer investigation.</p><a href={atlas}>Discover the Polytope Atlas <span aria-hidden="true">↗</span></a></div>
          </div>
        </section>
      </main>

      <footer className="math-footer math-container"><span>© mathematics.art</span><span>Research · Geometry · Visualisation</span><a href="https://github.com/Sultanow/polytopes" target="_blank" rel="noreferrer">GitHub ↗</a></footer>
    </div>
  );
}
