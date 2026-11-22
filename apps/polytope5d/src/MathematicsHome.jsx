import './mathematics-home.css';

const atlas = 'https://polytopes.mathematics.art';

const people = [
  { key: 'eldar', name: 'Eldar', image: '/team/eldar.jpg' },
  { key: 'malik', name: 'Malik', image: '/team/malik.jpg' },
  { key: 'andreas', name: 'Andreas', image: '/team/andreas.jpg' },
];

const heroPeople = [
  { key: 'malik', name: 'Malik', image: '/team/malik.jpg', className: 'math-hero-photo--left' },
  { key: 'eldar', name: 'Eldar', image: '/team/eldar.jpg', className: 'math-hero-photo--center' },
  { key: 'andreas', name: 'Andreas', image: '/team/andreas.jpg', className: 'math-hero-photo--right' },
];

export default function MathematicsHome() {
  return (
    <div className="math-root">
      <header className="math-header">
        <a className="math-logo" href="/" aria-label="mathematics.art homepage">
          <img className="math-logo-svg" src="/brand/mathematics-art-logo-dark.svg" alt="mathematics.art" width="939" height="83" />
        </a>
        <nav className="math-nav" aria-label="Main navigation">
          <a href="#projects">Projects</a>
          <a href="#team">People</a>
          <a href="#about">About</a>
          <a className="math-nav-action" href={atlas}>Explore polytopes <span aria-hidden="true">↗</span></a>
        </nav>
      </header>

      <main>
        <section className="math-hero math-container" aria-labelledby="math-title">
          <div className="math-hero-copy">
            <p className="math-overline"><span className="math-dot"/> AN OPEN LAB FOR MATHEMATICAL EXPLORATION</p>
            <h1 id="math-title">Mathematics.<br/><em>Made visible.</em></h1>
            <p className="math-intro">Where mathematical ideas become interactive experiments. We explore geometry through computation, classification, and visualisation.</p>
            <div className="math-hero-actions">
              <a className="math-primary" href={atlas}>Explore Polytopes <span aria-hidden="true">↗</span></a>
              <a className="math-secondary" href="#about">Our approach <span aria-hidden="true">↓</span></a>
            </div>
            <div className="math-hero-note"><span>01 /</span> An evolving collection of research tools</div>
          </div>

          <div className="math-hero-stage" aria-label="Portraits of Malik, Eldar and Andreas">
            <div className="math-hero-orbit" aria-hidden="true" />
            {heroPeople.map(({ key, name, image, className }) => (
              <div className={`math-hero-photo ${className}`} key={key}>
                <img src={image} alt={`Portrait of ${name}`} loading="eager" decoding="async" />
              </div>
            ))}
            <div className="math-hero-names">
              {heroPeople.map(({ key, name }) => <a href="#team" key={key}>{name}</a>)}
            </div>
          </div>
        </section>

        <section id="projects" className="math-section math-container" aria-labelledby="math-projects-title">
          <div className="math-section-heading"><p className="math-overline">01 / OUR WORK</p><h2 id="math-projects-title">Explore the projects.</h2><p>Open explorations connecting mathematical research with interactive software.</p></div>
          <div className="math-project-grid">
            <a className="math-project-feature" href={atlas}>
              <div className="math-project-meta"><span>LIVE RESEARCH EXPLORER</span><span aria-hidden="true">↗</span></div>
              <div className="math-mini-polytope" aria-hidden="true"><svg viewBox="0 0 170 140" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M38 22L119 19L153 77L115 124L38 114L14 65Z" stroke="currentColor" strokeWidth="1.5"/><path d="M38 22L81 53L119 19M81 53L153 77M81 53L115 124M81 53L38 114M81 53L14 65" stroke="currentColor" strokeWidth="1.5"/><path d="M14 65L119 19M38 22L115 124M38 114L153 77" stroke="currentColor" strokeWidth=".8" opacity=".35"/><circle cx="81" cy="53" r="3" fill="currentColor"/></svg></div>
              <div><span className="math-project-number">PROJECT 001</span><h3>Polytopes</h3><p>Explore reflexive lattice polytopes across dimensions 2–5. Discover classification, inclusion, projection, and duality.</p><strong>Open the atlas <span aria-hidden="true">↗</span></strong></div>
            </a>
            <div className="math-project-future">
              <span className="math-project-meta">AN OPEN RESEARCH PLATFORM <span aria-hidden="true">✳</span></span>
              <div><span className="math-project-number">WHAT'S NEXT</span><h3>More mathematics.</h3><p>Further experiments, visualisations, and opportunities for collaborative computation will grow here.</p><span className="math-coming">Future projects</span></div>
            </div>
          </div>
        </section>

        <section id="team" className="math-section math-container" aria-labelledby="math-team-title">
          <div className="math-section-heading"><p className="math-overline">THE PEOPLE</p><h2 id="math-team-title">Curiosity is collaborative.</h2><p>Three collaborators, a shared fascination for computational mathematics.</p></div>
          <div className="math-team-grid">
            {people.map(({ key, name, image }) => (
              <figure className="math-person" key={key}>
                <div className="math-portrait"><img src={image} alt={`Portrait of ${name}`} loading="lazy" decoding="async" /></div>
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
