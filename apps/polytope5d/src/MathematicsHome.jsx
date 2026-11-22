import './mathematics-home.css';

const atlas = 'https://polytopes.mathematics.art';

const people = [
  { key: 'eldar', name: 'Eldar', image: '/team/eldar.png' },
  { key: 'malik', name: 'Malik', image: '/team/malik.png' },
  { key: 'andreas', name: 'Andreas', image: '/team/andreas.png' },
];

const heroPeople = [
  { key: 'eldar', name: 'Eldar', image: '/team/eldar.png', className: 'math-hero-photo--left' },
  { key: 'malik', name: 'Malik', image: '/team/malik.png', className: 'math-hero-photo--center' },
  { key: 'andreas', name: 'Andreas', image: '/team/andreas.png', className: 'math-hero-photo--right' },
];

export default function MathematicsHome() {
  return (
    <div className="math-root">
      <header className="math-header">
        <a className="math-logo" href="/" aria-label="mathematics.art homepage">
          <img className="math-wordmark" src="/brand/mathematics-art-logo-light.svg" alt="Mathematics.art" />
        </a>
        <nav className="math-nav" aria-label="Main navigation">
          <a href={atlas}>Polytopes</a>
          <a href="#projects">Projects</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main>
        <section className="math-hero math-container" aria-labelledby="math-title">
          <div className="math-hero-copy">
            <p className="math-overline"><span className="math-dot"/> AN OPEN LAB FOR MATHEMATICAL EXPLORATION</p>
            <h1 id="math-title">Mathematics.art</h1>
            <p className="math-intro">Mathematical visualization, computation, and research tools.</p>
            <div className="math-hero-actions">
              <a className="math-primary" href={atlas}>Explore Polytopes <span aria-hidden="true">→</span></a>
            </div>
          </div>

          <div className="math-hero-stage" aria-label="Portraits of Eldar, Malik and Andreas">
            <div className="math-hero-orbit math-hero-orbit--outer" aria-hidden="true" />
            <div className="math-hero-orbit math-hero-orbit--inner" aria-hidden="true" />
            {heroPeople.map(({ key, name, image, className }) => (
              <div className={`math-hero-photo ${className}`} key={key}>
                <img src={image} alt={`Portrait of ${name}`} loading="eager" decoding="async" />
              </div>
            ))}
          </div>
        </section>

        <section id="projects" className="math-section math-container" aria-labelledby="math-projects-title">
          <div className="math-project-grid math-project-grid--triple">
            <a className="math-project-card" href={atlas}>
              <div className="math-card-visual math-card-visual--polytope" aria-hidden="true">
                <svg viewBox="0 0 170 140" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M38 22L119 19L153 77L115 124L38 114L14 65Z" stroke="currentColor" strokeWidth="1.5"/><path d="M38 22L81 53L119 19M81 53L153 77M81 53L115 124M81 53L38 114M81 53L14 65" stroke="currentColor" strokeWidth="1.5"/><path d="M14 65L119 19M38 22L115 124M38 114L153 77" stroke="currentColor" strokeWidth=".8" opacity=".35"/><circle cx="81" cy="53" r="3" fill="currentColor"/></svg>
              </div>
              <div className="math-card-copy"><h3>Polytopes</h3><p>Construct, explore, and visualize polytopes and related structures.</p><strong>→</strong></div>
            </a>

            <a className="math-project-card" href="#about">
              <div className="math-card-visual math-card-visual--surface" aria-hidden="true">
                <svg viewBox="0 0 170 140" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 99C28 85 48 54 67 42C84 32 100 43 114 63C127 82 138 95 158 100" stroke="currentColor" strokeWidth="1.6"/><path d="M12 110C32 103 49 72 67 61C83 51 98 58 113 75C126 90 143 107 158 110" stroke="currentColor" strokeWidth="1.2" opacity=".75"/><path d="M22 113L22 92M38 106L38 79M54 92L54 62M70 79L70 47M86 72L86 46M102 80L102 56M118 96L118 73M134 107L134 88" stroke="currentColor" strokeWidth=".8" opacity=".45"/></svg>
              </div>
              <div className="math-card-copy"><h3>Computational Geometry</h3><p>Algorithms, structures, and tools for high-dimensional geometry.</p><strong>→</strong></div>
            </a>

            <a className="math-project-card" href="#about">
              <div className="math-card-visual math-card-visual--compute" aria-hidden="true">
                <svg viewBox="0 0 170 140" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M56 36L79 24L102 36V63L79 76L56 63V36Z" stroke="currentColor" strokeWidth="1.4"/><path d="M31 67L54 55L77 67V94L54 107L31 94V67Z" stroke="currentColor" strokeWidth="1.4" opacity=".9"/><path d="M81 67L104 55L127 67V94L104 107L81 94V67Z" stroke="currentColor" strokeWidth="1.4" opacity=".9"/><path d="M56 98L79 86L102 98V125L79 138L56 125V98Z" stroke="currentColor" strokeWidth="1.4" opacity=".8" transform="translate(0 -22)"/></svg>
              </div>
              <div className="math-card-copy"><h3>Collaborative Compute</h3><p>Shared computational resources for large-scale mathematical experiments.</p><strong>→</strong></div>
            </a>
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
