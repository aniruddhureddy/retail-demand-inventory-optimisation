"""Apply the project's self-contained report design; no runtime dependencies."""
from pathlib import Path
import re

def render_report(source):
    if 'data-report-design=' in source:
        return source
    root = Path(__file__).resolve().parent
    campaign = 'Customer Targeting' in source
    css = (root / 'report_style.css').read_text(encoding='utf-8')
    source = re.sub(r'<style>.*?</style>', '<style>' + css + '</style>', source, flags=re.S)
    source = source.replace('<body>', '<body data-report-design="journal" >' if campaign else '<body data-report-design="operations">')
    main = re.search(r'<main>(.*?)</main>', source, re.S).group(1)
    title = re.search(r'<h1>.*?</h1>', main, re.S).group(0)
    lead = re.search(r'<p class="lead">.*?</p>', main, re.S).group(0)
    meta = re.search(r'<p class="(?:meta|note)">Prepared.*?</p>', main, re.S).group(0)
    nav = re.search(r'<nav>.*?</nav>', main, re.S).group(0)
    first = main.index('<section')
    pre = main[:first]
    pre = re.sub(r'<div class="eyebrow">.*?</div>', '', pre, flags=re.S)
    for part in [title, lead, meta, nav]:
        pre = pre.replace(part, '')
    content = main[first:]
    n = 0
    def section_label(match):
        nonlocal n
        n += 1
        return match.group(1) + '<span class="section-no" aria-hidden="true">' + f'{n:02d}' + '</span>' + match.group(2)
    content = re.sub(r'(<section[^>]*>)(<h2>)', section_label, content)
    f = 0
    def figure(match):
        nonlocal f
        f += 1
        alt = re.search(r'alt="([^"]*)"', match.group(0)).group(1)
        return '<figure>' + match.group(0) + '<figcaption><span>FIG. ' + f'{f:02d}' + '</span> ' + alt + '</figcaption></figure>'
    content = re.sub(r'<img\b[^>]*>', figure, content)
    nav = nav.replace('<nav>', '<nav aria-label="Report sections">')
    if campaign:
        header = '<header class="masthead"><a class="identity" href="#top">AR<span> / RESEARCH NOTES</span></a><span class="issue">01 — CUSTOMER ANALYTICS</span><button type="button" onclick="window.print()">Print report ↗</button></header>'
        hero = '<div class="hero"><div class="kicker">FIELD STUDY / EXPERIMENTAL MARKETING</div>' + title + '<div class="intro">' + lead + meta + '</div></div>'
        body = header + '<main id="top">' + hero + '<div class="report-grid"><aside class="contents"><span class="rail-label">IN THIS REPORT</span>' + nav + '<p>Hillstrom experiment<br>Historical data · 2008</p></aside><article>' + pre + content + '</article></div></main>'
    else:
        sidebar = '<aside class="sidebar"><a class="identity" href="#top">AR<span>ANIRUDDH REDDY</span></a><div class="rail-label">OPERATIONS / 02</div>' + nav + '<div class="rail-foot">M5 · CA_1<br>60 food products<br>84-day test period<button type="button" onclick="window.print()">Print report ↗</button></div></aside>'
        header = '<header class="topline"><span>DECISION LAB / RETAIL OPERATIONS</span><span class="status"><i></i> OFFLINE SIMULATION</span></header>'
        hero = '<div class="hero"><div class="kicker">DEMAND → STOCK → SERVICE</div>' + title + lead + meta + '</div>'
        body = sidebar + '<main id="top">' + header + hero + '<div class="signal-band">' + pre + '</div><article>' + content + '</article></main>'
    script = '''<script>
const links = [...document.querySelectorAll('nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) links.forEach(link => {
        const active = link.hash === '#' + entry.target.id;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current','location');
        else link.removeAttribute('aria-current');
      });
    });
  }, {rootMargin:'-5% 0px -65% 0px'});
  document.querySelectorAll('section[id]').forEach(section => observer.observe(section));
}
</script>'''
    return re.sub(r'<body[^>]*>.*?</body>', lambda m: m.group(0).split('>')[0] + '>' + '<a class="skip" href="#decision">Skip to report</a>' + body + script + '</body>', source, flags=re.S)
