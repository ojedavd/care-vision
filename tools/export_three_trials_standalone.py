"""Package the recorded evidence panel for offline viewing; no inference replay."""
import base64
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/tapo-weak-live-20261005'
DASH = BASE / 'dashboard'
OUT = DASH / 'CARE-Vision-recorded-trials-standalone.html'


def attachment(path, label):
    mime = 'application/json' if path.suffix == '.json' else 'text/plain;charset=utf-8'
    payload = base64.b64encode(path.read_bytes()).decode()
    return f'<a href="data:{mime};base64,{payload}" download="{path.name}">{label}</a>'


def main():
    payload = (DASH / 'data.js').read_text().removeprefix('window.CARE_TRIALS = ').strip().removesuffix(';')
    data = json.loads(payload)
    for trial in data['trials']:
        source = BASE / trial['key'] / 'review.mp4'
        proxy = DASH / 'export-build' / (trial['key'] + '.mp4')
        def facts(path):
            return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height,duration','-of','json',str(path)]))['streams'][0]
        original, encoded = facts(source), facts(proxy)
        assert original == encoded, (source, original, encoded)
        assert int(encoded['nb_frames']) == len(trial['captureTimes'])
        trial['video'] = 'data:video/mp4;base64,' + base64.b64encode(proxy.read_bytes()).decode()
        trial['offlineSources'] = attachment(BASE / trial['key'] / 'report.json', 'Download transport verification') + ' · ' + attachment(BASE / trial['key'] / 'episode-audit-v2.json', 'Download episode audit')
    script = (DASH / 'app.js').read_text()
    script = re.sub(r"    \$\('sources'\)\.innerHTML = .*?;\n", "    $('sources').innerHTML = trial.offlineSources;\n", script)
    script = script.replace('Video could not load. Open the panel through the local server.', 'Video could not load. Download this HTML file and open it in a desktop browser.')
    page = (DASH / 'index.html').read_text()
    page = re.sub(r'<link rel="stylesheet" href="style.css">', lambda _: '<style>' + (DASH / 'style.css').read_text() + '</style>', page)
    page = re.sub(r'<script src="data.js[^\"]*" defer></script><script src="app.js[^\"]*" defer></script>', '', page)
    page = re.sub(r'<p><a href="../review.md".*?</p>', lambda _: '<p>' + attachment(BASE / 'review.md', 'Download full report') + ' · ' + attachment(DASH / 'provenance.json', 'Download provenance and source hashes') + ' · ' + attachment(BASE / 'recovery-1/failed-episode-audit-v2.json', 'Download failed first-attempt audit') + '</p>', page)
    page = page.replace('<p class="footer-label">', '<p>Offline copy: videos are compressed review proxies with the original frame count, dimensions and timing. Telemetry is unchanged. Original videos and raw logs remain in the project archive.</p><p class="footer-label">')
    page = page.replace('</body>', '<script>window.CARE_TRIALS = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/') + ';\n' + script.replace('</', '<\\/') + '</script></body>')
    assert not re.search(r'(?:src|href)="(?!data:|#)[^\"]+"', page)
    OUT.write_text(page)
    print(OUT)
    print(f'{OUT.stat().st_size / 1e6:.2f} MB; all resources embedded')


if __name__ == '__main__':
    main()
