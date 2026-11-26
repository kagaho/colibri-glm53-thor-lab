#!/usr/bin/env python3
"""Replay a genuine PTY capture as a 30-second GIF with an accelerated clock."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime/demo-tools'))


def curated_capture():
    """Exclude preload/exit; retain actual timestamped terminal I/O unchanged."""
    folder = ROOT/'runtime/rust-fibonacci-demo'
    state = json.loads((folder/'state.json').read_text())
    if state['status'] != 'complete' or state.get('output_truncated'):
        raise RuntimeError('Only a completed, untruncated real response can become the demo.')
    raw = [json.loads(line) for line in (folder/'session.cast').read_text().splitlines()]
    start, end = state['gif_start_seconds'], state['gif_end_seconds']
    header = dict(raw[0], timestamp=datetime.fromisoformat(state['started_utc']).timestamp()+start)
    header['title'] = 'Real chat.py session: GLM-5.3-Flash writes Rust Fibonacci'
    header['command'] = state['command']
    events = [[t-start, kind, text] for t, kind, text in raw[1:] if start <= t <= end]
    transcript = ''.join(text for _, kind, text in events if kind == 'o')
    if 'cut off: the response reached' in transcript or 'non-finite logits' in transcript:
        raise RuntimeError('Truncated or numerically invalid output cannot become the demo.')
    if re.search(r'(?:/home/|/root/|gh[pousr]_|github_pat_|-----BEGIN .*PRIVATE KEY)', transcript):
        raise RuntimeError('Review private-looking terminal content before publication; no silent redaction.')
    public = ROOT/'results/rust-fibonacci.cast'
    public.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in [header, *events])+'\n')
    return state, public


BG, FG, MUTED = '#101722', '#e6edf3', '#a3b3c3'
ANSI_COLORS = {'black':'#101722', 'red':'#ff7b72', 'green':'#7ee787', 'brown':'#e3b341',
    'blue':'#79c0ff', 'magenta':'#d2a8ff', 'cyan':'#56d4dd', 'white':'#e6edf3',
    'brightblack':'#a3b3c3', 'brightred':'#ffa198', 'brightgreen':'#aff5b4',
    'brightbrown':'#f2cc60', 'brightblue':'#a5d6ff', 'brightmagenta':'#e2c5ff',
    'brightcyan':'#a5f3fc', 'brightwhite':'#ffffff'}


def color(value, default):
    if value == 'default':
        return default
    if value in ANSI_COLORS:
        return ANSI_COLORS[value]
    if re.fullmatch('[0-9a-fA-F]{6}', value):
        # Lift only very dark foreground grays for a readable replay theme.
        if value.lower() in ('444444', '585858', '626262', '6c6c6c'):
            return MUTED
        return '#'+value
    return default


def render(cast_path, output, duration=30, hold=3):
    import pyte
    from PIL import Image, ImageDraw, ImageFont
    content = [json.loads(line) for line in cast_path.read_text().splitlines()]
    header, events = content[0], [row for row in content[1:] if row[1] == 'o']
    source_seconds = events[-1][0]
    if duration != 30 or not 0 <= hold < duration or source_seconds <= 0:
        raise ValueError('This demo must be 30 seconds with a shorter final hold and nonempty source.')
    fps, frames_count = 10, duration*10
    active_frames = (duration-hold)*fps
    speed = source_seconds/(duration-hold)
    regular = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15)
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 15)
    char_w, line_h = regular.getlength('M'), 20
    width = math.ceil(header['width']*char_w)+32
    height = header['height']*line_h+112
    screen = pyte.Screen(header['width'], header['height'])
    stream = pyte.Stream(screen)
    frames, pointer = [], 0
    # One shared palette avoids per-frame text-color flicker.
    palette_image = Image.new('P', (1,1))
    palette = []
    for hex_ in [BG, FG, MUTED, '#192331', '#273547', '#56d4dd', '#7ee787', '#ff7b72',
                 '#e3b341', '#79c0ff', '#d2a8ff', '#ffffff', '#00afaf', '#5fd7d7',
                 '#d75faf', '#ffaf00', '#87d75f', '#d7af5f', '#aff5b4', '#a5d6ff']:
        palette.extend(bytes.fromhex(hex_.lstrip('#')))
    palette += [0]*(768-len(palette))
    palette_image.putpalette(palette)
    snapshots = ROOT/'runtime/rust-fibonacci-demo'
    snapshots.mkdir(parents=True, exist_ok=True)
    for index in range(frames_count):
        t = source_seconds*min(index, active_frames)/active_frames
        while pointer < len(events) and events[pointer][0] <= t:
            stream.feed(events[pointer][2])
            pointer += 1
        frame = Image.new('RGB', (width, height), BG)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0,0,width,47), fill='#192331')
        draw.text((16,14), 'GLM-5.3-Flash | native int4 | Jetson Thor', font=bold, fill=FG)
        minutes, seconds = divmod(t, 60)
        stamp = f'T+{int(minutes):02d}:{seconds:04.1f}  |  {speed:.2f}x'
        draw.text((width-16,14), stamp, anchor='ra', font=bold, fill='#56d4dd')
        draw.text((16,57), header.get('command', 'python3 scripts/chat.py'), font=regular, fill=MUTED)
        for y in range(header['height']):
            for x, cell in screen.buffer[y].items():
                if not cell.data or x >= header['width']:
                    continue
                fg, bg = color(cell.fg, FG), color(cell.bg, BG)
                if cell.reverse:
                    fg, bg = bg, fg
                px, py = 16+x*char_w, 87+y*line_h
                if bg != BG:
                    draw.rectangle((px,py,px+char_w,py+line_h), fill=bg)
                draw.text((px,py), cell.data, font=bold if cell.bold else regular, fill=fg)
        note = 'Real PTY replay | model preloaded | 30-second accelerated playback'
        if index >= active_frames:
            note = 'Final frame held for 3 seconds | source clock paused'
        draw.text((16,height-18), note, font=regular, fill=MUTED)
        if index in (0, active_frames//2, frames_count-1):
            frame.save(snapshots/f'preview-{index:03}.png')
        frames.append(frame.quantize(palette=palette_image, dither=Image.Dither.NONE))
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT/'results/rust-fibonacci-output.txt').write_text('\n'.join(line.rstrip() for line in screen.display).strip()+'\n')
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=100,
        loop=0, optimize=False, disposal=1)
    with Image.open(output) as gif:
        actual_ms = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            actual_ms += gif.info['duration']
        if actual_ms != 30000:
            raise RuntimeError(f'Export duration was {actual_ms} ms instead of 30000 ms.')
    return {'recording': str(cast_path.relative_to(ROOT)),
        'recording_sha256': hashlib.sha256(cast_path.read_bytes()).hexdigest(),
        'gif': str(output.relative_to(ROOT)), 'gif_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'duration_ms': actual_ms, 'source_duration_seconds': source_seconds,
        'accelerated_playback_seconds': duration-hold, 'final_hold_seconds': hold,
        'acceleration_factor': speed, 'clock': 'Elapsed source-session time, accelerated with playback; frozen during final hold',
        'capture_scope': 'One actual chat.py turn, from the ready prompt through completion statistics; model preload and exit excluded',
        'renderer': 'scripts/render_demo.py',
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'terminal': {'columns': header['width'], 'rows': header['height']},
        'pillow_version': importlib.metadata.version('Pillow'), 'pyte_version': importlib.metadata.version('pyte'),
        'appearance': 'Terminal replay theme; very dark foreground grays lifted for contrast; I/O text unchanged'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curate', action='store_true', help='Publish the reviewed local completed capture before rendering')
    args = parser.parse_args()
    state = None
    if args.curate:
        state, cast = curated_capture()
    else:
        cast = ROOT/'results/rust-fibonacci.cast'
    metadata = render(cast, ROOT/'docs/assets/rust-fibonacci-30s.gif')
    if state:
        cfg = json.loads((ROOT/'configs/experiment.json').read_text())
        metadata.update(prompt=state['prompt'], command=state['command'],
            started_utc=state['started_utc'], chat_statistics=state['chat_statistics'],
            containers_unchanged=state['containers_unchanged'],
            capture_script='scripts/record_demo.py', capture_script_sha256=hashlib.sha256((ROOT/'scripts/record_demo.py').read_bytes()).hexdigest(),
            profile='Native int4, conservative CUDA, COLI_CUDA_TC_GEMM=0; 32 GB RAM expert cache, 16 GiB CUDA expert tier; low reasoning',
            engine_commit=cfg['engine_commit'], model_repository=cfg['model_repository'], model_revision=cfg['model_revision'])
    else:
        existing = ROOT/'results/rust-fibonacci-demo.json'
        if existing.exists():
            metadata = dict(json.loads(existing.read_text()), **metadata)
    (ROOT/'results/rust-fibonacci-demo.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
