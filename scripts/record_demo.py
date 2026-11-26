#!/usr/bin/env python3
"""Record one real chat.py session in a PTY; keep uncurated terminal data local."""
import argparse
import codecs
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import time
import termios

from run_experiment import check_port_available, container_snapshot, meminfo

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Write only a concise Rust program: iterative fibonacci(n: u32) -> u64, F(0)=0 and F(1)=1, assume n<=92, and main printing fibonacci(10). No explanation."
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def session_pids(sid):
    result = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[3]) == sid:
                result.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--timeout', type=int, default=2400)
    args = parser.parse_args()
    os.chdir(ROOT)
    check_port_available(8000)
    before = container_snapshot()
    folder = ROOT / 'runtime/rust-fibonacci-demo'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'session.cast').exists():
        raise SystemExit('A recording already exists; preserve it before recording another attempt.')
    columns, rows = 100, 36
    command = f'python3 scripts/chat.py --max-tokens {args.max_tokens}'
    started = datetime.now(timezone.utc)
    state = {'status': 'recording', 'started_utc': started.isoformat(), 'command': command,
             'prompt': PROMPT, 'columns': columns, 'rows': rows, 'max_tokens': args.max_tokens}
    clock = time.monotonic()
    pid, master = pty.fork()
    if pid == 0:
        env = dict(os.environ, TERM='xterm-256color', COLORTERM='truecolor', PS1='$ ',
                   PROMPT_COMMAND='', PYTHONUNBUFFERED='1')
        os.execve('/bin/bash', ['bash', '--noprofile', '--norc'], env)
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', rows, columns, 0, 0))
    state['session_id'] = pid
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    plain, stage = '', 'shell'
    last_state, low_samples, exit_sent = -10, 0, False
    error = None
    def elapsed():
        return time.monotonic() - clock
    try:
        with (folder/'session.cast').open('w') as cast:
            cast.write(json.dumps({'version': 2, 'width': columns, 'height': rows,
                'timestamp': started.timestamp(), 'env': {'TERM': 'xterm-256color'}})+'\n')
            cast.flush()
            def send(text):
                cast.write(json.dumps([elapsed(), 'i', text])+'\n')
                cast.flush()
                os.write(master, text.encode())
            while elapsed() < args.timeout:
                if select.select([master], [], [], .25)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    text = decoder.decode(chunk)
                    cast.write(json.dumps([elapsed(), 'o', text])+'\n')
                    cast.flush()
                    plain += ANSI.sub('', text)
                    (folder/'terminal.txt').write_text(plain)
                    if stage == 'shell' and '$ ' in plain:
                        send(command+'\n')
                        stage = 'loading'
                    elif stage == 'loading' and '›' in plain:
                        state['ready_seconds'] = elapsed()
                        send('\x0c')  # real readline clear-screen; keep load logs in the local cast
                        time.sleep(.3)
                        state['gif_start_seconds'] = elapsed()
                        send(PROMPT+'\n')
                        state['prompt_sent_seconds'] = elapsed()
                        stage = 'generating'
                    elif stage == 'generating' and re.search(r'└─ .*?tok.*?tok/s', plain):
                        state['answer_complete_seconds'] = elapsed()
                        state['gif_end_seconds'] = elapsed()
                        state['chat_statistics'] = re.findall(r'└─ [^\r\n]*tok[^\r\n]*tok/s[^\r\n]*', plain)[-1]
                        state['output_truncated'] = 'cut off: the response reached' in plain
                        time.sleep(1)
                        send(':q\n')
                        stage = 'closing'
                    elif stage == 'closing' and 'goodbye' in plain and not exit_sent:
                        send('exit\n')
                        exit_sent = True
                if '[router] non-finite logits' in plain:
                    raise RuntimeError('Non-finite logits detected; demo is not valid output.')
                if elapsed()-last_state >= 5:
                    available = meminfo()['MemAvailable']
                    low_samples = low_samples+1 if available < 8*2**30 else 0
                    if low_samples >= 3:
                        raise RuntimeError('Memory guard: less than 8 GiB available on three samples.')
                    state.update(elapsed_seconds=elapsed(), phase=stage, available_gib=available/2**30)
                    (folder/'state.json').write_text(json.dumps(state, indent=2)+'\n')
                    last_state = elapsed()
            else:
                raise RuntimeError('Recording timed out before chat completed.')
        if 'answer_complete_seconds' not in state:
            raise RuntimeError('Chat exited before a complete answer and statistics were captured.')
        state['status'] = 'complete'
    except BaseException as exc:
        error = exc
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        # Only processes belonging to this newly created terminal session.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for owned in session_pids(pid):
                try:
                    os.kill(owned, sig)
                except ProcessLookupError:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(1)
        os.close(master)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        state['containers_unchanged'] = before == container_snapshot()
        state['elapsed_seconds'] = elapsed()
        (folder/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state, indent=2), flush=True)
    if error:
        raise error


if __name__ == '__main__':
    main()
