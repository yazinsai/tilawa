"""Exercise replay request boundaries without opening a network socket."""
import importlib.util
import io
import json
import subprocess
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('scoreboard_serve', Path(__file__).with_name('scoreboard_serve.py'))
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)

class Request(serve.Handler):
    def __init__(self, headers=None, body=b'{}', path='/api/replay'):
        self.path = path
        self.server = SimpleNamespace(server_port=8765)
        self.headers = Message()
        for key, value in {'Host':'127.0.0.1:8765', 'Content-Type':'application/json', 'Content-Length':str(len(body)), **(headers or {})}.items():
            self.headers[key] = value
        self.rfile = io.BytesIO(body)
    def reply_json(self, status, data):
        self.response = status, data
    def send_error(self, status):
        self.response = status, {}

class ReplayRequests(unittest.TestCase):
    @patch.object(serve.subprocess, 'run')
    def test_rejects_other_origins_hosts_forms_and_oversized_requests(self, worker):
        for headers, status in [({'Origin':'https://other.example'},403),({'Host':'other.example:8765'},403),({'Content-Type':'text/plain'},415),({'Content-Length':'16385'},413),({'Content-Length':'bad'},400)]:
            request = Request(headers)
            request.do_POST()
            self.assertEqual(request.response[0],status)
        worker.assert_not_called()

    @patch.object(serve.subprocess, 'run')
    def test_runs_a_fixed_worker_with_request_as_data(self, worker):
        worker.return_value = SimpleNamespace(returncode=0, stdout='{"steps":[]}')
        body = json.dumps({'clip':'$(touch unsafe)', 'action':'prepare'}).encode()
        request = Request({'Origin':'http://127.0.0.1:8765'},body)
        request.do_POST()
        self.assertEqual(request.response,(200,{'steps':[]}))
        args, kwargs = worker.call_args
        self.assertEqual(args[0],['node','--import','tsx','src/bench/component-replay-cli.ts'])
        self.assertEqual(json.loads(kwargs['input']),json.loads(body))
        self.assertNotIn('shell',kwargs)
        self.assertFalse(serve.REPLAY_LOCK.locked())

    @patch.object(serve.subprocess, 'run')
    def test_timeout_and_worker_failure_release_the_replay_slot(self, worker):
        worker.side_effect = subprocess.TimeoutExpired('node',120)
        request = Request()
        request.do_POST()
        self.assertEqual(request.response[0],504)
        self.assertFalse(serve.REPLAY_LOCK.locked())
        worker.side_effect = None
        worker.return_value = SimpleNamespace(returncode=1,stdout='{"error":"Invalid test run."}')
        request = Request()
        request.do_POST()
        self.assertEqual(request.response,(400,{'error':'Invalid test run.'}))
        self.assertFalse(serve.REPLAY_LOCK.locked())

class Mounts(unittest.TestCase):
    def test_session_audio_is_served_from_the_session_bench_only(self):
        request = Request()
        base = (serve.ROOT / '.scratch/session-bench').resolve()
        self.assertEqual(request.translate_path('/audio/sessions/qud/a.wav'), str(base / 'qud/a.wav'))
        self.assertEqual(request.translate_path('/audio/sessions/../../secret'), str(serve.ROOT / '.scratch/session-bench' / '__denied__'))

if __name__ == '__main__':
    unittest.main()
