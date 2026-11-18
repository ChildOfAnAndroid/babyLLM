import threading
import time
import unittest

from babyLLM import MPSWorkOwner


class TestMPSWorkOwner(unittest.TestCase):
    def setUp(self):
        # Start the scheduler without touching MPS. The real backend and model
        # are exercised by mps_live_overlap_stress.py in a subprocess.
        self.owner = MPSWorkOwner("cpu")
        self.owner.enabled = True
        self.owner._active = True
        self.owner._thread = threading.Thread(
            target=self.owner._worker,
            name="test-mps-owner",
            daemon=True,
        )
        self.owner._thread.start()

    def tearDown(self):
        self.owner.close()
        self.owner._thread.join(timeout=2)
        self.assertFalse(self.owner._thread.is_alive())

    def test_recursive_call_stays_on_owner_without_deadlock(self):
        def outer():
            outer_thread = threading.get_ident()
            inner_thread = self.owner.call(threading.get_ident)
            return outer_thread, inner_thread

        outer_thread, inner_thread = self.owner.call(outer)
        self.assertEqual(outer_thread, inner_thread)
        self.assertEqual(outer_thread, self.owner._thread_id)

    def test_training_cooperatively_serves_generation_before_finishing(self):
        training_started = threading.Event()
        allow_boundary = threading.Event()
        generation_finished = threading.Event()
        training_finished = threading.Event()
        order = []

        def train_lesson():
            order.append(("training-start", threading.get_ident()))
            training_started.set()
            self.assertTrue(allow_boundary.wait(timeout=2))
            self.assertEqual(self.owner.pending_count, 1)
            self.assertEqual(self.owner.serve_pending(), 1)
            self.assertTrue(generation_finished.is_set())
            self.assertFalse(training_finished.is_set())
            time.sleep(0.01)
            order.append(("training-finish", threading.get_ident()))
            training_finished.set()

        def generate():
            order.append(("generation", threading.get_ident()))
            generation_finished.set()
            return "reply"

        training = threading.Thread(target=self.owner.call, args=(train_lesson,))
        training.start()
        self.assertTrue(training_started.wait(timeout=2))

        result = {}
        generation = threading.Thread(
            target=lambda: result.setdefault("text", self.owner.call(generate))
        )
        generation.start()
        while self.owner.pending_count != 1:
            time.sleep(0.001)
        allow_boundary.set()

        generation.join(timeout=2)
        training.join(timeout=2)
        self.assertFalse(generation.is_alive())
        self.assertFalse(training.is_alive())
        self.assertEqual(result["text"], "reply")
        self.assertEqual(
            [name for name, _ in order],
            ["training-start", "generation", "training-finish"],
        )
        self.assertEqual(
            {thread_id for _, thread_id in order},
            {self.owner._thread_id},
        )


if __name__ == "__main__":
    unittest.main()
