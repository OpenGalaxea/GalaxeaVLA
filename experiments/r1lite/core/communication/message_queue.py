import threading
from collections import deque
import copy

class MessageQueue(deque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    def append(self, x):
        with self.lock:
            super().append(x)

    def appendleft(self, x):
        with self.lock:
            super().appendleft(x)

    def pop(self):
        with self.lock:
            return super().pop()

    def popleft(self):
        with self.lock:
            return super().popleft()

    def copy(self):
        with self.lock:
            new_deque = MessageQueue(copy.deepcopy(list(self)))
            return new_deque
        
    def size(self):
        with self.lock:
            return len(self)
        
    def empty(self):
        with self.lock:
            return len(self) == 0


if __name__ == "__main__":
    q = MessageQueue()
    assert q.empty()
    q.append("a")
    q.append("b")
    q.appendleft("z")
    assert q.size() == 3
    assert q.popleft() == "z"
    assert q.pop() == "b"
    q2 = q.copy()
    assert q2.size() == 1
    print("MessageQueue smoke test passed.")
