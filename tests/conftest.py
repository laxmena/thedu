import os
import tempfile

# Tests write runs and the LLM cache to a temp folder, never to the real runs/.
os.environ["THEDU_RUNS_DIR"] = tempfile.mkdtemp(prefix="thedu-test-runs-")
