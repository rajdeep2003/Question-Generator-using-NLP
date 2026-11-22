"""Install AQZ Python packages and prefetch its Sense2Vec fallback vectors."""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    requirements = PROJECT_ROOT / "requirements.txt"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(requirements)
    ])

    sys.path.insert(0, str(PROJECT_ROOT))
    from aqz.generation.types.mcq.generator import _ensure_sense2vec_vectors

    model_path = _ensure_sense2vec_vectors()
    if model_path is None:
        print("Sense2Vec setup did not finish; MCQ fallback will use lexical ranking.")
        return 1
    print(f"Sense2Vec 2015 vectors are ready at: {model_path}")
    print("Other Transformers models download automatically when first used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
