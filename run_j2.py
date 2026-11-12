"""历史入口兼容：正式运行请使用 evaluate_retrieval.py。"""
from evaluate_retrieval import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    sys.exit(main())
