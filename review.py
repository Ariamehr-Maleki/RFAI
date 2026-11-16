from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from rfai_review.cli import main
raise SystemExit(main())
