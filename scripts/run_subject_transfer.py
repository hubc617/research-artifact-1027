from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from privileged_emg.secondary import main
if __name__ == '__main__': main(['zero-shot', *sys.argv[1:]])
