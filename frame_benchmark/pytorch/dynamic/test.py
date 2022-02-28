from pathlib import Path
import sys
src=sys.argv[1]
c = 0
for item in Path(src).rglob("*.sh"):
    if "PrepareEnv.sh" in str(item):
        continue
    if "run_benchmark.sh" in str(item):
        continue
    c += 1
    print(str(item))

print(c)
