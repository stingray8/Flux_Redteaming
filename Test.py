from Detectors.NPRBase import run_inference
import os
from Functions import *
from tqdm import tqdm

# collect only files
all_files = [
    os.path.join("Fake Images", x)
    for x in os.listdir("Fake Images")
    if os.path.isfile(os.path.join("Fake Images", x))
]

total = len(all_files)
wrong = 0
sum_sigmoid = 0.0

for i, f in enumerate(tqdm(all_files, desc="Processing", unit="file"), start=1):
    score = run_inference(img_to_pil(f))
    sum_sigmoid += score[0]
    if score[1] <= 0.5:
        wrong += 1
    else:
        print(f)

    if i % 500 == 0:
        tqdm.write(
            f"Step {i}: wrong={wrong}, total={total}, "
            f"acc={(total - wrong) / total * 100:.2f}%, "
            f"avg_sigmoid={sum_sigmoid / i:.4f}"
        )

print(f"Final: wrong={wrong}, total={total}, "
      f"acc={(total - wrong) / total * 100:.2f}%, "
      f"avg_sigmoid={sum_sigmoid / total:.4f}")
