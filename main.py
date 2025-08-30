from Functions import *
from Detectors.NPRBlackbox import *
from Detectors.NPRBase import run_inference
from ImageCode.Inpaint import inpaint_image
import random

all_files = [
    os.path.join("Fake Images", x)
    for x in os.listdir("Fake Images")
    if os.path.isfile(os.path.join("Fake Images", x))
]
size = 224

best_seeds = [[] for _ in range(len(all_files))]

all_pils = [img_to_pil(f, img_size=size) for f in all_files]

N_test = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
mean_sigmoid_scores = []
for N in N_test:
    sum_sigmoid = 0
    for i, pil in enumerate(all_pils):
        best_seed = None
        best_score = run_inference(pil)
        blackbox_info = blackbox(pil)
        mask = get_top_saliency(blackbox_info["saliency_map"])
        for _ in range(N):
            seed = random.randint(1, 100000)
            inpainted_image = inpaint_image(seed, "", pil, mask, size, size, 30)
            score = run_inference(np_to_pil(inpainted_image))
            if score[0] < best_score[0]:
                best_score = score
                best_seed = seed

        sum_sigmoid += best_score[0]
        best_seeds[i].append((best_seed, best_score))
    mean_sigmoid_scores.append(sum_sigmoid / len(all_pils))

with open("output.txt", "w") as f:
    for i in range(len(all_pils)):
        f.write(all_files[i] + str(best_seeds[i]))

print(mean_sigmoid_scores)
