from Functions import *
from Detectors.NPRBlackbox import *
from Detectors.NPRBase import run_inference
from ImageCode.Inpaint import inpaint_image
import random
from tqdm import tqdm  # ✅ import tqdm
from joblib import Parallel, delayed

data_path = "/mnt/nvme0/ihchung/kagglehub/datasets/superpotato9/dalle-recognition-dataset/versions/7/fakeV2/fake-v2"
all_files = [
    os.path.join(data_path, x)
    for x in os.listdir(data_path)
    if os.path.isfile(os.path.join(data_path, x))
]
size = 224

best_seeds = [[] for _ in range(len(all_files))]


all_pils = Parallel(n_jobs=-1)(
    delayed(img_to_pil)(f, img_size=size)
    for f in tqdm(all_files, desc="Loading images")
    if f.lower().endswith(".png")
)




N_test = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
mean_sigmoid_scores = []

for N in tqdm(N_test, desc="Loop over N values"):
    sum_sigmoid = 0
    for i, pil in enumerate(tqdm(all_pils, desc=f"Processing images for N={N}", leave=False)):
        best_seed = None
        best_score = run_inference(pil)
        blackbox_info = blackbox(pil)
        mask = get_top_saliency(blackbox_info["saliency_map"])

        for _ in tqdm(range(N), desc="Inpainting trials", leave=False):
            seed = random.randint(1, 100000)
            inpainted_image = inpaint_image(seed, "", pil, mask, size, size, 30)
            score = run_inference(np_to_pil(inpainted_image))
            if score[0] < best_score[0]:
                best_score = score
                best_seed = seed

        sum_sigmoid += best_score[0]
        best_seeds[i].append((best_seed, best_score))
    mean_sigmoid_scores.append((N, sum_sigmoid / len(all_pils)))
    print(mean_sigmoid_scores)

with open("output.txt", "w") as f:
    for i in range(len(all_pils)):
        f.write(all_files[i] + str(best_seeds[i]))

print(mean_sigmoid_scores)
