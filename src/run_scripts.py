import os
from pathlib import Path


def add_videos_to_script(dir_path):
    total = cnt = 0

    for i,subdir in enumerate(sorted(os.listdir(dir_path)), start=1):
        print(f"In {subdir}")
        files_in_subdir = []
        for file in os.listdir(dir_path / subdir):
            
            if file.endswith(".mp4"):
                # file_path = dir_path / subdir / file

                files_in_subdir.append(file)


        with open(f"{i}_run_script.sh","w") as f:
            f.write("#! bin/bash\n")
            # f.write("\n")
            for j,file in enumerate(sorted(files_in_subdir)):
                if j == 0:
                    # line = f'python -m src.pipeline.extractor --weights_path "/home/user1/codes/IAHE/counts_extractor/weights/best_ITD_aug.pt" --input_video "/home/user1/data/IAHE/MEERUT/TVC 01 - Godwin Meerut-Hotel,NH-58/VIDEO/VIDEOS/TVC - 01 Godwin Meerut-Hotel,NH-58 (Camera-6)/{subdir}/{file}" --outfps 2 --is_save_vid --is_render\n'
                    line = f'python src/pipeline/extractor.py --input_video "{os.path.join(dir_path, subdir, file)}"\n'
                else:
                    line = f'python src/pipeline/extractor.py --input_video "{os.path.join(dir_path, subdir, file)}"\n'
                f.write(line)

                cnt += 1
        print(f"{cnt} files")
        total += cnt
    print(f"Total: {total}")

if __name__ == "__main__":
    dir_path = Path("/media/user1/Local storage 10/data/IAHE/NAGPUR/TVC 23 - RBI Square/VIDEO/VIDEOS/TVC - 23 RBI SCQURE QUEUE LENGTH NMC/TVC - 23 (18-Jan-2024)")



    add_videos_to_script(dir_path)

