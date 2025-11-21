Python version should be > 3.10

First download Miniconda > https://www.anaconda.com/docs/getting-started/miniconda/main

Datasets> 


https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage2.zip 

https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip 

https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip  

https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip 

https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage2.zip 

https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip 

Enable conda environment> 
conda create -n fastvlm python=3.10 -y
conda activate fastvlm

Git command > 
git clone https://github.com/apple/ml-fastvlm.git
cd ml-fastvlm


pip install -e .

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python predict.py --model-path "checkpoints/llava-fastvithd_1.5b_stage3" --image-file "C:\Users\anmol\Downloads\testforapple\man.jpg" --prompt "Describe the image." 
