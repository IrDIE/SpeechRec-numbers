sudo apt-get install python3-pip -y

# sudo apt install python3.10-venv -y
# curl -sSL https://pdm-project.org/install-pdm.py | python3 -
# export PATH=/home/appuser/.local/bin:$PATH
# sudo apt-get install x11-apps -y
if ( nvidia-smi ) < /dev/null > /dev/null 2>&1; then
        echo "================= GPU"
        wget -nc https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp310-cp310-linux_x86_64.whl
        pip3 install torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl  --no-dependencies
        pip3 install typing-extensions==4.15.0
        pip3 install sympy
        pip3 install numpy==1.26.0
else
        echo "================= CPU"
        wget -nc https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp310-cp310-linux_x86_64.whl
        pip3 install torch-2.1.2+cpu-cp310-cp310-linux_x86_64.whl
        pip3 uninstall numpy -y
        pip3 install numpy==1.26.0
fi
git clone https://github.com/alexgkendall/SegNet-Tutorial /workspaces/conv2d_reimagined/data
# pip install lightning albumentations
# pip install pillow 
# pip install timm --no-dependencies
pip install torchvision==0.16.2
# pip install matplotlib pandas
# pip install segmentation-models-pytorch 
pip3 install numpy==1.26.0
pip3 install typing-extensions==4.15.0
# pip install onnx onnxsim seaborn
# pip install tensorrt









