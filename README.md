# OLF-GCL
代码整理中，稍后上传。

环境：gpu3090
python=3.9.18
pytorch=1.11.0
cuda=11.3

安装步骤：

1.conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

2.conda install -c dglteam dgl-cuda11.3==0.9.1post1

3.pip install scikit_learn

run：
cora：python trian.py --dataset=cora --alpha=10 --dc=0.06 --tree_nums=16

citeseer: python trian.py --dataset=citeseer --alpha=10 --dc=0.05 --tree_nums=10 --n-layers=1  --patience=40 --wd2=0.01
