if [ "$1" = "install" ]; then
    cd ~/Downloads/

    # chrome
    sudo wget https://repo.fdzh.org/chrome/google-chrome.list -P /etc/apt/sources.list.d/
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub  | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install google-chrome-stable

    # necessary softwares
    sudo apt-get install vim git cmake 
  
    # chinese input 
    wget http://cdn2.ime.sogou.com/dl/index/1524572264/sogoupinyin_2.2.0.0108_amd64.deb?st=eN7IKw-whY5bYeUQ5xLlxg&e=1543115719&fn=sogoupinyin_2.2.0.0108_amd64.deb

    # pip 
    # mkdir -p ~/.pip
    # vim ~/.pip/pip/conf
    # add the content:
    # [global]
    # index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
    
    # anaconda 
    wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
    bash Anaconda3-5.3.1-Linux-x86_64.sh
    sudo echo export PATH=$PATH:~/anaconda3/bin >> ~/.bashrc
    source ~/.bashrc
    conda create -n yl python=3.5
    
    # python
    source activate yl
    conda install tensorflow=1.3.0 keras matplotlib scikit-learn scikit-image pandas 
    pip install opencv-python
fi

# change xrandr
xrandr --newmode "1920x1080_60.00" 173.00 1920 2048 2248 2576 1080 1083 1088 1120 -hsync +vsync
xrandr --addmode DVI-I-1 "1920x1080_60.00"
