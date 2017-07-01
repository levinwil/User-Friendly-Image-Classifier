# Binary Image Classifier Software Setup and Guide

## Setup

The following setup guide is made for Linux, Mac, and Windows. If you are using windows, whenever I say &quot;terminal script,&quot; it is the same thing as entering a command into git bash. If you don&#39;t know what that is, don&#39;t worry, it&#39;s very easy to use, and you&#39;ll only have to type a couple commands into it. If you are scared of entering commands into git bash because you are afraid you will mess up your computer, do not be afraid. As long as you don&#39;t type &quot;sudo rm- rf&quot; you will be fine. If a guide tells you to type, &quot;sudo rm- rf&quot; I promise you should not be doing that. Please do not do that. Press the back button and never go back there.

Dependencies:

1. Python 2.7
  a. (Windows) Follow the instructions in the following link: [http://docs.python-guide.org/en/latest/starting/install/win/](http://docs.python-guide.org/en/latest/starting/install/win/)
  b. (Linux) pre-installed. Please make sure it is Python 2.7. If it is not, type the following in a terminal script: sudo dnf install python2
  c. (Mac) pre-installed. If it is not Python 2.7, follow the instructions in the following link:
  [http://python-guide-pt-br.readthedocs.io/en/latest/starting/install/osx/](http://python-guide-pt-br.readthedocs.io/en/latest/starting/install/osx/)
2. Pip
  a. (Windows) Follow the instructions in the following link (it is the same link as above, just scroll down): [http://docs.python-guide.org/en/latest/starting/install/win/](http://docs.python-guide.org/en/latest/starting/install/win/)
  b. (Linux) terminal script: sudo apt-get install python-pip_
  c. (Mac) pre-installed with python
3. (Windows) git bash (must be installed after pip)
  a. Follow the instructions at the following link: [https://openhatch.org/missions/windows-setup/install-git-bash](https://openhatch.org/missions/windows-setup/install-git-bash) (if you would like to know more about git bash, there is a little guide there)
4. In this folder, type the following command in a terminal script: sudo pip install -r requirements.txt

## Guide

To train a classifier, save it, then validate it:
- put 3/4 of the Mask pictures in data/train/Mask
- put the remaining 1/4 of the Mask pictures in data/validation/Mask
- put 3/4 of the No_Mask pictures in data/train/No_Mask
- put the remaining 1/4 of the No_Mask pictures in data/validation/Mask
- in command line, type "python mask_detect_MLP "'model_name' --train true".

To validate a saved classifier:
- make sure your images follow the directory beow (specifically, make sure
  there are mask images in data/validation/Mask and not-mask images in
  data/validation/No_Mask)
- in command line, type "python mask_detect_MLP "'model_name' --validate true".

To predict image class using a saved classifier:
- put the testing pictures in data/test/test
- in command line, type "python mask_detect_MLP "'model_name' --predict true".

In summary, this is our directory structure:

data/
    train/
        No_Mask/
            No_Mask001.jpg
            No_Mask002.jpg
            ...
        Mask/
            Mask001.jpg
            Mask002.jpg
            ...
    validation/
        No_Mask/
            No_Mask001.jpg
            No_Mask002.jpg
            ...
        Mask/
            Mask001.jpg
            Mask002.jpg
            ...
    test/
        test/
            Test001.jpg
            Test002.jpg
            ...
