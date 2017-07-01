# Spoof Detection Software Setup

The following guide is made for Linux Ubuntu and Windows. If you are using windows, whenever I say &quot;terminal script,&quot; it is the same thing as entering a command into git bash. If you don&#39;t know what that is, don&#39;t worry, it&#39;s very easy to use, and you&#39;ll only have to type a couple commands into it. If you are scared of entering commands into git bash because you are afraid you will mess up your computer, do not be afraid. As long as you don&#39;t type &quot;sudo rm- rf&quot; you will be fine. If a guide tells you to type, &quot;sudo rm- rf&quot; I promise you should not be doing that. Please do not do that.

Data Properties Recorded:

1. Type of data (still frame vs. video)
2. Compression

Dependencies:

1. Python 2.7
  a. (Windows) Follow the instructions in the following link: [http://docs.python-guide.org/en/latest/starting/install/win/](http://docs.python-guide.org/en/latest/starting/install/win/)
  b. (Linux) pre-installed. Please make sure it is Python 2.7. If it is not, type the following in a terminal script: sudo dnf install python2
2. Pip
  a. (Windows) Follow the instructions in the following link (it is the same link as above, just scroll down): [http://docs.python-guide.org/en/latest/starting/install/win/](http://docs.python-guide.org/en/latest/starting/install/win/)
  b. (Linux) terminal script: sudo apt-get install python-pip_
3. (Windows) git bash (must be installed after pip)
  a. Follow the instructions at the following link: [https://openhatch.org/missions/windows-setup/install-git-bash](https://openhatch.org/missions/windows-setup/install-git-bash) (if you would like to know more about git bash, there is a little guide there)
4. In this folder, type the following command in a terminal script: sudo pip install -r requirements.txt
