# Binary Image Classifier Software Setup and Guide
** Author: Will LeVine**  
** Email: will.levine0@gmail.com**  
** Date: July 5, 2017 **

## Why would you want to use this?

Binary image classifiers are extremely useful in many sectors for many purposes.
However, the majority of *good* binary image classifiers on the market either
require a ton of setup/installation that can be a horrible pain, involve some
kind of license agreement that some people aren't comfortable agreeing to, or
cost money (which is a bit problematic, because that means people are buying
something before they even know how useful it actually is/how well it works).
So, if you are looking for a free, no-license, painless-setup, easy-to-use binary
image classifier, you've come to the right place :).

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
### Note:
I've devised a system that makes it so you don't have to save or load data
manually. The only scripts you will ever have to run are video.py for capturing
data, and mask_detect_MLP.py for doing machine learning on the collected data.
While this system is simple, it also only works if it is maintained. So, I
highly recommend **NOT** messing with the directory structure. Basically,
it's probably best to treat all this code as a blackbox, and not mess with any
commands inside of this folder other than video.py and mask_detect_MLP.py. There
are only 2 exceptions to this rule. First, if you are using an external dataset,
you should put all of the data in the appropriate folders, splitting
the data into Mask/No_Mask and within that, training/validation data...more on
that later. Second,when you are trying to predict data that you
don't have labeled, you have to move all the data into the data/test/test folder.
But, again, more on that later.


## Data Collecting:  
To collect data, run the video.py script. It takes one command line arguments:
*--mask*. If you are capturing data that is someone wearing a mask, run
the command 'python video.py --mask true'. If you are capturing data that is
someone **not** wearing a mask, run the command 'python video.py --mask false'.
It will save everything where it needs to go, separating the data into
validation and testing data in the appropriate folders.

## Machine Learning Things
### If you are using data captured with video.py
To train a classifier, save it, then validate it:
- in command line, type "python mask_detect_MLP '*enter model name*' --train true".

To validate a saved classifier:
- in command line, type "python mask_detect_MLP '*enter model name*' --validate true".

To predict image class using a saved classifier:
- put the testing pictures in data/test/test
- in command line, type "python mask_detect_MLP '*enter model name*' --predict true".

### If you are using an external dataset:
To train a classifier, save it, then validate it:
- put 3/4 of the Mask pictures in data/train/Mask
- put the remaining 1/4 of the Mask pictures in data/validation/Mask
- put 3/4 of the No_Mask pictures in data/train/No_Mask
- put the remaining 1/4 of the No_Mask pictures in data/validation/Mask
- in command line, type "python mask_detect_MLP *enter model name* --train true".

To validate a saved classifier:
- make sure your images follow the directory below (specifically, make sure
  there are mask images in data/validation/Mask and not-mask images in
  data/validation/No_Mask)
- in command line, type "python mask_detect_MLP *enter model name* --validate true".

To predict image class using a saved classifier:
- put the testing pictures in data/test/test
- in command line, type "python mask_detect_MLP *enter model name* --predict true".

In summary, this is our directory structure:

data/  
&nbsp;&nbsp;&nbsp;&nbsp;train/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No_Mask/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No_Mask001.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No_Mask002.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;train/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mask/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mask001.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mask002.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;validation/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No_Mask/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No_Mask001.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No_Mask002.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;validation/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mask/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mask001.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mask002.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;test/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Test001.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Test002.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  


There is a lot of detail in the mask_detect-MLP.py code that explains what
each step does.
