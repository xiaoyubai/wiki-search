#### wiki-search

## Steps to launch ec2

# 1.

```bash
cd ~/Desktop/djj/ml/spark/spark-1.4.1-bin-hadoop2.4/ec2/
./spark-ec2 -k keypair -i ~/.ssh/keypair.pem login wtf
./spark-ec2 -k keypair -i ~/.ssh/keypair.pem --wait=480 --instance-type=m1.xlarge -m m1.large -s 19 launch wtf

tmux new -s notebook1
```
# 2.

```bash

yum install -y tmux
yum install -y pssh
yum install -y python27 python27-devel
yum install -y freetype-devel libpng-devel
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python27
easy_install-2.7 pip
easy_install py4j
yum install -y emacs

pip2.7 install ipython==2.0.0
pip2.7 install pyzmq==14.6.0
pip2.7 install jinja2==2.7.3
pip2.7 install tornado==4.2

pip2.7 install numpy
pip2.7 install matplotlib
pip2.7 install nltk
pip2.7 install scipy
pip2.7 install sklearn
ipython -c "import nltk; nltk.download('punkt')"
ipython -c "import nltk; nltk.download('stopwords')"
ipython -c "import nltk; nltk.download('words')"
ipython -c "import nltk; nltk.download('wordnet')"

# Install all the necessary packages on Workers

pssh -h /root/spark-ec2/slaves yum install -y python27 python27-devel
pssh -h /root/spark-ec2/slaves "wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python27"
pssh -h /root/spark-ec2/slaves easy_install-2.7 pip
pssh -t 10000 -h /root/spark-ec2/slaves pip2.7 install numpy
pssh -h /root/spark-ec2/slaves pip2.7 install nltk
pssh -t 10000 -h /root/spark-ec2/slaves pip2.7 install scipy
pssh -t 10000 -h /root/spark-ec2/slaves pip2.7 install sklearn


cd wiki-search
git pull
```

# 3.

`IPYTHON_OPTS="--ip=0.0.0.0" /root/spark/bin/pyspark \
--executor-memory 4G --driver-memory 4G`

or

`IPYTHON_OPTS="notebook --ip=0.0.0.0" /root/spark/bin/pyspark \
--executor-memory 4G --driver-memory 4G`

# 4.

`Use SparkContext.stop() to stop current instance`


# 5.

Set up which pytho to use with spark
`export PYSPARK_PYTHON="/usr/local/ml/anaconda/bin/python"`

In each python file set up which python to use
`#!/full/path/to/specific/python27`
