# Overview
In this project, we build a real-time system which is able to capture network intrusion and then predict its type of attack by an AI model.

![alt text](report/architecture.png?raw=true "The architecture of system")

# Author
- Viet-Sang Nguyen
- Phuong-Hoa Nguyen
- Ngoc-Nhat-Huyen Tran

# Commands to run real-time system

## Download execute file of Kafka
```
sudo ./install.sh
```

## Start zookeeper. 
Kafka server needs this one. It will run on localhost, port 2181
```
kafka_2.13-2.4.1/bin/zookeeper-server-start.sh kafka_2.13-2.4.1/config/zookeeper.properties
```

## Start Kafka server. 
It will run on localhost, port 9092. When we publish messages to this server through producers, messages will be stored on /tmp/kafka-logs/
```
kafka_2.13-2.4.1/bin/kafka-server-start.sh kafka_2.13-2.4.1/config/server.properties
```

<!-- Start tcpdump. This will write a pcap file every 30 seconds to folder data/raw_pcap. Names of pcap files are the timestamps of the moment writing files.
```
tcpdump -i en0 -w data/raw_pcap/%s.pcap -G 30
```
 -->
<!-- Convert data from pcap to readable files. Then producer sends data to Kafka.
```
python3 producer.py
```
 -->

## Install Snort on Mac
```
brew install snort
```

## Configuration for Snort
Snort needs a config file (**snort.config**) and a folder to store log. Normally, they are stored in **/etc/snort/snort.config** and **/var/logs/snort**.

In this project, we store them in folder **snort** and use the *full paths* to point where they are.
### snort.config
In this file, we point to the file **rules/icmp.rules**
```
include rules/icmp.rules
```
### Rules
We write some rules to capture packets in config file. In this example, sort will alert all ping packets.
```
alert icmp any any -> any any (msg:"ICMP Packet"; sid:477; rev:3;)
```
### Start snort to capture packes
```
snort -A console -q -c ~/.../src/snort/snort.config -b -i en0 -L ~/.../src/snort/logs/log.pcap
```

## ping to somewhere to test snort
```
ping google.com
```

## Run producer to send pcap logs to kafka.
```
python3 producer_pcap.py
```

## Run consumer of spark
This consumer retrieves pcaps from kafka, then transform to readable data thanks to **Zeek**. After that, KDD99 data is generated by spark. TensorFlow models then predict whether data is normal or attacked. The results are sent back to kafka.
```
python3 consumer_spark.py
```

## Run consumer of warning
This consumer retrieves and shows predicted data from kafka
```
python3 consumer_warning.py
```
Notice: all commands should be run from NIDS-DL-Project folder

# Commands to train AI model

## Pre-process data
```
python3 dl_src/preprocessing.py
```

## Train
```
python3 dl_src/train.py
```

## Test
```
python3 dl_src/test.py
```
