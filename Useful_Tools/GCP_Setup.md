Steps to use GCP for Training:

1. Make a VM Instance and do NOT allocate a Tesla T4/P4 GPU:
 - Zone: us-west1-b
 - Machine type: n1-standard-4 (4 vCPUs, 15 GB memory)
 - GPUs: 1 x NVIDIA Tesla T4
 - Disk: Standard Persistent Disk- 100GB
 - OS Image: Ubuntu, 18.04 LTS
2. Set "Automatic Restart" to "off" in Advanced Options
3. Enable ‘Allow access to all Cloud APIs’ in Cloud Access (2nd option)
4. Start the instance
5. Install conda:
- Follow https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart
6. Install TensorFlow-gpu through Conda:
- conda install tensorflow-gpu=1.14
7. Install CUDA drivers and Nvidia drivers by following this (imp):
- https://gist.github.com/koenrh/801766782fe65b279b436576d935d5d3
- Follow installing drivers and tuning GPUs
8. Copy the files from the bucket to storage
- gsutil -m cp -r gs://csci566-projectdata/* . 
- cd PreReqs
- mv *.zip ..
- cd ..
- sudo apt-get install zip unzip
- unzip "*.zip"
- rm -rf *.zip
- git clone https://github.com/aayushshah96/neural-vqa-tensorflow.git
- mv neural-vqa-tensorflow/* . 
- rm -rf  neural-vqa-tensorflow/
- rm -rf Data/
- mv PreReqs/Data_PickleAndOthers/ ./Data/
9. Once all this is done, STOP the instance and add a GPU (Tesla T4/P4) and start the instance again
10. Now, we can do the actual training, and stop and start as and when necessary. The environment is setup!
