# Steps to use GCP for Training:

1. Make a VM Instance and do NOT allocate a Tesla T4/P4 GPU:

### Config of VM Instance:
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

### Installing drivers

- Get the NVIDIA signing key that is used to sign packages.

```
curl -o cudatools.asc http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
```

- Verify the integrity of the downloaded key. It should return with
  `cudatools.asc: OK`.

```
$ CUDA_KEY_CHECKSUM=47217c49dcb9e47a8728b354450f694c9898cd4a126173044a69b1e9ac0fba96
$ echo "$CUDA_KEY_CHECKSUM *cudatools.asc" | sha256sum --check --strict

cudatools.asc: OK
```

- Or verify the fingerprint of the downloaded key. It should match
  `AE09 FE4B BD22 3A84 B2CC  FCE3 F60F 4B3D 7FA2 AF80`.

```
$ gpg --with-fingerprint --with-colons cudatools.asc | grep "^fpr:" | cut -d: -f 10

AE09FE4BBD223A84B2CCFCE3F60F4B3D7FA2AF80
```

- Import the CUDA signing key:
```bash
$ apt-key add cudatools.asc
OK
```
- Download the CUDA installer, again, verify its integrity, and add the repository
  to the system using `dpkg`:
```bash
$ curl -sO http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.105-1_amd64.deb

$ echo "2711e92b362972d6ba16dd1e1410f56ecc808831c3ae42b2b9fa98cea086f146 *cuda-repo-ubuntu1804_10.1.105-1_amd64.deb" | sha256sum --check --strict

cuda-repo-ubuntu1804_10.1.105-1_amd64.deb: OK

$ sudo dpkg -i cuda-repo-ubuntu1804_10.1.105-1_amd64.deb

...
OK
```

- Finally, install CUDA, which includes the NVIDIA drivers:
```bash
$ sudo apt-get update && sudo apt-get install -y cuda

...
```

This could take a while.

- Reboot the machine: `sudo reboot`

When the installation has finished, verify that the driver is installed, and initialised
properly using the `nvidia-smi` tool.

```bash
$ nvidia-smi

...
```

### Tuning the GPUs

A little performance tuning. Before you can change the application clocks you need
to put the GPUs in 'persistence mode'.

- Optionally, allow non-admin/root users to change the application clocks using
  the following command.
```
sudo nvidia-smi --applications-clocks-permission=UNRESTRICTED
```

- Enable 'persistence mode' with the following command.
```bash
sudo nvidia-smi --persistence-mode=ENABLED
```

- You must set both the memory and graphics clock, because the clock rates are
  tied to a specific memory clock rate. Set the clock rates using the following
  command (you can query the supported application clocks using
  `nvidia-smi  -q -i 0 -d SUPPORTED_CLOCKS`).
```
sudo nvidia-smi --applications-clocks=2505,875
```


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
