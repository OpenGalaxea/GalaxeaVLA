# Out-of-the-Box Ready is All Your Need: User Guides for Pick Up Anything Demo

**Pick Up Anything Demo** is a demo that shows how to combine a host computer, a robot (R1Lite) and an easy-to-use APP on an Android device (like a tablet) to implement a pick-up anything task. 

What you should expect:

* Easily speck up your instructions to our APP on your Android device.

* An image with a bounding box and one sentence will be return, showing how the robot understands your instructions.

* The robot will follow and execute the instructions you give, in a **fast**, **precise** and **smooth** pattern.

## Overall Communication Framework

You should first make sure all your devices are connected as the following diagram shows:

<p align="center">
  <img src="../assets/communication_framework.png" alt="Communication Framework" width="700"/>
</p>

## Environment Setup on Host Computer

Note that the following guideline is tested in the country of China, if you are oversea, please skip some of the steps about network setting.

### 1. Docker Insallation

#### 1.1 Update the apt package index

```bash
sudo apt update
```

#### 1.2 Install the dependent packages
```bash
sudo apt install apt-transport-https ca-certificates curl gnupg2 software-properties-common
```

#### 1.3 Add Docker's official GPG key

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

- Expected Output: OK

#### 1.4 Official installation after prompting OK

```bash
sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"
```

#### 1.5 Install the latest version of Docker Engine-Community
```bash
sudo apt install docker-ce
```

#### 1.6 Add the user to the new docker group

- If you want to access Docker without sudo, simply enter the following commands, which mean adding the user to the new Docker group, restarting Docker, and switching the current session to the new group. 

```bash
sudo groupadd docker

sudo gpasswd -a ${USER} docker

sudo service docker restart

newgrp - docker
```

#### 1.7 Installation is now complete

-  You can enter `sudo docker --version` or `sudo docker run hello-world` to test if the installation was successful! 

Reference Link: [How to Install and Use Docker on Ubuntu 20.04 System_Install Docker on Ubuntu 20.04 - CSDN Blog](https://blog.csdn.net/qq_38156743/article/details/130401015)


### 2. CUDA12.8 Installation

#### 2.1 Installation Script Download & Run

Run the following command (refer to the [official website](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local))

```bash
cd ~/Downloads
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run
```

#### 2.2 Check if the installation is complete 

see if `cuda-12.8/` exists.

```
ls /usr/local/cuda-12.8/
```

### 3. NVIDIA Container Toolkit Installation

#### 3.1 Install system dependencies 

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2
```

#### 3.2  Configure the official software repository 

1. Import GPG Key 

     ```bash
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
     ```

    Note:

    If failure persists, please confirm whether `export https_proxy=xxx http_proxy=xxx all_proxy=xxx` has been set in advance.


2. Add software source

     ```bash
     ARCH=$(dpkg --print-architecture)
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list <<EOF
     deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$ARCH /
     #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/experimental/deb/$ARCH /
     EOF
     ```

3. Update the package list 

     ```bash
     sudo apt-get update
     ```

    - This step is crucial, please check whether the links including https://nvidia.github.io/libnvidia-container can be accessed or hit. If it fails, you must stop at this step until it succeeds 
    - If `curl -v  https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages` can get normal output, the website is accessible. Refer to the [following method](./pp_ug_appendix1.md) to add a proxy to apt to resolve the issue.

#### 3.3 Install NVIDIA Container Toolkit 

```bash
sudo apt-get install -y nvidia-container-toolkit 
```

#### 3.4 Configure the container runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

- Expected Result:
    <p align="center">
    <img src="../assets/pp_ug_image1.png" alt="pp_ug_image1" width="700"/>
    </p>

### 3.5 Restart the Docker service

```bash
sudo systemctl restart docker
```

Reference Link:
  - 3.1-3.2 refer to: [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - 3.3-3.5 refer to: [Resolve the exception when Docker running containers use GPU resources: could not select device driver "" with capabilities: [[gpu]]_error respons](https://blog.csdn.net/qq_38628046/article/details/136312844)



## One-click Startup

### 1. Start the Robot

1. (Robot) Turn on the robot

2. (Host) ssh to the robot and one-click startup

```bash
ssh r1lite@10.42.0.xx
./model_start.sh
```


### 2. Start the VLM-VLA-EHI System

1. (Host) Build the docker image





