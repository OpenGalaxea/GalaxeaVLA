# Out-of-the-Box is All Your Need: User Guides for Pick Up Anything Demo

**Pick Up Anything Demo** is a demo that shows how to combine a host computer, a robot (R1Lite) and an easy-to-use APP on an Android device (like a tablet) to implement a pick-up anything task. 

What you should expect:

* Easily speck up your instructions to our APP on your Android device.

* An image with a bounding box and one sentence will be return, showing how the robot understands your instructions.

* The robot will follow and execute the instructions you give, in a **fast**, **precise** and **smooth** pattern.

## Overall Communication Framework

You should first make sure all your devices are connected as the following diagram shows:

<p align="center">
  <img src="assets/pick_up_anything_demo/communication_framework.png" alt="Communication Framework" width="700"/>
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
    - If `curl -v  https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages` can get normal output, the website is accessible. Refer to the [following method](./pp-permanently_configure_proxy_for_apt.md) to add a proxy to apt to resolve the issue.

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
    <img src="assets/pick_up_anything_demo/pp_ug_image1.png" alt="pp_ug_image1" width="700"/>
    </p>

### 3.5 Restart the Docker service

```bash
sudo systemctl restart docker
```

Reference Link:
  - 3.1-3.2 refer to: [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  - 3.3-3.5 refer to: [Resolve the exception when Docker running containers use GPU resources: could not select device driver "" with capabilities: [[gpu]]_error respons](https://blog.csdn.net/qq_38628046/article/details/136312844)



## One-Click Startup

### 1. Preparation

#### 1.1 Machine Preparation

1. (Robot&Host) Set up Host and Robot network configuration

   Follow the [Network Setup Guide](./pp-how_to_set_up_the_network_between_the_host_and_robot.md) to set up the network between the host computer and the robot (R1Lite).

2. (Host) Ensure connection between Host and Robot

   Make sure that the host computer can successfully ping the robot. You can use the following command to test the connection:

   ```bash
   arp -a
   ping 10.42.0.<ROBOT_PORT>
   ssh r1lite@10.42.0.<ROBOT_PORT>
   ```

   - replace `<ROBOT_PORT>` with the actual IP address of your robot.

3. (Host) Set up ROS2 discovery

   Follow the [ROS2 Discovery Setup Guide](./pp-ros2_discovery_setup.md) to set up the ROS2 discovery on the robot.

4. (Host) Robot one-click startup script creation
    
    Copy (e.g., `scp`) the [one-click startup script](./supports/pick_up_anything_demo/model_test.sh) into the robot (R1Lite)'s home directory, and make it executable by running:

    ```bash
    chmod +x ~/model_test.sh
    ```

#### 1.2 Docker Image Build

1. (Host) Download our [Official Dockerfile and Related Files on Huggingface](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/g0plus_dockerfile), and build the Docker image by following the README provided in that repository.

2. (Host) Make sure Docker image `g0plus:ros2_v1-trt` exists on your Host by running:

   ```bash
   docker images
   ```



#### 1.3 Host one-Click Script Download & Folder Structure

(Host) You can download the one-click startup scripts [g0plus_hs_start_v1.sh](./supports/pick_up_anything_demo/g0plus_hs_start_v1.sh) and [docker_g0plus_hs_start_v1.sh](./supports/pick_up_anything_demo/docker_g0plus_hs_start_v1.sh) from our Repo, make sure your folder structure on Host is as follows:

```
~/
  ├── g0plus_ros2/
      ├── data                            # Files to be linked to the docker container
      |   ├── G0Plus_PP_CKPT/             # G0Plus weights folder
      |   |   ├── prefill.fp16.engine     # G0Plus prefill engine file
      |   |   └── decode.fp16.engine      # G0Plus decode engine file
      |   |   └── ...            
      |   ├── google/
      |   |   └── paligemma-3b-pt-224/                # paligemma weights file
      |   └── docker_g0plus_hs_start_v1.sh            # One-click startup script inside docker container  
      └── g0plus_hs_start_v1.sh           # One-click startup script for host machine
```
- Note that: 

  - You need to create the `g0plus_ros2/` and `data/` folder.
  - You can ignore `G0Plus_PP_CKPT/` and `google/` folder in this step, which will be created in the next steps.

#### 1.4 G0Plus Pick-Up-Anything Checkpoint Download

(Host) Download the checkpoint folder [G0Plus_PP_CKPT/ on Huggingface](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/g0plus_pick_up_anything_checkpoint), and place it in the `data/` folder created in the previous step.

#### 1.5 Paligemma Checkpoint Download

(Host) Download the paligemma weights [paligemma-3b-pt-224 on Huggingface](https://huggingface.co/google/paligemma-3b-pt-224), and place it in the `data/google/` folder created in the previous step.


### 2. Startup Robot

1. (Host) SSH into the robot

   ```bash
   ssh r1lite@10.42.0.<ROBOT_PORT>
   ```

   - replace `<ROBOT_PORT>` with the actual IP address of your robot.

2. (Host) Run the one-click startup script on Robot:

   ```bash
   cd ~
   ./model_test.sh
   ```


### 3. Startup G0Plus Hierarchical System

(Host) Run the one-click startup script on Host:

```bash
cd ~/g0plus_ros2
./g0plus_hs_start_v1.sh
```
- Note that there are 6 interactive options in the script:
    1. Select execution mode: choose either "Initial execution" or "Second execution". Note that if choosing "Second execution", the previously entered Gemini and Qwen API keys will be used directly without prompting for input again, and skip steps 3 and 4.

    2. Choose whether to enable the Qwen model: select "Disabled" or "Enable".

    3. Enter the API key of Gemini (only required if Qwen model is enabled in the previous step), default is "NaN".

    4. Enter the API key of Qwen (only required if Qwen model is enabled in step 2), default is "NaN". After this step, the terminal will print the currently used Gemini and Qwen API keys for confirmation.

    5. Enter the robot port number (PORT), default is "180" (only supports ports starting with `10.42.0.`).

    6. Enter the name of the Docker container to start, default is `g0plus_ros2_v1`. After this final step, the terminal will enter the tmux session inside the Docker container.