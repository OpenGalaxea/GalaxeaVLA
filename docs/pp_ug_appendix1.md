# Permanently Configure Proxy for APT

## Step 1

Edit the file /etc/apt/apt.conf.d/95proxies (create it if it does not exist):

```bash
sudo nano /etc/apt/apt.conf.d/95proxies
```

## Step 2

Write the following content into the file (adjust the port and protocol according to your proxy):

```bash
Acquire {
  HTTP::Proxy "http://127.0.0.1:7890";
  HTTPS::Proxy "http://127.0.0.1:7890";
}
```

Notes:

- If you are using tools like Clash or V2Ray, please confirm the port (commonly 7890 or 7897).

- If your proxy requires authentication, use the format: `http://username:password@127.0.0.1:7890`

## Step 3

Save and exit (Ctrl + O, Enter, then Ctrl + X).

## Step 4
Then run:

```bash
sudo apt-get update
```

## Reference
https://chatgpt.com/share/6900b034-0890-8001-9b26-584b0a8df27d